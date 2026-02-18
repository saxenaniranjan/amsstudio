"""FastAPI server for Ticket-X analytics."""

from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import plotly.io as pio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .agentic import build_summary_export, run_autonomous_mode, run_enabler_mode
from .graph_catalog import build_composite_graph, build_prd_graph, build_word_cloud_plotly
from .insights import build_insight_report
from .pipeline import run_ticket_pipeline
from .prd_metrics import build_prd_cards
from .workspace import build_upload_history_entry, suggest_global_filters, suggest_mapping_candidates


@dataclass
class SessionData:
    session_id: str
    workspace_name: str
    created_at: str
    enriched_df: pd.DataFrame
    upload_history: list[dict[str, Any]]


SESSION_STORE: dict[str, SessionData] = {}


def _cache_dir() -> Path:
    path = Path(os.getenv("TICKETX_SESSION_CACHE_DIR", ".session_cache"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _session_paths(session_id: str) -> tuple[Path, Path]:
    base = _cache_dir()
    return base / f"{session_id}.meta.json", base / f"{session_id}.df.pkl"


def _persist_session(session: SessionData) -> None:
    meta_path, df_path = _session_paths(session.session_id)
    metadata = {
        "session_id": session.session_id,
        "workspace_name": session.workspace_name,
        "created_at": session.created_at,
        "upload_history": session.upload_history,
    }
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    session.enriched_df.to_pickle(df_path)


def _load_session_from_disk(session_id: str) -> SessionData | None:
    meta_path, df_path = _session_paths(session_id)
    if not meta_path.exists() or not df_path.exists():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        df = pd.read_pickle(df_path)
        return SessionData(
            session_id=session_id,
            workspace_name=str(metadata.get("workspace_name", "Recovered Workspace")),
            created_at=str(metadata.get("created_at", datetime.now().isoformat())),
            enriched_df=df,
            upload_history=metadata.get("upload_history", []),
        )
    except Exception:
        return None


class FilterPayload(BaseModel):
    filters: dict[str, list[str]] = Field(default_factory=dict)
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class GraphPayload(FilterPayload):
    graph_id: str


class QueryPayload(FilterPayload):
    query: str = ""
    mode: str = "enabler"


class CompositePayload(FilterPayload):
    x_col: str
    y_col: str
    chart_type: str
    color_col: Optional[str] = None


def _parse_json(text: str, default: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return default


def _read_csv_bytes(payload: bytes) -> pd.DataFrame:
    attempts: list[str] = []
    encodings = ["utf-8", "utf-8-sig", "utf-16", "latin-1"]
    separators: list[str | None] = [None, ",", ";", "\t", "|"]

    for encoding in encodings:
        for separator in separators:
            try:
                frame = pd.read_csv(
                    BytesIO(payload),
                    sep=separator,
                    engine="python",
                    encoding=encoding,
                )
                if frame.empty and len(frame.columns) == 0:
                    continue
                return frame
            except Exception as exc:
                attempts.append(f"encoding={encoding}, sep={separator!r}: {exc}")

    sample = "; ".join(attempts[:3])
    raise ValueError(f"Unable to parse CSV payload. Attempts failed: {sample}")


def _read_upload_file(file: UploadFile) -> pd.DataFrame:
    payload = file.file.read()
    if not payload:
        return pd.DataFrame()

    name = (file.filename or "").lower()
    try:
        if name.endswith((".csv", ".txt")):
            return _read_csv_bytes(payload)

        if name.endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
            return pd.read_excel(BytesIO(payload), engine="openpyxl")

        if name.endswith(".xls"):
            return pd.read_excel(BytesIO(payload), engine="xlrd")

        if name.endswith(".xlsb"):
            return pd.read_excel(BytesIO(payload), engine="pyxlsb")

        # Generic fallback for unknown extensions.
        return pd.read_excel(BytesIO(payload))
    except Exception as exc:
        raise ValueError(f"Failed to parse '{file.filename}': {exc}") from exc


def _df_to_records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    frame = df.copy()
    if limit is not None:
        frame = frame.head(limit)
    for col in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            frame[col] = frame[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _figure_to_json(figure: Any) -> dict[str, Any]:
    return json.loads(pio.to_json(figure, validate=False))


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return _df_to_records(value)
    if isinstance(value, pd.Series):
        if pd.api.types.is_datetime64_any_dtype(value):
            return [x.isoformat() if pd.notna(x) else None for x in value]
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _apply_filters(df: pd.DataFrame, payload: FilterPayload) -> pd.DataFrame:
    filtered = df.copy()

    for col, values in payload.filters.items():
        if col not in filtered.columns:
            continue
        if not values:
            continue
        allowed = {str(v) for v in values}
        filtered = filtered[filtered[col].astype(str).isin(allowed)]

    if payload.start_date and "created_at" in filtered.columns:
        try:
            start = pd.Timestamp(payload.start_date)
            filtered = filtered[filtered["created_at"] >= start]
        except Exception:
            pass

    if payload.end_date and "created_at" in filtered.columns:
        try:
            end = pd.Timestamp(payload.end_date)
            filtered = filtered[filtered["created_at"] <= end]
        except Exception:
            pass

    return filtered


def _get_session(session_id: str) -> SessionData:
    session = SESSION_STORE.get(session_id)
    if session is None:
        restored = _load_session_from_disk(session_id)
        if restored is None:
            raise HTTPException(status_code=404, detail="Session not found")
        SESSION_STORE[session_id] = restored
        session = restored
    return session


def _format_file_errors(file_errors: list[dict[str, str]], limit: int = 3) -> str:
    if not file_errors:
        return ""
    items = file_errors[:limit]
    text = "; ".join([f"{item['file_name']}: {item['error']}" for item in items])
    remaining = len(file_errors) - len(items)
    if remaining > 0:
        text += f"; +{remaining} more"
    return text


def _overview_payload(df: pd.DataFrame) -> dict[str, Any]:
    cards = build_prd_cards(df)
    report = build_insight_report(df)
    filters = suggest_global_filters(df)

    filter_values: dict[str, list[str]] = {}
    for col in filters:
        options = sorted(df[col].dropna().astype(str).unique().tolist())
        filter_values[col] = options[:200]

    min_date = None
    max_date = None
    if "created_at" in df.columns and df["created_at"].notna().any():
        min_date = pd.Timestamp(df["created_at"].min()).isoformat()
        max_date = pd.Timestamp(df["created_at"].max()).isoformat()

    return {
        "cards": _json_safe(cards),
        "report": _json_safe(report),
        "columns": list(df.columns),
        "sample_rows": _df_to_records(df, limit=25),
        "filter_columns": filters,
        "filter_values": filter_values,
        "date_range": {"min": min_date, "max": max_date},
    }


def create_app() -> FastAPI:
    app = FastAPI(title="Ticket-X Analytics API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/sessions/process")
    async def process_session(
        files: list[UploadFile] = File(...),
        workspace_name: str = Form("Default Workspace"),
        user_mapping: str = Form("{}"),
        sla_threshold_hours: str = Form("{}"),
    ) -> JSONResponse:
        frames: list[pd.DataFrame] = []
        upload_history: list[dict[str, Any]] = []
        file_errors: list[dict[str, str]] = []

        for file in files:
            try:
                frame = _read_upload_file(file)
            except Exception as exc:
                file_errors.append({"file_name": file.filename or "unknown", "error": str(exc)})
                continue

            if frame.empty:
                file_errors.append(
                    {
                        "file_name": file.filename or "unknown",
                        "error": "File parsed but contains no data rows",
                    }
                )
                continue
            frames.append(frame)
            entry = build_upload_history_entry(file.filename, len(frame))
            upload_history.append(entry.__dict__)

        if not frames:
            suffix = _format_file_errors(file_errors)
            detail = "No valid rows found in uploaded files."
            if suffix:
                detail = f"{detail} {suffix}"
            raise HTTPException(status_code=400, detail=detail)

        raw_df = pd.concat(frames, ignore_index=True)
        mapping = _parse_json(user_mapping, {})
        sla_map = _parse_json(sla_threshold_hours, {})

        enriched = run_ticket_pipeline(raw_df, user_mapping=mapping, sla_threshold_hours=sla_map)

        session_id = str(uuid.uuid4())
        SESSION_STORE[session_id] = SessionData(
            session_id=session_id,
            workspace_name=workspace_name,
            created_at=datetime.now().isoformat(),
            enriched_df=enriched,
            upload_history=upload_history,
        )
        _persist_session(SESSION_STORE[session_id])

        payload = {
            "session_id": session_id,
            "workspace_name": workspace_name,
            "rows": int(len(enriched)),
            "columns": list(enriched.columns),
            "upload_history": upload_history,
            "file_errors": file_errors,
            "overview": _overview_payload(enriched),
        }
        return JSONResponse(content=payload)

    @app.post("/api/columns/preview")
    async def preview_columns(files: list[UploadFile] = File(...)) -> JSONResponse:
        frames: list[pd.DataFrame] = []
        file_errors: list[dict[str, str]] = []
        for file in files:
            try:
                frame = _read_upload_file(file)
            except Exception as exc:
                file_errors.append({"file_name": file.filename or "unknown", "error": str(exc)})
                continue
            if frame.empty:
                file_errors.append(
                    {
                        "file_name": file.filename or "unknown",
                        "error": "File parsed but contains no data rows",
                    }
                )
                continue
            frames.append(frame)

        if not frames:
            suffix = _format_file_errors(file_errors)
            detail = "No valid rows found in uploaded files."
            if suffix:
                detail = f"{detail} {suffix}"
            raise HTTPException(status_code=400, detail=detail)

        raw_df = pd.concat(frames, ignore_index=True)
        columns = [str(col) for col in raw_df.columns]
        suggestions = suggest_mapping_candidates(raw_df)
        return JSONResponse(content={"columns": columns, "mapping_suggestions": suggestions, "file_errors": file_errors})

    @app.post("/api/sessions/{session_id}/overview")
    def get_overview(session_id: str, payload: FilterPayload) -> JSONResponse:
        session = _get_session(session_id)
        filtered = _apply_filters(session.enriched_df, payload)
        return JSONResponse(content=_overview_payload(filtered))

    @app.post("/api/sessions/{session_id}/graphs")
    def get_graph(session_id: str, payload: GraphPayload) -> JSONResponse:
        session = _get_session(session_id)
        filtered = _apply_filters(session.enriched_df, payload)

        output = build_prd_graph(
            filtered,
            payload.graph_id,
            start=pd.Timestamp(payload.start_date) if payload.start_date else None,
            end=pd.Timestamp(payload.end_date) if payload.end_date else None,
        )

        return JSONResponse(
            content={
                "graph_id": output.graph_id,
                "title": output.title,
                "insight_hint": output.insight_hint,
                "figure": _figure_to_json(output.figure),
                "data": _df_to_records(output.data, limit=500),
            }
        )

    @app.post("/api/sessions/{session_id}/word-cloud")
    def get_word_cloud(session_id: str, payload: FilterPayload) -> JSONResponse:
        session = _get_session(session_id)
        filtered = _apply_filters(session.enriched_df, payload)
        fig = build_word_cloud_plotly(filtered)
        return JSONResponse(content={"figure": _figure_to_json(fig)})

    @app.post("/api/sessions/{session_id}/composite")
    def get_composite(session_id: str, payload: CompositePayload) -> JSONResponse:
        session = _get_session(session_id)
        filtered = _apply_filters(session.enriched_df, payload)
        fig = build_composite_graph(
            filtered,
            x_col=payload.x_col,
            y_col=payload.y_col,
            chart_type=payload.chart_type,
            color_col=payload.color_col,
        )
        return JSONResponse(content={"figure": _figure_to_json(fig)})

    @app.post("/api/sessions/{session_id}/query")
    def run_query(session_id: str, payload: QueryPayload) -> JSONResponse:
        session = _get_session(session_id)
        filtered = _apply_filters(session.enriched_df, payload)

        mode = payload.mode.lower().strip()
        if mode == "autonomous":
            response = run_autonomous_mode(filtered)
            return JSONResponse(
                content={
                    "mode": response.mode,
                    "summary": response.summary,
                    "findings": response.findings,
                    "recommendations": response.recommendations,
                    "suggested_questions": response.suggested_questions,
                }
            )

        result = run_enabler_mode(payload.query, filtered)
        figure_json = None
        chart = result.chart or {}
        if chart.get("type") == "plotly_figure" and chart.get("figure") is not None:
            figure_json = _figure_to_json(chart["figure"])

        return JSONResponse(
            content={
                "kind": result.kind,
                "text": result.text,
                "data": _df_to_records(result.data, limit=500) if result.data is not None else [],
                "chart": result.chart if chart.get("type") != "plotly_figure" else {"type": "plotly_figure"},
                "figure": figure_json,
                "agent_trace": _json_safe(result.agent_trace) if result.agent_trace is not None else None,
            }
        )

    @app.get("/api/sessions/{session_id}/enriched-preview")
    def enriched_preview(session_id: str, limit: int = 200) -> JSONResponse:
        session = _get_session(session_id)
        return JSONResponse(content={"rows": _df_to_records(session.enriched_df, limit=limit)})

    @app.get("/api/sessions/{session_id}/export/enriched.csv")
    def export_enriched_csv(session_id: str) -> StreamingResponse:
        session = _get_session(session_id)
        payload = session.enriched_df.to_csv(index=False).encode("utf-8")
        return StreamingResponse(
            iter([payload]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={session.workspace_name}_enriched.csv"},
        )

    @app.get("/api/sessions/{session_id}/export/summary")
    def export_summary(session_id: str) -> JSONResponse:
        session = _get_session(session_id)
        summary = build_summary_export(session.enriched_df)
        return JSONResponse(content=_json_safe(summary))

    return app
