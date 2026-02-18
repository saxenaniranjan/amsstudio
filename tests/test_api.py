from __future__ import annotations

import json
from pathlib import Path
from io import BytesIO

from fastapi.testclient import TestClient

from app import app
from ticket_analytics import api_server as api_server_module


def _to_excel_bytes(df) -> bytes:
    stream = BytesIO()
    with __import__("pandas").ExcelWriter(stream, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    stream.seek(0)
    return stream.read()


def _to_csv_bytes(df, sep: str = ",", encoding: str = "utf-8") -> bytes:
    return df.to_csv(index=False, sep=sep).encode(encoding)


def test_api_end_to_end(raw_ticket_df) -> None:
    client = TestClient(app)

    excel_bytes = _to_excel_bytes(raw_ticket_df)

    process_response = client.post(
        "/api/sessions/process",
        data={
            "workspace_name": "PRD Workspace",
            "user_mapping": json.dumps({}),
            "sla_threshold_hours": json.dumps({"P1": 4, "P2": 8, "P3": 24, "P4": 72}),
        },
        files=[
            (
                "files",
                (
                    "tickets.xlsx",
                    excel_bytes,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ),
            )
        ],
    )

    assert process_response.status_code == 200
    payload = process_response.json()
    assert payload["rows"] == len(raw_ticket_df)
    session_id = payload["session_id"]

    overview_response = client.post(
        f"/api/sessions/{session_id}/overview",
        json={"filters": {}, "start_date": None, "end_date": None},
    )
    assert overview_response.status_code == 200
    overview = overview_response.json()
    assert "cards" in overview
    assert "report" in overview

    graph_response = client.post(
        f"/api/sessions/{session_id}/graphs",
        json={"graph_id": "graph_1", "filters": {}, "start_date": None, "end_date": None},
    )
    assert graph_response.status_code == 200
    assert "figure" in graph_response.json()

    query_response = client.post(
        f"/api/sessions/{session_id}/query",
        json={"mode": "enabler", "query": "show graph 2", "filters": {}, "start_date": None, "end_date": None},
    )
    assert query_response.status_code == 200
    query_payload = query_response.json()
    assert "text" in query_payload

    mttr_query_response = client.post(
        f"/api/sessions/{session_id}/query",
        json={
            "mode": "enabler",
            "query": "give me mttr line chart over last 3 months for P2 tickets only",
            "filters": {},
            "start_date": None,
            "end_date": None,
        },
    )
    assert mttr_query_response.status_code == 200
    mttr_payload = mttr_query_response.json()
    assert mttr_payload["kind"] == "chart"
    assert mttr_payload["figure"] is not None
    assert mttr_payload["agent_trace"] is not None

    autonomous_response = client.post(
        f"/api/sessions/{session_id}/query",
        json={"mode": "autonomous", "query": "", "filters": {}, "start_date": None, "end_date": None},
    )
    assert autonomous_response.status_code == 200
    auto_payload = autonomous_response.json()
    assert "findings" in auto_payload

    composite_response = client.post(
        f"/api/sessions/{session_id}/composite",
        json={
            "x_col": "team",
            "y_col": "mttr_hours",
            "chart_type": "bar",
            "color_col": None,
            "filters": {},
            "start_date": None,
            "end_date": None,
        },
    )
    assert composite_response.status_code == 200
    assert "figure" in composite_response.json()

    export_response = client.get(f"/api/sessions/{session_id}/export/summary")
    assert export_response.status_code == 200
    assert "cards" in export_response.json()


def test_api_process_semicolon_csv(raw_ticket_df) -> None:
    client = TestClient(app)
    csv_bytes = _to_csv_bytes(raw_ticket_df, sep=";")

    response = client.post(
        "/api/sessions/process",
        data={
            "workspace_name": "CSV Workspace",
            "user_mapping": json.dumps({}),
            "sla_threshold_hours": json.dumps({"P1": 4, "P2": 8, "P3": 24, "P4": 72}),
        },
        files=[("files", ("tickets.csv", csv_bytes, "text/csv"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["rows"] == len(raw_ticket_df)


def test_api_process_invalid_file_returns_400() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/sessions/process",
        data={
            "workspace_name": "Bad Workspace",
            "user_mapping": json.dumps({}),
            "sla_threshold_hours": json.dumps({"P1": 4, "P2": 8, "P3": 24, "P4": 72}),
        },
        files=[("files", ("broken.xlsx", b"not-an-excel-file", "application/octet-stream"))],
    )

    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "No valid rows found in uploaded files" in detail


def test_api_preview_columns_invalid_file_returns_400() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/columns/preview",
        files=[("files", ("broken.xlsx", b"not-an-excel-file", "application/octet-stream"))],
    )

    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "No valid rows found in uploaded files" in detail


def test_session_restores_from_disk_after_memory_reset(raw_ticket_df, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TICKETX_SESSION_CACHE_DIR", str(tmp_path))
    client = TestClient(app)
    excel_bytes = _to_excel_bytes(raw_ticket_df)

    process_response = client.post(
        "/api/sessions/process",
        data={
            "workspace_name": "Restore Workspace",
            "user_mapping": json.dumps({}),
            "sla_threshold_hours": json.dumps({"P1": 4, "P2": 8, "P3": 24, "P4": 72}),
        },
        files=[
            (
                "files",
                (
                    "tickets.xlsx",
                    excel_bytes,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ),
            )
        ],
    )
    assert process_response.status_code == 200
    session_id = process_response.json()["session_id"]

    # Simulate API memory reset/restart.
    api_server_module.SESSION_STORE.clear()

    overview_response = client.post(
        f"/api/sessions/{session_id}/overview",
        json={"filters": {}, "start_date": None, "end_date": None},
    )
    assert overview_response.status_code == 200
    assert session_id in api_server_module.SESSION_STORE


def test_cache_dir_defaults_to_tmp_on_vercel(monkeypatch) -> None:
    monkeypatch.delenv("TICKETX_SESSION_CACHE_DIR", raising=False)
    monkeypatch.setenv("VERCEL", "1")

    cache_path = api_server_module._cache_dir()
    assert cache_path == Path("/tmp/ticketx_session_cache")
    assert cache_path.exists()


def test_api_process_succeeds_when_persist_session_fails(raw_ticket_df, monkeypatch) -> None:
    client = TestClient(app)
    excel_bytes = _to_excel_bytes(raw_ticket_df)

    monkeypatch.setattr(
        api_server_module,
        "_persist_session",
        lambda _session: "Session cache persistence unavailable: read-only filesystem",
    )

    response = client.post(
        "/api/sessions/process",
        data={
            "workspace_name": "Warning Workspace",
            "user_mapping": json.dumps({}),
            "sla_threshold_hours": json.dumps({"P1": 4, "P2": 8, "P3": 24, "P4": 72}),
        },
        files=[
            (
                "files",
                (
                    "tickets.xlsx",
                    excel_bytes,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ),
            )
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload.get("session_warning")
    assert "persistence unavailable" in payload["session_warning"].lower()
