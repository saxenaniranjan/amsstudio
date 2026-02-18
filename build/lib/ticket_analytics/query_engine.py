"""Natural-language style query slicing over enriched ticket data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .constants import CHART_KEYWORDS
from .insights import build_insight_report


@dataclass
class QueryResult:
    kind: str
    text: str
    data: pd.DataFrame | None = None
    chart: dict[str, Any] | None = None


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _detect_chart_type(query: str) -> str | None:
    lowered = query.lower()
    for keyword, chart in CHART_KEYWORDS.items():
        if keyword in lowered:
            return chart
    if any(x in lowered for x in ["graph", "plot", "visualize", "chart"]):
        return "bar"
    return None


def _detect_metric(query: str) -> tuple[str, str]:
    q = query.lower()
    if any(k in q for k in ["mttr", "mean time to resolve", "resolution time"]):
        return "mttr_hours", "avg_mttr_hours"
    if any(k in q for k in ["breach rate", "sla breach rate"]):
        return "is_sla_breached", "breach_rate"
    if any(k in q for k in ["breached", "sla breach", "sla violations"]):
        return "is_sla_breached", "breached_tickets"
    if any(k in q for k in ["open ticket", "backlog"]):
        return "is_open", "open_tickets"
    if any(k in q for k in ["performance", "team performance"]):
        return "team_performance_index", "team_performance_index"
    return "ticket_id", "ticket_count"


def _detect_dimension(query: str, df: pd.DataFrame) -> str | None:
    q = query.lower()
    explicit_map = {
        "team": "team",
        "category": "category_derived",
        "business function": "business_function_derived",
        "priority": "priority",
        "status": "status",
        "service": "service",
    }
    for key, value in explicit_map.items():
        if key in q and value in df.columns:
            return value

    if any(k in q for k in ["daily", "per day", "over time", "trend", "timeline", "weekly", "monthly"]):
        return "created_at"

    for candidate in ["team", "category_derived", "business_function_derived", "priority"]:
        if candidate in df.columns:
            return candidate
    return None


def _time_grain(query: str) -> str:
    q = query.lower()
    if "monthly" in q or "per month" in q:
        return "M"
    if "weekly" in q or "per week" in q:
        return "W"
    return "D"


def _extract_filters(query: str, df: pd.DataFrame) -> dict[str, list[str]]:
    query_norm = f" {_normalize_text(query)} "
    filters: dict[str, list[str]] = {}

    candidate_columns = ["team", "category_derived", "business_function_derived", "priority", "status", "service"]
    for column in candidate_columns:
        if column not in df.columns:
            continue
        unique_values = df[column].dropna().astype(str).unique()
        matched: list[str] = []
        for value in unique_values:
            token = _normalize_text(value)
            if not token:
                continue
            if f" {token} " in query_norm:
                matched.append(value)
        if matched:
            filters[column] = matched

    return filters


def _apply_filters(df: pd.DataFrame, filters: dict[str, list[str]]) -> pd.DataFrame:
    filtered = df.copy()
    for column, allowed in filters.items():
        filtered = filtered[filtered[column].astype(str).isin([str(x) for x in allowed])]
    return filtered


def _aggregate(
    df: pd.DataFrame,
    metric_source: str,
    metric_name: str,
    dimension: str | None,
    query: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    if dimension == "created_at":
        granularity = _time_grain(query)
        time_frame = df.dropna(subset=["created_at"]).copy()
        if time_frame.empty:
            return pd.DataFrame()
        time_frame["period"] = time_frame["created_at"].dt.to_period(granularity).dt.to_timestamp()
        group_col = "period"
    else:
        group_col = dimension if dimension else None

    def _metric_agg(frame: pd.DataFrame) -> pd.DataFrame:
        if metric_name == "ticket_count":
            return frame.groupby(group_col, dropna=False).size().rename(metric_name).reset_index()
        if metric_name == "avg_mttr_hours":
            return frame.groupby(group_col, dropna=False)["mttr_hours"].mean().rename(metric_name).reset_index()
        if metric_name == "breach_rate":
            return frame.groupby(group_col, dropna=False)["is_sla_breached"].mean().rename(metric_name).reset_index()
        if metric_name == "breached_tickets":
            return frame.groupby(group_col, dropna=False)["is_sla_breached"].sum().rename(metric_name).reset_index()
        if metric_name == "open_tickets":
            return frame.groupby(group_col, dropna=False)["is_open"].sum().rename(metric_name).reset_index()
        if metric_name == "team_performance_index":
            return (
                frame.groupby(group_col, dropna=False)["team_performance_index"]
                .mean()
                .rename(metric_name)
                .reset_index()
            )
        return frame.groupby(group_col, dropna=False).size().rename("ticket_count").reset_index()

    if group_col is None:
        result = pd.DataFrame(
            [{metric_name: _metric_agg(df.assign(_single_group="all")).iloc[0][metric_name]}]
        )
    else:
        result = _metric_agg(df)

    sort_col = metric_name
    if sort_col in result.columns and group_col != "period":
        result = result.sort_values(sort_col, ascending=False)

    if "breach_rate" in result.columns:
        result["breach_rate"] = (result["breach_rate"] * 100).round(2)

    if "avg_mttr_hours" in result.columns:
        result["avg_mttr_hours"] = result["avg_mttr_hours"].round(2)

    if "team_performance_index" in result.columns:
        result["team_performance_index"] = result["team_performance_index"].round(2)

    return result


def _format_summary(aggregated: pd.DataFrame, metric_name: str, dimension: str | None, row_count: int) -> str:
    if aggregated.empty:
        return "No matching records were found for this query."

    if dimension is None:
        value = aggregated.iloc[0][metric_name]
        return f"Result: {metric_name} = {value}. Records analyzed: {row_count}."

    top_row = aggregated.iloc[0]
    dim_value = top_row[dimension if dimension != "created_at" else "period"]
    metric_value = top_row[metric_name]
    return (
        f"Computed {metric_name} by {dimension}. Top segment: {dim_value} with value {metric_value}. "
        f"Records analyzed: {row_count}."
    )


def answer_query(query: str, df: pd.DataFrame) -> QueryResult:
    text = query.strip()
    if not text:
        report = build_insight_report(df)
        metrics = report["metrics"]
        return QueryResult(
            kind="text",
            text=(
                f"Dataset has {metrics['total_tickets']} tickets, average MTTR {metrics['avg_mttr_hours']}h, "
                f"and SLA breach rate {round(metrics['breach_rate'] * 100, 2)}%."
            ),
        )

    lowered = text.lower()
    if any(k in lowered for k in ["recommend", "improve", "insight", "what should", "action"]):
        report = build_insight_report(df)
        recs = report["recommendations"]
        body = "\n".join([f"- {rec}" for rec in recs])
        return QueryResult(kind="text", text=f"Recommended actions:\n{body}")

    chart_type = _detect_chart_type(text)
    metric_source, metric_name = _detect_metric(text)
    dimension = _detect_dimension(text, df)

    filters = _extract_filters(text, df)
    filtered_df = _apply_filters(df, filters)

    aggregated = _aggregate(filtered_df, metric_source, metric_name, dimension, text)

    if "top" in lowered and not aggregated.empty and dimension is not None:
        match = re.search(r"top\s+(\d+)", lowered)
        limit = int(match.group(1)) if match else 5
        aggregated = aggregated.head(limit)

    summary = _format_summary(aggregated, metric_name, dimension, len(filtered_df))

    chart_payload: dict[str, Any] | None = None
    if chart_type and not aggregated.empty and dimension is not None:
        x_col = "period" if dimension == "created_at" else dimension
        chart_payload = {
            "type": chart_type,
            "x": x_col,
            "y": metric_name,
            "title": f"{metric_name} by {x_col}",
        }

    kind = "chart" if chart_payload else "table"
    return QueryResult(kind=kind, text=summary, data=aggregated, chart=chart_payload)
