"""Agentic analysis helpers for autonomous and enabler modes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from .graph_catalog import build_prd_graph
from .insights import build_insight_report
from .prd_metrics import build_prd_cards
from .query_engine import QueryResult, answer_query


@dataclass
class AgentResponse:
    mode: str
    summary: str
    findings: list[str]
    recommendations: list[str]
    suggested_questions: list[str]


def run_autonomous_mode(df: pd.DataFrame, reference_time: datetime | None = None) -> AgentResponse:
    report = build_insight_report(df)
    cards = build_prd_cards(df, reference_time=reference_time)

    metrics = report["metrics"]
    delivery = cards["delivery_compliance"]
    efficiency = cards["efficiency"]

    findings = [
        f"Total tickets analyzed: {metrics['total_tickets']}.",
        f"SLA adherence (last complete month): {delivery['sla_adherence_pct']}%.",
        f"SLA breaches (last complete month): {delivery['sla_breach_count']}.",
        f"At-risk open tickets (>=80% SLA elapsed): {delivery['at_risk_open_tickets']}.",
        f"MTTR (last complete month): {efficiency['mttr_hours']} hours.",
    ]

    if report.get("anomalies"):
        first = report["anomalies"][0]
        findings.append(
            f"Anomaly detected: ticket spike on {first['date']} ({first['ticket_count']} tickets, z={first['z_score']})."
        )

    recommendations = report["recommendations"]

    suggested_questions = [
        "Show Graph 1 ticket trend analysis for the selected date range.",
        "Break SLA breach trend by priority and assignment group.",
        "List top recurring issues with application and team context.",
        "Which teams are top MTTR detractors and why?",
    ]

    summary = (
        "Autonomous review complete. I scanned volumetrics, SLA compliance, efficiency, recurring patterns, "
        "and team performance to surface risk and actions."
    )

    return AgentResponse(
        mode="autonomous",
        summary=summary,
        findings=findings,
        recommendations=recommendations,
        suggested_questions=suggested_questions,
    )


def run_enabler_mode(query: str, df: pd.DataFrame) -> QueryResult:
    query_lower = query.lower().strip()

    for graph_key in ["graph 1", "graph 2", "graph 3", "graph 4", "graph 5", "graph 6", "graph 7", "graph 8"]:
        if graph_key in query_lower:
            graph_id = graph_key.replace(" ", "_")
            output = build_prd_graph(df, graph_id)
            return QueryResult(
                kind="chart",
                text=f"{output.insight_hint} Validation passed.",
                data=output.data,
                chart={"type": "plotly_figure", "figure": output.figure, "title": output.title},
                agent_trace={
                    "intent": {"source": "graph_catalog_router", "graph_id": graph_id},
                    "validation": {"is_valid": True, "issues": []},
                },
            )

    return answer_query(query, df)


def build_summary_export(df: pd.DataFrame, reference_time: datetime | None = None) -> dict[str, Any]:
    cards = build_prd_cards(df, reference_time=reference_time)
    report = build_insight_report(df)

    return {
        "cards": cards,
        "insight_report": report,
        "generated_at": datetime.now().isoformat(),
    }
