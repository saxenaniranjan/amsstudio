from __future__ import annotations

from ticket_analytics.agentic import run_autonomous_mode, run_enabler_mode
from ticket_analytics.pipeline import run_ticket_pipeline


def test_autonomous_mode(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)
    response = run_autonomous_mode(enriched, reference_time=reference_time)

    assert response.mode == "autonomous"
    assert len(response.findings) > 0
    assert len(response.recommendations) > 0


def test_enabler_mode_graph_query(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)
    result = run_enabler_mode("show graph 2", enriched)

    assert result.kind == "chart"
    assert result.chart is not None
    assert result.chart["type"] == "plotly_figure"
