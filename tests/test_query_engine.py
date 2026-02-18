from __future__ import annotations

from ticket_analytics.pipeline import run_ticket_pipeline
from ticket_analytics.query_engine import answer_query


def test_query_chart_generation(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)
    result = answer_query("Show bar chart of ticket count by team", enriched)

    assert result.kind == "chart"
    assert result.chart is not None
    assert result.chart["type"] == "bar"
    assert result.data is not None
    assert not result.data.empty


def test_query_recommendations(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)
    result = answer_query("Give recommendations to improve MTTR", enriched)

    assert result.kind == "text"
    assert "Recommended actions" in result.text


def test_query_filter_by_team(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)
    result = answer_query("open tickets for service desk", enriched)

    assert result.data is not None
    assert not result.data.empty
    assert set(result.data["team"].astype(str).unique()) == {"Service Desk"}
