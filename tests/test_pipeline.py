from __future__ import annotations

from ticket_analytics.pipeline import TicketAnalysisSession, run_ticket_pipeline
from ticket_analytics.visualization import build_figure


def test_pipeline_end_to_end(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)

    assert len(enriched) == len(raw_ticket_df)
    assert "team_performance_index" in enriched.columns
    assert "business_function_derived" in enriched.columns


def test_session_and_plotly_figure(raw_ticket_df, reference_time) -> None:
    session = TicketAnalysisSession.from_dataframe(raw_ticket_df, reference_time=reference_time)
    result = session.ask("line trend of ticket count over time")
    fig = build_figure(result)

    assert result.kind == "chart"
    assert result.chart is not None
    assert result.chart["type"] == "plotly_figure"
    assert result.chart["chart_type"] == "line"
    assert fig is not None
