from __future__ import annotations

from ticket_analytics.pipeline import run_ticket_pipeline
from ticket_analytics.workspace import suggest_global_filters, suggest_mapping_candidates


def test_suggest_mapping_candidates(raw_ticket_df) -> None:
    suggestions = suggest_mapping_candidates(raw_ticket_df)

    assert "ticket_id" in suggestions
    assert "team" in suggestions
    assert "service" in suggestions
    assert len(suggestions["ticket_id"]) > 0
    assert "Incident ID" in suggestions["ticket_id"]


def test_suggest_global_filters(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)
    filters = suggest_global_filters(enriched)

    assert "priority" in filters or "team" in filters
