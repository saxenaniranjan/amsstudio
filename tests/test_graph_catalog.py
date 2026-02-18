from __future__ import annotations

import pandas as pd

from ticket_analytics.graph_catalog import build_composite_graph, build_prd_graph, build_word_cloud_plotly
from ticket_analytics.pipeline import run_ticket_pipeline


def test_prd_graph_catalog_builds_all_graphs(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)

    for graph_id in [f"graph_{i}" for i in range(1, 9)]:
        output = build_prd_graph(enriched, graph_id)
        assert output.figure is not None
        assert output.title


def test_word_cloud_and_composite(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)

    wc = build_word_cloud_plotly(enriched)
    assert wc is not None

    composite = build_composite_graph(enriched, x_col="team", y_col="mttr_hours", chart_type="bar")
    assert composite is not None


def test_graph_1_handles_nullable_is_resolved(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)
    nullable = enriched.copy()
    nullable["is_resolved"] = nullable["is_resolved"].astype(object)
    nullable.loc[nullable.index[0], "is_resolved"] = None

    output = build_prd_graph(nullable, "graph_1")
    assert output.figure is not None
    assert {"period", "inflow", "outflow", "backlog"}.issubset(set(output.data.columns))


def test_graph_8_is_hour_of_day_over_week_heatmap(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)
    output = build_prd_graph(enriched, "graph_8")

    assert output.figure is not None
    assert len(output.figure.data) == 1
    trace = output.figure.data[0]
    assert list(trace.y) == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    assert len(trace.x) == 24
    assert trace.x[0] == "00:00"
    assert trace.x[-1] == "23:00"
    assert int(output.data["ticket_count"].sum()) == int(enriched["created_at"].notna().sum())


def test_graph_1_falls_back_to_updated_or_resolved_when_created_missing(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)
    fallback_df = enriched.copy()
    fallback_df["created_at"] = pd.NaT
    fallback_df["updated_at"] = fallback_df["resolved_at"]

    output = build_prd_graph(fallback_df, "graph_1")
    assert output.figure is not None
    assert not output.data.empty


def test_graph_1_handles_mixed_timezone_datetime_inputs(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time).copy()
    mixed = enriched.copy()
    mixed["created_at"] = mixed["created_at"].astype(str)
    if len(mixed) >= 2:
        mixed.loc[mixed.index[0], "created_at"] = "2026-01-01T00:00:00Z"
        mixed.loc[mixed.index[1], "created_at"] = "2026-01-02 00:00:00"

    output = build_prd_graph(mixed, "graph_1")
    assert output.figure is not None
    assert {"period", "inflow", "outflow", "backlog"}.issubset(set(output.data.columns))
