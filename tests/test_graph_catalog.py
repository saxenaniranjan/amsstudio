from __future__ import annotations

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
