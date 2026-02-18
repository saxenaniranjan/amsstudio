from __future__ import annotations

import pandas as pd
import pytest

from ticket_analytics.pipeline import run_ticket_pipeline
from ticket_analytics.query_engine import AgenticQueryIntent, IntentAgent, answer_query


def test_query_chart_generation(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)
    result = answer_query("Show bar chart of ticket count by team", enriched)

    assert result.kind == "chart"
    assert result.chart is not None
    assert result.chart["type"] == "plotly_figure"
    assert result.chart["chart_type"] == "bar"
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


def test_agentic_mttr_line_chart_with_time_and_priority_slice(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)
    result = answer_query("give me mttr line chart over last 3 months for P2 tickets only", enriched)

    assert result.kind == "chart"
    assert result.chart is not None
    assert result.chart["type"] == "plotly_figure"
    assert result.chart["chart_type"] == "line"
    assert result.data is not None
    assert "avg_mttr_hours" in result.data.columns

    trace = result.agent_trace or {}
    intent = trace.get("intent", {})
    assert intent.get("metric") == "avg_mttr_hours"
    assert intent.get("filters", {}).get("priority") == ["P2"]
    assert trace.get("validation", {}).get("is_valid") is True


def test_agentic_query_returns_data_unavailable_message(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)
    result = answer_query("show mttr line chart over last 3 months for team UnknownTeamX", enriched)

    assert result.kind == "chart"
    assert result.chart is not None
    assert result.chart["type"] == "plotly_figure"
    assert result.chart["chart_type"] == "line"
    assert "No data available" in result.text
    assert result.chart["figure"] is not None
    trace = result.agent_trace or {}
    assert trace.get("validation", {}).get("is_valid") is True


def test_agentic_mttr_last_month_priority_filters_generate_line_time_graph(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)
    result = answer_query("Generate a graph on mttr over last month for P2 and P3 tickets", enriched)

    assert result.kind == "chart"
    assert result.chart is not None
    assert result.chart["type"] == "plotly_figure"
    assert result.chart["chart_type"] == "line"
    assert result.data is not None
    assert "period" in result.data.columns
    assert "avg_mttr_hours" in result.data.columns
    assert "priority" in result.data.columns
    assert set(result.data["priority"].astype(str).unique()) == {"P2", "P3"}
    assert result.chart["figure"] is not None
    assert len(result.chart["figure"].data) == 2

    trace = result.agent_trace or {}
    intent = trace.get("intent", {})
    assert intent.get("dimension") == "created_at"
    assert intent.get("lookback_value") == 1
    assert intent.get("lookback_unit") == "month"


def test_mttr_priority_query_with_priority_text_variants_uses_time_dimension(raw_ticket_df, reference_time) -> None:
    with_priority_labels = raw_ticket_df.copy()
    with_priority_labels["Severity"] = ["P1 - Critical", "P2 - High", "P3 - Medium", "P2 - High", "P4 - Low", "P3 - Medium"]
    enriched = run_ticket_pipeline(with_priority_labels, reference_time=reference_time)

    result = answer_query("Create a line graph for mttr over last month for Priority P2 and P3", enriched)

    assert result.kind == "chart"
    assert result.chart is not None
    assert result.data is not None
    assert not result.data.empty

    trace = result.agent_trace or {}
    intent = trace.get("intent", {})
    assert intent.get("dimension") == "created_at"
    assert intent.get("filters", {}).get("priority") == ["P2", "P3"]
    assert set(result.data["priority"].astype(str).unique()) == {"P2", "P3"}


def test_explicit_bar_chart_request_is_not_overridden_to_line(raw_ticket_df, reference_time) -> None:
    with_priority_labels = raw_ticket_df.copy()
    with_priority_labels["Severity"] = ["P1 - Critical", "P2 - High", "P3 - Medium", "P2 - High", "P4 - Low", "P3 - Medium"]
    enriched = run_ticket_pipeline(with_priority_labels, reference_time=reference_time)

    result = answer_query("Create a bar chart for mttr over last month for Priority P2 and P3", enriched)

    assert result.kind == "chart"
    assert result.chart is not None
    assert result.chart["chart_type"] == "bar"
    trace = result.agent_trace or {}
    intent = trace.get("intent", {})
    assert intent.get("dimension") == "created_at"
    assert intent.get("filters", {}).get("priority") == ["P2", "P3"]


def test_complex_top_applications_mttr_query_uses_priority_series_split(raw_ticket_df, reference_time) -> None:
    custom = raw_ticket_df.copy()
    custom["Business Service"] = ["App A", "App A", "App B", "App B", "App C", "App C"]
    custom["Severity"] = ["1", "2", "3", "4", "1", "4"]
    custom["State"] = ["Closed"] * len(custom)

    opened = pd.to_datetime(custom["Opened At"])
    custom["Resolved At"] = (
        opened + pd.to_timedelta([2, 6, 10, 14, 18, 22], unit="h")
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    enriched = run_ticket_pipeline(custom, reference_time=reference_time)
    result = answer_query(
        "Create a bar chart for top 3 applications with highest mttr across all priority labelled as different colours for P1,P2,P3 and P4",
        enriched,
    )

    assert result.kind == "chart"
    assert result.chart is not None
    assert result.chart["chart_type"] == "bar"
    assert result.data is not None
    assert {"service", "priority", "avg_mttr_hours"}.issubset(set(result.data.columns))
    assert set(result.data["service"].astype(str).unique()) == {"App A", "App B", "App C"}

    trace = result.agent_trace or {}
    intent = trace.get("intent", {})
    assert intent.get("metric") == "avg_mttr_hours"
    assert intent.get("dimension") == "service"
    assert intent.get("series_dimension") == "priority"
    assert intent.get("top_n") == 3
    assert len(result.chart["figure"].data) == 4


def test_mttr_over_all_time_across_priorities_uses_time_and_priority_series(raw_ticket_df, reference_time) -> None:
    with_priority_labels = raw_ticket_df.copy()
    with_priority_labels["Severity"] = ["P1 - Critical", "P2 - High", "P3 - Medium", "P2 - High", "P4 - Low", "P3 - Medium"]
    enriched = run_ticket_pipeline(with_priority_labels, reference_time=reference_time)

    result = answer_query("Create a line chart for MTTR over all Time across P1,P2,P3 and P4.", enriched)

    assert result.kind == "chart"
    assert result.chart is not None
    assert result.chart["chart_type"] == "line"
    assert result.data is not None
    assert not result.data.empty

    trace = result.agent_trace or {}
    intent = trace.get("intent", {})
    assert intent.get("dimension") == "created_at"
    assert intent.get("series_dimension") == "priority"
    assert intent.get("filters", {}).get("priority") == ["P1", "P2", "P3", "P4"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "Show SLA compliance trend over last 3 months by priority",
            {"metric": "sla_compliance_pct", "dimension": "created_at", "series_dimension": "priority"},
        ),
        (
            "Plot open tickets by assignment group for incident tickets",
            {"metric": "open_tickets", "dimension": "team"},
        ),
        (
            "Visualize breached tickets over all time across P1,P2,P3 and P4",
            {"metric": "breached_tickets", "dimension": "created_at", "series_dimension": "priority"},
        ),
        (
            "Bar chart for top three applications with highest MTTR split by priority",
            {"metric": "avg_mttr_hours", "dimension": "service", "series_dimension": "priority"},
        ),
        (
            "Line chart of MTTR trend by assignment group for this month",
            {"metric": "avg_mttr_hours", "dimension": "created_at", "series_dimension": "team"},
        ),
        (
            "Show backlog trend for on hold tickets over last 4 weeks",
            {"metric": "open_tickets", "dimension": "created_at"},
        ),
        (
            "Chart SLA breach rate by status for current quarter",
            {"metric": "breach_rate_pct", "dimension": "status"},
        ),
        (
            "Plot team performance index by assignment group over last quarter",
            {"metric": "team_performance_index", "dimension": "team"},
        ),
    ],
)
def test_itil_style_queries_are_interpreted_without_clarification(
    raw_ticket_df,
    reference_time,
    query: str,
    expected: dict[str, str],
) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)
    result = answer_query(query, enriched)

    assert result.kind == "chart"
    trace = result.agent_trace or {}
    intent = trace.get("intent", {})
    assert intent.get("metric") == expected["metric"]
    assert intent.get("dimension") == expected["dimension"]
    if "series_dimension" in expected:
        assert intent.get("series_dimension") == expected["series_dimension"]
    assert result.chart is not None


def test_llm_plan_is_refined_with_rule_fallbacks(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)
    planner = IntentAgent(enriched)
    llm_intent = AgenticQueryIntent(
        request_kind="table",
        chart_type="bar",
        metric="ticket_count",
        dimension=None,
        time_grain="M",
        filters={},
        lookback_value=None,
        lookback_unit=None,
        top_n=None,
        source="llm_intent_agent",
    )

    refined = planner._refine_with_rules("Generate a graph on mttr over last month for P2 and P3 tickets", llm_intent)

    assert refined.request_kind == "graph"
    assert refined.chart_type == "line"
    assert refined.metric == "avg_mttr_hours"
    assert refined.dimension == "created_at"
    assert refined.lookback_value == 1
    assert refined.lookback_unit == "month"
    assert refined.filters.get("priority") == ["P2", "P3"]


def test_sla_compliance_top_assignment_groups_query_maps_to_expected_intent(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)
    result = answer_query("Generate a graph on sla compliance over last month for top 3 assignment groups", enriched)

    assert result.kind == "chart"
    assert result.chart is not None
    assert result.chart["type"] == "plotly_figure"
    assert result.chart["chart_type"] == "bar"
    assert result.data is not None
    assert "team" in result.data.columns
    assert "sla_compliance_pct" in result.data.columns
    assert len(result.data) <= 3

    trace = result.agent_trace or {}
    intent = trace.get("intent", {})
    assert intent.get("metric") == "sla_compliance_pct"
    assert intent.get("dimension") == "team"
    assert intent.get("top_n") == 3
    assert intent.get("lookback_value") == 1
    assert intent.get("lookback_unit") == "month"


def test_clarification_when_graph_request_lacks_dimension(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)
    result = answer_query("Generate a graph for SLA compliance", enriched)

    assert result.kind == "clarification"
    assert "clarification" in result.text.lower()
    assert result.chart is None
