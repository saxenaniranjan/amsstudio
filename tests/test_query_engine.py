from __future__ import annotations

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
