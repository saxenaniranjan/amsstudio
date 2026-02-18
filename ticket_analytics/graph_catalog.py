"""PRD graph catalog built entirely with Plotly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import Figure

from .constants import PRD_GRAPH_IDS, RESOLVED_STATUSES
from .prd_metrics import determine_dynamic_grain


@dataclass
class GraphOutput:
    graph_id: str
    title: str
    figure: Figure
    data: pd.DataFrame
    insight_hint: str


def list_prd_graphs() -> dict[str, str]:
    return PRD_GRAPH_IDS


def _resample_count(frame: pd.DataFrame, datetime_col: str, freq: str, value_name: str) -> pd.DataFrame:
    if datetime_col not in frame or frame[datetime_col].isna().all():
        return pd.DataFrame(columns=["period", value_name])

    return (
        frame.dropna(subset=[datetime_col])
        .set_index(datetime_col)
        .resample(freq)
        .size()
        .rename(value_name)
        .reset_index()
        .rename(columns={datetime_col: "period"})
    )


def _graph_1_ticket_trend(df: pd.DataFrame, freq: str) -> GraphOutput:
    resolved_mask = pd.Series(False, index=df.index)
    if "is_resolved" in df.columns:
        raw_mask = df["is_resolved"]
        if pd.api.types.is_bool_dtype(raw_mask):
            resolved_mask = raw_mask.fillna(False)
        else:
            text_mask = raw_mask.astype(str).str.strip().str.lower()
            resolved_mask = text_mask.isin({"true", "1", "yes", "y"}) | text_mask.isin(RESOLVED_STATUSES)
    if "resolved_at" in df.columns:
        resolved_mask = resolved_mask | df["resolved_at"].notna()

    inflow = _resample_count(df, "created_at", freq, "inflow")
    outflow = _resample_count(df[resolved_mask.fillna(False)], "resolved_at", freq, "outflow")

    merged = pd.merge(inflow, outflow, on="period", how="outer").fillna(0).sort_values("period")
    merged["backlog"] = (merged["inflow"] - merged["outflow"]).cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["period"], y=merged["inflow"], mode="lines+markers", name="Inflow"))
    fig.add_trace(go.Scatter(x=merged["period"], y=merged["outflow"], mode="lines+markers", name="Outflow"))
    fig.add_trace(go.Scatter(x=merged["period"], y=merged["backlog"], mode="lines+markers", name="Backlog"))
    fig.update_layout(title="Graph 1: Ticket Trend Analysis", xaxis_title="Period", yaxis_title="Ticket Count")

    return GraphOutput(
        graph_id="graph_1",
        title="Ticket Trend Analysis",
        figure=fig,
        data=merged,
        insight_hint="Track inflow vs outflow to detect backlog growth and forecast resource pressure.",
    )


def _graph_2_sla_breach_trend(df: pd.DataFrame, freq: str) -> GraphOutput:
    resolved = df[df["is_resolved"]].dropna(subset=["resolved_at"]).copy()
    if resolved.empty:
        empty = pd.DataFrame(columns=["period", "priority", "breaches"])
        fig = px.line(empty, x="period", y="breaches", color="priority", title="Graph 2: SLA Breach Trend")
        return GraphOutput("graph_2", "SLA Breach Trend", fig, empty, "No resolved tickets available.")

    resolved["period"] = resolved["resolved_at"].dt.to_period(freq).dt.to_timestamp()
    grouped = (
        resolved.groupby(["period", "priority"], dropna=False)["is_sla_breached"]
        .sum()
        .rename("breaches")
        .reset_index()
        .sort_values("period")
    )

    fig = px.line(grouped, x="period", y="breaches", color="priority", markers=True, title="Graph 2: SLA Breach Trend")
    return GraphOutput(
        graph_id="graph_2",
        title="SLA Breach Trend",
        figure=fig,
        data=grouped,
        insight_hint="Detect priority-wise breach spikes and sustained compliance risks.",
    )


def _graph_3_top_issue_categories(df: pd.DataFrame) -> GraphOutput:
    grouped = (
        df["category_derived"].value_counts().head(5).rename_axis("category").reset_index(name="ticket_count")
    )
    grouped["share_pct"] = (grouped["ticket_count"] / max(len(df), 1) * 100).round(2)
    fig = px.bar(
        grouped,
        x="category",
        y="ticket_count",
        color="share_pct",
        title="Graph 3: Top Issue Categories",
        text="share_pct",
    )
    fig.update_traces(texttemplate="%{text}%")
    return GraphOutput(
        graph_id="graph_3",
        title="Top Issue Categories",
        figure=fig,
        data=grouped,
        insight_hint="Categories dominating ticket share indicate where root-cause programs should focus.",
    )


def _graph_4_top_mttr_groups(df: pd.DataFrame) -> GraphOutput:
    grouped = (
        df.groupby("team", dropna=False)["mttr_hours"].mean().rename("mttr_hours").reset_index().sort_values("mttr_hours", ascending=False).head(5)
    )
    grouped["mttr_hours"] = grouped["mttr_hours"].round(2)
    fig = px.bar(grouped, x="team", y="mttr_hours", title="Graph 4: Top MTTR Groups")
    return GraphOutput(
        graph_id="graph_4",
        title="Top MTTR Groups",
        figure=fig,
        data=grouped,
        insight_hint="Highest-MTTR teams are operational detractors and should be targeted for runbook/process fixes.",
    )


def _graph_5_top_recurring_issues(df: pd.DataFrame) -> GraphOutput:
    recurring = df[df["is_recurring_issue"]]
    grouped = (
        recurring.groupby(["issue_signature", "service", "team"], dropna=False)
        .size()
        .rename("ticket_count")
        .reset_index()
        .sort_values("ticket_count", ascending=False)
        .head(10)
    )

    if grouped.empty:
        grouped = pd.DataFrame(columns=["issue_signature", "service", "team", "ticket_count"])

    fig = px.bar(
        grouped,
        x="issue_signature",
        y="ticket_count",
        color="team" if "team" in grouped else None,
        title="Graph 5: Top Recurring Issues",
    )
    fig.update_layout(xaxis_title="Recurring Issue", yaxis_title="Ticket Count")

    return GraphOutput(
        graph_id="graph_5",
        title="Top Recurring Issues",
        figure=fig,
        data=grouped,
        insight_hint="Recurring issues with high volume indicate candidates for automation and structural fixes.",
    )


def _graph_6_team_performance(df: pd.DataFrame, freq: str) -> GraphOutput:
    if "assignee" not in df:
        df = df.assign(assignee="Unknown")

    base = df.dropna(subset=["created_at"]).copy()
    if base.empty:
        empty = pd.DataFrame(columns=["period", "band", "count"])
        fig = px.bar(empty, x="period", y="count", color="band", barmode="stack", title="Graph 6: Team Performance")
        return GraphOutput("graph_6", "Team Performance", fig, empty, "No dated records available.")

    base["period"] = base["created_at"].dt.to_period(freq).dt.to_timestamp()
    median_score = float(base["team_performance_index"].median()) if "team_performance_index" in base else 50.0
    base["band"] = np.where(base["team_performance_index"] >= median_score, "Above Average", "Below Average")

    grouped = (
        base.groupby(["period", "band"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("period")
    )
    fig = px.bar(grouped, x="period", y="count", color="band", barmode="stack", title="Graph 6: Team Performance")
    return GraphOutput(
        graph_id="graph_6",
        title="Team Performance",
        figure=fig,
        data=grouped,
        insight_hint="Monitor periods where below-average team share rises; this is an early operational warning.",
    )


def _graph_7_top_aged_open_tickets(df: pd.DataFrame) -> GraphOutput:
    open_tickets = df[df["is_open"]].copy()
    top = (
        open_tickets.sort_values("ticket_age_days", ascending=False)
        [["ticket_id", "description", "team", "ticket_age_days"]]
        .head(10)
        .rename(columns={"description": "short_description", "team": "assigned_team"})
    )

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=["Ticket", "Short Description", "Assigned Team", "Days Open"]),
                cells=dict(
                    values=[
                        top.get("ticket_id", pd.Series(dtype=object)),
                        top.get("short_description", pd.Series(dtype=object)),
                        top.get("assigned_team", pd.Series(dtype=object)),
                        top.get("ticket_age_days", pd.Series(dtype=float)),
                    ]
                ),
            )
        ]
    )
    fig.update_layout(title="Graph 7: Top Aged Open Tickets")

    return GraphOutput(
        graph_id="graph_7",
        title="Top Aged Open Tickets",
        figure=fig,
        data=top,
        insight_hint="Aged tickets highlight unresolved bottlenecks and likely near-term SLA risk.",
    )


def _graph_8_time_trend_heatmap(df: pd.DataFrame) -> GraphOutput:
    frame = df.dropna(subset=["created_at"]).copy()
    if frame.empty:
        empty = pd.DataFrame(columns=["weekday", "hour", "hour_label", "ticket_count"])
        fig = go.Figure(
            data=[
                go.Heatmap(
                    x=[f"{hour:02d}:00" for hour in range(24)],
                    y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                    z=np.zeros((7, 24)),
                    colorscale="YlOrRd",
                    colorbar={"title": "Tickets"},
                )
            ]
        )
        fig.update_layout(
            title="Graph 8: Time Trend Analysis",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
        )
        return GraphOutput("graph_8", "Time Trend Analysis", fig, empty, "No dated records available.")

    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hour_order = list(range(24))

    frame["weekday"] = frame["created_at"].dt.day_name().str[:3]
    frame["hour"] = frame["created_at"].dt.hour

    grouped = (
        frame.groupby(["weekday", "hour"], dropna=False)
        .size()
        .rename("ticket_count")
        .reset_index()
    )

    full_grid = pd.MultiIndex.from_product([weekday_order, hour_order], names=["weekday", "hour"]).to_frame(index=False)
    grouped = full_grid.merge(grouped, on=["weekday", "hour"], how="left")
    grouped["ticket_count"] = grouped["ticket_count"].fillna(0).astype(int)
    grouped["weekday"] = pd.Categorical(grouped["weekday"], categories=weekday_order, ordered=True)
    grouped["hour_label"] = grouped["hour"].map(lambda hour: f"{int(hour):02d}:00")

    pivot = grouped.pivot(index="weekday", columns="hour_label", values="ticket_count").reindex(weekday_order)
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=list(pivot.columns),
                y=list(pivot.index),
                z=pivot.to_numpy(),
                colorscale="YlOrRd",
                colorbar={"title": "Tickets"},
            )
        ]
    )
    fig.update_layout(title="Graph 8: Time Trend Analysis", xaxis_title="Hour of Day", yaxis_title="Day of Week")
    fig.update_yaxes(categoryorder="array", categoryarray=weekday_order)

    return GraphOutput(
        graph_id="graph_8",
        title="Time Trend Analysis",
        figure=fig,
        data=grouped,
        insight_hint="Heatmap hotspots indicate workload concentration by weekday and hour.",
    )


def build_prd_graph(df: pd.DataFrame, graph_id: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> GraphOutput:
    working = df.copy()
    if start is not None and "created_at" in working:
        working = working[working["created_at"] >= pd.Timestamp(start)]
    if end is not None and "created_at" in working:
        working = working[working["created_at"] <= pd.Timestamp(end)]

    if "created_at" in working and working["created_at"].notna().any():
        min_date = working["created_at"].min()
        max_date = working["created_at"].max()
        freq = determine_dynamic_grain(min_date, max_date)
    else:
        freq = "M"

    graph_id = graph_id.strip().lower()
    if graph_id == "graph_1":
        return _graph_1_ticket_trend(working, freq)
    if graph_id == "graph_2":
        return _graph_2_sla_breach_trend(working, freq)
    if graph_id == "graph_3":
        return _graph_3_top_issue_categories(working)
    if graph_id == "graph_4":
        return _graph_4_top_mttr_groups(working)
    if graph_id == "graph_5":
        return _graph_5_top_recurring_issues(working)
    if graph_id == "graph_6":
        return _graph_6_team_performance(working, freq)
    if graph_id == "graph_7":
        return _graph_7_top_aged_open_tickets(working)
    if graph_id == "graph_8":
        return _graph_8_time_trend_heatmap(working)

    raise ValueError(f"Unsupported graph id: {graph_id}")


def build_word_cloud_plotly(df: pd.DataFrame, top_n: int = 50) -> Figure:
    text = (
        df["description"].fillna("").astype(str)
        + " "
        + df.get("closure_notes", pd.Series(["" for _ in range(len(df))], index=df.index)).fillna("").astype(str)
    )
    tokens = (
        text.str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.split()
        .explode()
        .dropna()
    )
    tokens = tokens[tokens.str.len() > 3]

    freq = tokens.value_counts().head(top_n).reset_index()
    freq.columns = ["word", "count"]

    if freq.empty:
        return px.scatter(pd.DataFrame({"x": [], "y": []}), x="x", y="y", title="Word Cloud")

    freq["x"] = np.cos(np.linspace(0, 6.28, len(freq))) * np.linspace(1, 10, len(freq))
    freq["y"] = np.sin(np.linspace(0, 6.28, len(freq))) * np.linspace(1, 10, len(freq))

    fig = px.scatter(freq, x="x", y="y", size="count", text="word", title="Word Cloud")
    fig.update_traces(marker_opacity=0.2)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(showlegend=False)
    return fig


def build_composite_graph(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    chart_type: str,
    color_col: str | None = None,
) -> Figure:
    chart_type = chart_type.lower()

    if chart_type == "bar":
        return px.bar(df, x=x_col, y=y_col, color=color_col, title="Composite Graph")
    if chart_type == "line":
        return px.line(df, x=x_col, y=y_col, color=color_col, markers=True, title="Composite Graph")
    if chart_type == "scatter":
        return px.scatter(df, x=x_col, y=y_col, color=color_col, title="Composite Graph")
    if chart_type == "histogram":
        return px.histogram(df, x=x_col, y=y_col if y_col in df else None, color=color_col, title="Composite Graph")
    if chart_type == "box":
        return px.box(df, x=x_col, y=y_col, color=color_col, title="Composite Graph")

    return px.bar(df, x=x_col, y=y_col, color=color_col, title="Composite Graph")
