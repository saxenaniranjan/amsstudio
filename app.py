from __future__ import annotations

from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

from ticket_analytics.insights import build_insight_report
from ticket_analytics.pipeline import run_ticket_pipeline
from ticket_analytics.query_engine import answer_query
from ticket_analytics.visualization import build_figure


st.set_page_config(page_title="ITIL Ticket Analytics", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def _load_uploaded_file(file_name: str, payload: bytes) -> pd.DataFrame:
    buffer = BytesIO(payload)
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(buffer)
    return pd.read_excel(buffer)


@st.cache_data(show_spinner=False)
def _run_pipeline_cached(df: pd.DataFrame) -> pd.DataFrame:
    return run_ticket_pipeline(df)


st.title("ITIL Ticket Analytics Platform")
st.caption(
    "Upload a ticket dump (Excel/CSV), derive analyst-grade features, and ask natural-language questions for charts and insights."
)

uploaded = st.file_uploader("Upload ticket dump", type=["xlsx", "xls", "csv"])
if uploaded is None:
    st.info("Upload an ITIL ticket export to start analysis.")
    st.stop()

raw = _load_uploaded_file(uploaded.name, uploaded.getvalue())
enriched = _run_pipeline_cached(raw)

st.sidebar.header("Filters")

team_options = sorted(enriched["team"].dropna().astype(str).unique().tolist())
priority_options = sorted(enriched["priority"].dropna().astype(str).unique().tolist())
category_options = sorted(enriched["category_derived"].dropna().astype(str).unique().tolist())

selected_teams = st.sidebar.multiselect("Team", team_options, default=team_options)
selected_priorities = st.sidebar.multiselect("Priority", priority_options, default=priority_options)
selected_categories = st.sidebar.multiselect("Category", category_options, default=category_options)

filtered = enriched.copy()
if selected_teams:
    filtered = filtered[filtered["team"].astype(str).isin(selected_teams)]
if selected_priorities:
    filtered = filtered[filtered["priority"].astype(str).isin(selected_priorities)]
if selected_categories:
    filtered = filtered[filtered["category_derived"].astype(str).isin(selected_categories)]

report = build_insight_report(filtered)
metrics = report["metrics"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Tickets", f"{metrics['total_tickets']}")
col2.metric("Open Tickets", f"{metrics['open_tickets']}")
col3.metric("Avg MTTR (hrs)", f"{metrics['avg_mttr_hours']}")
col4.metric("SLA Breach Rate", f"{round(metrics['breach_rate'] * 100, 2)}%")

overview_tab, query_tab, data_tab = st.tabs(["Overview", "Ask Analyst", "Data Explorer"])

with overview_tab:
    left, right = st.columns(2)

    with left:
        st.subheader("Ticket Volume by Category")
        category_frame = (
            filtered.groupby("category_derived", dropna=False)
            .size()
            .rename("ticket_count")
            .reset_index()
            .sort_values("ticket_count", ascending=False)
            .head(10)
        )
        if not category_frame.empty:
            fig = px.bar(category_frame, x="category_derived", y="ticket_count", title="Top Categories")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("SLA Breach Rate by Team")
        team_breach = (
            filtered.groupby("team", dropna=False)["is_sla_breached"]
            .mean()
            .mul(100)
            .round(2)
            .rename("breach_rate_pct")
            .reset_index()
            .sort_values("breach_rate_pct", ascending=False)
            .head(12)
        )
        if not team_breach.empty:
            fig = px.bar(team_breach, x="team", y="breach_rate_pct", title="Team Breach %")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Ticket Trend")
        trend = (
            filtered.dropna(subset=["created_at"])
            .set_index("created_at")
            .resample("D")
            .size()
            .rename("tickets")
            .reset_index()
        )
        if not trend.empty:
            fig = px.line(trend, x="created_at", y="tickets", title="Daily Ticket Trend", markers=True)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Team Performance Index")
        team_perf = (
            filtered.groupby("team", dropna=False)["team_performance_index"]
            .mean()
            .round(2)
            .rename("team_performance_index")
            .reset_index()
            .sort_values("team_performance_index", ascending=False)
        )
        if not team_perf.empty:
            fig = px.bar(team_perf, x="team", y="team_performance_index", title="Team Performance")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Analyst Recommendations")
    for recommendation in report["recommendations"]:
        st.write(f"- {recommendation}")

    if report["anomalies"]:
        st.subheader("Detected Volume Anomalies")
        st.dataframe(pd.DataFrame(report["anomalies"]), use_container_width=True)

with query_tab:
    st.subheader("Ask Any Question")
    st.caption(
        "Examples: 'Show bar chart of ticket count by team', 'Top 5 categories by breach rate', 'Give recommendations to reduce MTTR'."
    )

    query = st.text_area("Query", placeholder="Type your question about the ticket data", height=120)
    run = st.button("Run Query")

    if run:
        result = answer_query(query, filtered)
        st.markdown(result.text)
        fig = build_figure(result)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        if result.data is not None and not result.data.empty:
            st.dataframe(result.data, use_container_width=True)

with data_tab:
    st.subheader("Enriched Dataset")
    st.dataframe(filtered, use_container_width=True)

    st.download_button(
        "Download Enriched CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="enriched_ticket_analysis.csv",
        mime="text/csv",
    )
