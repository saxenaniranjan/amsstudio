"""Unified plot generation with Plotly for all chart types."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure

from .query_engine import QueryResult


def build_figure(result: QueryResult) -> Figure | None:
    if result.chart is None or result.data is None or result.data.empty:
        return None

    chart = result.chart
    df: pd.DataFrame = result.data
    chart_type = chart["type"]
    x = chart["x"]
    y = chart["y"]
    title = chart.get("title", "Ticket Analysis")

    if chart_type == "bar":
        return px.bar(df, x=x, y=y, title=title)
    if chart_type == "line":
        return px.line(df, x=x, y=y, markers=True, title=title)
    if chart_type == "pie":
        return px.pie(df, names=x, values=y, title=title)
    if chart_type == "histogram":
        return px.histogram(df, x=x, y=y if y in df.columns else None, title=title)
    if chart_type == "scatter":
        return px.scatter(df, x=x, y=y, title=title)
    if chart_type == "heatmap":
        return px.density_heatmap(df, x=x, y=y, title=title)

    return px.bar(df, x=x, y=y, title=title)
