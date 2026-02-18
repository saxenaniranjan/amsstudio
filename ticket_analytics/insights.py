"""Insight and recommendation engine for enriched ticket data."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_ratio(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def build_overview_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    total = int(len(df))
    open_count = int(df["is_open"].sum()) if "is_open" in df else 0
    resolved_count = int(df["is_resolved"].sum()) if "is_resolved" in df else total - open_count
    breached = int(df["is_sla_breached"].sum()) if "is_sla_breached" in df else 0

    mttr = float(df["mttr_hours"].mean()) if "mttr_hours" in df and df["mttr_hours"].notna().any() else 0.0
    median_mttr = float(df["mttr_hours"].median()) if "mttr_hours" in df and df["mttr_hours"].notna().any() else 0.0
    breach_rate = _safe_ratio(breached, total)

    recurring_pct = (
        round(float(df["is_recurring_issue"].mean()) * 100.0, 2)
        if "is_recurring_issue" in df and not df.empty
        else 0.0
    )

    return {
        "total_tickets": total,
        "open_tickets": open_count,
        "resolved_tickets": resolved_count,
        "sla_breached_tickets": breached,
        "breach_rate": round(breach_rate, 4),
        "avg_mttr_hours": round(mttr, 2),
        "median_mttr_hours": round(median_mttr, 2),
        "recurring_issue_pct": recurring_pct,
    }


def find_volume_anomalies(df: pd.DataFrame) -> list[dict[str, Any]]:
    if "created_at" not in df or df["created_at"].isna().all():
        return []

    daily = (
        df.set_index("created_at")
        .resample("D")
        .size()
        .rename("ticket_count")
        .to_frame()
        .reset_index()
    )
    if len(daily) < 7:
        return []

    mean = daily["ticket_count"].mean()
    std = daily["ticket_count"].std(ddof=0)
    if std == 0:
        return []

    daily["z_score"] = (daily["ticket_count"] - mean) / std
    anomalies = daily[daily["z_score"] >= 2.0]

    return [
        {
            "date": row["created_at"].date().isoformat(),
            "ticket_count": int(row["ticket_count"]),
            "z_score": round(float(row["z_score"]), 2),
        }
        for _, row in anomalies.iterrows()
    ]


def generate_recommendations(df: pd.DataFrame) -> list[str]:
    recommendations: list[str] = []
    metrics = build_overview_metrics(df)

    if metrics["breach_rate"] > 0.2:
        recommendations.append(
            "SLA breach rate is above 20%. Prioritize breach-prone priorities and add proactive escalation triggers."
        )

    if metrics["avg_mttr_hours"] > 24:
        recommendations.append(
            "Average MTTR is high. Introduce a fast-lane triage for repetitive incident patterns and automate standard fixes."
        )

    if "category_derived" in df and not df.empty:
        dominant_category_share = df["category_derived"].value_counts(normalize=True).max()
        dominant_category = df["category_derived"].value_counts().idxmax()
        if dominant_category_share >= 0.35:
            recommendations.append(
                f"{dominant_category} drives over 35% of tickets. Run targeted root-cause reduction for that category."
            )

    if "team_performance_index" in df and not df.empty:
        low_team = (
            df.groupby("team")["team_performance_index"].mean().sort_values().head(1)
        )
        if not low_team.empty and low_team.iloc[0] < 55:
            team_name = low_team.index[0]
            recommendations.append(
                f"Team '{team_name}' has the lowest performance index. Review queue load balancing and runbook coverage."
            )

    if "is_at_risk" in df:
        high_risk_open = int(df["is_at_risk"].sum())
        if high_risk_open > 0:
            recommendations.append(
                f"There are {high_risk_open} at-risk open tickets (>80% SLA elapsed). Trigger immediate ownership and daily burn-down reviews."
            )

    if "is_recurring_issue" in df and metrics["recurring_issue_pct"] > 20:
        recommendations.append(
            "Recurring issues exceed 20% of volume. Launch recurring-issue elimination with automation/runbook standardization."
        )

    if not recommendations:
        recommendations.append(
            "Current performance appears stable. Continue monitoring MTTR, breach-rate drift, and top-category volume weekly."
        )

    return recommendations


def build_insight_report(df: pd.DataFrame) -> dict[str, Any]:
    metrics = build_overview_metrics(df)

    top_categories = []
    if "category_derived" in df and not df.empty:
        top_categories = (
            df["category_derived"]
            .value_counts()
            .head(5)
            .rename_axis("category")
            .reset_index(name="tickets")
            .to_dict("records")
        )

    team_summary = []
    if "team" in df and "team_performance_index" in df and not df.empty:
        team_summary = (
            df.groupby("team", dropna=False)
            .agg(
                tickets=("ticket_id", "count"),
                avg_mttr_hours=("mttr_hours", "mean"),
                breach_rate=("is_sla_breached", "mean"),
                performance_index=("team_performance_index", "mean"),
            )
            .sort_values("performance_index", ascending=False)
            .reset_index()
        )
        team_summary["avg_mttr_hours"] = team_summary["avg_mttr_hours"].round(2)
        team_summary["breach_rate"] = team_summary["breach_rate"].round(3)
        team_summary["performance_index"] = team_summary["performance_index"].round(2)
        team_summary = team_summary.to_dict("records")

    recurring_issues = []
    if "is_recurring_issue" in df and df["is_recurring_issue"].any():
        recurring_issues = (
            df[df["is_recurring_issue"]]
            .groupby(["issue_signature", "service", "team"], dropna=False)
            .size()
            .rename("tickets")
            .reset_index()
            .sort_values("tickets", ascending=False)
            .head(10)
            .to_dict("records")
        )

    return {
        "metrics": metrics,
        "top_categories": top_categories,
        "team_summary": team_summary,
        "recurring_issues": recurring_issues,
        "anomalies": find_volume_anomalies(df),
        "recommendations": generate_recommendations(df),
    }
