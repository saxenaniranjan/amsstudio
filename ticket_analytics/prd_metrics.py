"""PRD-aligned card metrics and helpers for Ticket-X."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .constants import PRD_MANDATORY_FIELDS


@dataclass
class PeriodWindow:
    current: pd.Timestamp | None
    previous: pd.Timestamp | None


def _safe_ratio(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def _pct_change(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    return round(((current - previous) / previous) * 100.0, 2)


def determine_dynamic_grain(start: pd.Timestamp, end: pd.Timestamp) -> str:
    months_span = (end.year - start.year) * 12 + (end.month - start.month) + 1
    years = months_span / 12.0

    if years > 3:
        return "Y"
    if years >= 1:
        return "Q"
    if months_span >= 2:
        return "M"
    return "W"


def get_complete_month_window(df: pd.DataFrame, reference_time: datetime | None = None) -> PeriodWindow:
    if reference_time is None:
        reference_time = datetime.now()

    if "created_at" not in df or df["created_at"].isna().all():
        return PeriodWindow(current=None, previous=None)

    month_values = df["created_at"].dropna().dt.to_period("M").dt.to_timestamp().sort_values().unique()
    if len(month_values) == 0:
        return PeriodWindow(current=None, previous=None)

    current_month_start = pd.Timestamp(reference_time).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    complete_months = [m for m in month_values if m < current_month_start]
    if not complete_months:
        current = month_values[-1]
    else:
        current = complete_months[-1]

    previous_candidates = [m for m in month_values if m < current]
    previous = previous_candidates[-1] if previous_candidates else None

    return PeriodWindow(current=current, previous=previous)


def build_data_completeness(df: pd.DataFrame) -> dict[str, float]:
    mandatory_columns = list(PRD_MANDATORY_FIELDS.keys())
    existing = [col for col in mandatory_columns if col in df.columns]
    if not existing:
        return {"mandatory_rows_complete_pct": 0.0, "mandatory_columns_present_pct": 0.0}

    row_complete_mask = df[existing].notna().all(axis=1)
    rows_complete_pct = round(float(row_complete_mask.mean()) * 100, 2) if len(df) else 0.0
    cols_present_pct = round((len(existing) / len(mandatory_columns)) * 100, 2)

    return {
        "mandatory_rows_complete_pct": rows_complete_pct,
        "mandatory_columns_present_pct": cols_present_pct,
    }


def _monthly_inflow_outflow(df: pd.DataFrame) -> pd.DataFrame:
    inflow = (
        df.dropna(subset=["created_at"])
        .assign(month=lambda x: x["created_at"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")
        .size()
        .rename("inflow")
    )

    outflow = (
        df[df["is_resolved"]]
        .dropna(subset=["resolved_at"])
        .assign(month=lambda x: x["resolved_at"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")
        .size()
        .rename("outflow")
    )

    all_months = inflow.index.union(outflow.index).sort_values()
    if len(all_months) == 0:
        return pd.DataFrame(columns=["month", "inflow", "outflow", "backlog"])

    monthly = pd.DataFrame(index=all_months)
    monthly["inflow"] = inflow.reindex(all_months).fillna(0)
    monthly["outflow"] = outflow.reindex(all_months).fillna(0)
    monthly["backlog"] = (monthly["inflow"] - monthly["outflow"]).cumsum()
    return monthly.reset_index().rename(columns={"index": "month"})


def build_incident_volumetrics_card(df: pd.DataFrame, reference_time: datetime | None = None) -> dict:
    monthly = _monthly_inflow_outflow(df)
    month_count = max(len(monthly), 1)

    avg_inflow = round(float(monthly["inflow"].sum()) / month_count, 2) if not monthly.empty else 0.0
    avg_outflow = round(float(monthly["outflow"].sum()) / month_count, 2) if not monthly.empty else 0.0
    avg_age_days = round(float(df[df["is_open"]]["ticket_age_days"].mean()), 2) if (df["is_open"].any()) else 0.0

    period = get_complete_month_window(df, reference_time=reference_time)
    current_inflow = previous_inflow = current_outflow = previous_outflow = 0.0

    if period.current is not None and not monthly.empty:
        curr = monthly[monthly["month"] == period.current]
        if not curr.empty:
            current_inflow = float(curr.iloc[0]["inflow"])
            current_outflow = float(curr.iloc[0]["outflow"])

    if period.previous is not None and not monthly.empty:
        prev = monthly[monthly["month"] == period.previous]
        if not prev.empty:
            previous_inflow = float(prev.iloc[0]["inflow"])
            previous_outflow = float(prev.iloc[0]["outflow"])

    return {
        "avg_inflow_rate": avg_inflow,
        "avg_outflow_rate": avg_outflow,
        "avg_open_ticket_age_days": avg_age_days,
        "current_month_inflow": current_inflow,
        "current_month_outflow": current_outflow,
        "inflow_change_pct": _pct_change(current_inflow, previous_inflow),
        "outflow_change_pct": _pct_change(current_outflow, previous_outflow),
        "series": monthly,
    }


def build_delivery_compliance_card(df: pd.DataFrame, reference_time: datetime | None = None) -> dict:
    period = get_complete_month_window(df, reference_time=reference_time)

    resolved = df[df["is_resolved"]].copy()
    if "resolved_at" in resolved:
        resolved["month"] = resolved["resolved_at"].dt.to_period("M").dt.to_timestamp()

    def _breach_count(period_month: pd.Timestamp | None) -> int:
        if period_month is None or resolved.empty:
            return 0
        subset = resolved[resolved["month"] == period_month]
        return int(subset["is_sla_breached"].sum())

    def _adherence_pct(period_month: pd.Timestamp | None) -> float:
        if period_month is None or resolved.empty:
            return 0.0
        subset = resolved[resolved["month"] == period_month]
        if subset.empty:
            return 0.0
        within = int((~subset["is_sla_breached"]).sum())
        return round(_safe_ratio(within, len(subset)) * 100.0, 2)

    current_breach = _breach_count(period.current)
    previous_breach = _breach_count(period.previous)
    current_adherence = _adherence_pct(period.current)
    previous_adherence = _adherence_pct(period.previous)

    quarterly = (
        resolved.dropna(subset=["month"])
        .groupby("month")["is_sla_breached"]
        .sum()
        .reset_index(name="breaches")
        .sort_values("month")
        .tail(3)
    )

    at_risk = int(df["is_at_risk"].sum()) if "is_at_risk" in df else 0

    return {
        "sla_adherence_pct": current_adherence,
        "sla_adherence_change_pct": _pct_change(current_adherence, previous_adherence),
        "sla_breach_count": current_breach,
        "sla_breach_change_pct": _pct_change(current_breach, previous_breach),
        "at_risk_open_tickets": at_risk,
        "quarterly_breach_trend": quarterly,
    }


def build_efficiency_card(df: pd.DataFrame, reference_time: datetime | None = None) -> dict:
    period = get_complete_month_window(df, reference_time=reference_time)
    resolved = df[df["is_resolved"]].copy()
    resolved["month"] = resolved["resolved_at"].dt.to_period("M").dt.to_timestamp()

    def _period_mttr(period_month: pd.Timestamp | None) -> float:
        if period_month is None:
            return 0.0
        subset = resolved[resolved["month"] == period_month]
        if subset.empty:
            return 0.0
        return round(float(subset["mttr_hours"].mean()), 2)

    current_mttr = _period_mttr(period.current)
    previous_mttr = _period_mttr(period.previous)

    overall_mttr = float(resolved["mttr_hours"].mean()) if not resolved.empty else 0.0
    app_mttr = (
        resolved.groupby("service", dropna=False)
        .agg(app_mttr=("mttr_hours", "mean"), ticket_count=("ticket_id", "count"))
        .reset_index()
        .sort_values("app_mttr", ascending=False)
    )

    detractors = app_mttr[app_mttr["app_mttr"] > overall_mttr].copy()
    detractors["app_mttr"] = detractors["app_mttr"].round(2)

    top_detractors = detractors.head(3).to_dict("records")

    return {
        "mttr_hours": current_mttr,
        "mttr_change_pct": _pct_change(current_mttr, previous_mttr),
        "detractor_applications_count": int(len(detractors)),
        "top_detractor_applications": top_detractors,
        "overall_mttr_hours": round(overall_mttr, 2),
    }


def build_quality_card(df: pd.DataFrame) -> dict:
    recurring = df[df["is_recurring_issue"]].copy() if "is_recurring_issue" in df else df.iloc[0:0]
    recurring_pct = round(_safe_ratio(len(recurring), len(df)) * 100.0, 2)

    top_categories = (
        recurring["category_derived"].value_counts().head(3).rename_axis("category").reset_index(name="tickets")
        if not recurring.empty
        else pd.DataFrame(columns=["category", "tickets"])
    )

    top_issues = (
        recurring.groupby(["issue_signature", "service", "team"], dropna=False)
        .size()
        .rename("tickets")
        .reset_index()
        .sort_values("tickets", ascending=False)
        .head(10)
        if not recurring.empty
        else pd.DataFrame(columns=["issue_signature", "service", "team", "tickets"])
    )

    return {
        "recurring_issue_pct": recurring_pct,
        "top_recurring_categories": top_categories,
        "top_recurring_issues": top_issues,
    }


def build_performance_card(df: pd.DataFrame) -> dict:
    team_perf = (
        df.groupby("team", dropna=False)
        .agg(
            tickets=("ticket_id", "count"),
            avg_mttr_hours=("mttr_hours", "mean"),
            breach_rate=("is_sla_breached", "mean"),
            team_performance_index=("team_performance_index", "mean"),
            total_resolution_hours=("mttr_hours", "sum"),
        )
        .reset_index()
    )

    team_perf["avg_mttr_hours"] = team_perf["avg_mttr_hours"].fillna(0)
    total_resolve_hours = float(team_perf["total_resolution_hours"].sum())

    if not team_perf.empty:
        threshold = float(team_perf["team_performance_index"].quantile(0.25))
        underperforming = team_perf[team_perf["team_performance_index"] <= threshold]
    else:
        underperforming = team_perf

    top_mttr = team_perf.sort_values("avg_mttr_hours", ascending=False).head(5).copy()
    top_mttr["resolution_time_share_pct"] = np.where(
        total_resolve_hours > 0,
        (top_mttr["total_resolution_hours"] / total_resolve_hours) * 100,
        0,
    )
    top_mttr["avg_mttr_hours"] = top_mttr["avg_mttr_hours"].round(2)
    top_mttr["resolution_time_share_pct"] = top_mttr["resolution_time_share_pct"].round(2)

    return {
        "underperforming_teams_count": int(len(underperforming)),
        "team_mttr_breakdown": top_mttr[["team", "avg_mttr_hours", "resolution_time_share_pct", "tickets"]],
    }


def build_prd_cards(df: pd.DataFrame, reference_time: datetime | None = None) -> dict:
    return {
        "data_completeness": build_data_completeness(df),
        "incident_volumetrics": build_incident_volumetrics_card(df, reference_time=reference_time),
        "delivery_compliance": build_delivery_compliance_card(df, reference_time=reference_time),
        "efficiency": build_efficiency_card(df, reference_time=reference_time),
        "quality": build_quality_card(df),
        "performance": build_performance_card(df),
    }
