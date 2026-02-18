"""Derived feature generation for ticket analytics."""

from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd

from .constants import BUSINESS_FUNCTION_KEYWORDS, CATEGORY_KEYWORDS, SLA_THRESHOLD_HOURS


def _match_by_keywords(text: str, mapping: dict[str, list[str]], default: str = "Other") -> str:
    lowered = text.lower()
    for label, keywords in mapping.items():
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword.lower())}\b", lowered):
                return label
    return default


def _normalize_issue_signature(text: str) -> str:
    simplified = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    simplified = re.sub(r"\d+", " ", simplified)
    simplified = re.sub(r"\s+", " ", simplified).strip()
    if not simplified:
        return "unknown"
    return " ".join(simplified.split()[:8])


def infer_category(row: pd.Series) -> str:
    if "category" in row and str(row["category"]).strip().lower() not in {"", "unknown", "nan", "none"}:
        return str(row["category"]).strip().title()
    if "subcategory" in row and str(row["subcategory"]).strip().lower() not in {"", "unknown", "nan", "none"}:
        return str(row["subcategory"]).strip().title()

    text_parts = [
        str(row.get("description", "")),
        str(row.get("service", "")),
        str(row.get("subcategory", "")),
        str(row.get("team", "")),
    ]
    return _match_by_keywords(" ".join(text_parts), CATEGORY_KEYWORDS, default="Other")


def infer_business_function(row: pd.Series) -> str:
    if "sub_domain" in row and str(row["sub_domain"]).strip().lower() not in {"", "unknown", "nan", "none"}:
        return str(row["sub_domain"]).strip().title()
    if "domain" in row and str(row["domain"]).strip().lower() not in {"", "unknown", "nan", "none"}:
        return str(row["domain"]).strip().title()

    text_parts = [
        str(row.get("domain", "")),
        str(row.get("sub_domain", "")),
        str(row.get("service", "")),
        str(row.get("description", "")),
        str(row.get("team", "")),
    ]
    return _match_by_keywords(" ".join(text_parts), BUSINESS_FUNCTION_KEYWORDS, default="General")


def _resolution_bucket(hours: float) -> str:
    if np.isnan(hours):
        return "Unknown"
    if hours <= 4:
        return "0-4h"
    if hours <= 8:
        return "4-8h"
    if hours <= 24:
        return "8-24h"
    if hours <= 72:
        return "1-3d"
    return ">3d"


def _calculate_team_performance(df: pd.DataFrame) -> pd.Series:
    grouped = (
        df.groupby("team", dropna=False)
        .agg(
            avg_mttr=("mttr_hours", "mean"),
            breach_rate=("is_sla_breached", "mean"),
            volume=("ticket_id", "count"),
            reopen_rate=("reopen_count", lambda s: (s > 0).mean()),
        )
        .fillna(0)
    )

    def _minmax_inverse(series: pd.Series) -> pd.Series:
        spread = series.max() - series.min()
        if spread == 0:
            return pd.Series(1.0, index=series.index)
        return 1 - ((series - series.min()) / spread)

    def _minmax(series: pd.Series) -> pd.Series:
        spread = series.max() - series.min()
        if spread == 0:
            return pd.Series(1.0, index=series.index)
        return (series - series.min()) / spread

    mttr_score = _minmax_inverse(grouped["avg_mttr"].fillna(grouped["avg_mttr"].max()))
    breach_score = _minmax_inverse(grouped["breach_rate"])
    reopen_score = _minmax_inverse(grouped["reopen_rate"])
    volume_score = _minmax(grouped["volume"])

    score = 100 * (0.35 * mttr_score + 0.35 * breach_score + 0.2 * reopen_score + 0.1 * volume_score)
    return score.round(2)


def derive_features(
    df: pd.DataFrame,
    reference_time: datetime | None = None,
    sla_threshold_hours: dict[str, float] | None = None,
) -> pd.DataFrame:
    if reference_time is None:
        reference_time = datetime.now()

    sla_map = dict(SLA_THRESHOLD_HOURS)
    if sla_threshold_hours:
        for key, value in sla_threshold_hours.items():
            if value is None:
                continue
            sla_map[str(key)] = float(value)

    enriched = df.copy()

    enriched["category_derived"] = enriched.apply(infer_category, axis=1)
    enriched["business_function_derived"] = enriched.apply(infer_business_function, axis=1)
    enriched["issue_signature"] = enriched["description"].fillna("Unknown").astype(str).map(_normalize_issue_signature)

    if "resolved_at" not in enriched:
        enriched["resolved_at"] = pd.NaT
    if "created_at" not in enriched:
        enriched["created_at"] = pd.NaT
    if "resolution_time_hours" not in enriched:
        enriched["resolution_time_hours"] = np.nan

    delta_hours = (enriched["resolved_at"] - enriched["created_at"]).dt.total_seconds() / 3600
    enriched["mttr_hours"] = delta_hours

    fallback_mask = (
        (enriched["mttr_hours"].isna() | (enriched["mttr_hours"] <= 0))
        & (enriched["resolution_time_hours"] > 0)
    )
    enriched.loc[fallback_mask, "mttr_hours"] = enriched.loc[fallback_mask, "resolution_time_hours"]

    no_signal_mask = (
        enriched["is_resolved"]
        & (enriched["mttr_hours"] <= 0)
        & ((enriched["resolution_time_hours"].isna()) | (enriched["resolution_time_hours"] <= 0))
    )
    enriched.loc[no_signal_mask, "mttr_hours"] = np.nan

    enriched["mttr_hours"] = enriched["mttr_hours"].clip(lower=0)

    end_time = enriched["resolved_at"].fillna(pd.Timestamp(reference_time))
    enriched["ticket_age_hours"] = (end_time - enriched["created_at"]).dt.total_seconds() / 3600
    enriched["ticket_age_hours"] = enriched["ticket_age_hours"].clip(lower=0)
    enriched["ticket_age_days"] = (enriched["ticket_age_hours"] / 24.0).round(2)

    enriched["is_open"] = ~enriched["is_resolved"]

    enriched["sla_threshold_hours"] = enriched["priority"].map(sla_map).fillna(sla_map.get("Unknown", 24))

    resolved_duration = enriched["mttr_hours"].fillna(enriched["ticket_age_hours"])

    enriched["is_sla_breached"] = np.where(
        enriched["is_resolved"],
        resolved_duration > enriched["sla_threshold_hours"],
        enriched["ticket_age_hours"] > enriched["sla_threshold_hours"],
    )

    enriched["sla_progress_ratio"] = (
        (enriched["ticket_age_hours"] / enriched["sla_threshold_hours"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    enriched["is_at_risk"] = enriched["is_open"] & (enriched["sla_progress_ratio"] >= 0.8)

    enriched["sla_risk"] = np.where(
        ~enriched["is_open"],
        "Closed",
        np.where(
            enriched["sla_progress_ratio"] >= 1.0,
            "High",
            np.where(enriched["sla_progress_ratio"] >= 0.8, "Medium", "Low"),
        ),
    )

    enriched["resolution_bucket"] = enriched["mttr_hours"].map(_resolution_bucket)

    issue_counts = enriched["issue_signature"].value_counts(dropna=False)
    enriched["issue_occurrence_count"] = enriched["issue_signature"].map(issue_counts).fillna(0).astype(int)
    enriched["is_recurring_issue"] = enriched["issue_occurrence_count"] >= 5

    team_score = _calculate_team_performance(enriched)
    enriched["team_performance_index"] = enriched["team"].map(team_score).fillna(50.0)

    enriched["team_performance_band"] = pd.cut(
        enriched["team_performance_index"],
        bins=[-np.inf, 40, 65, 80, np.inf],
        labels=["Needs Attention", "Average", "Strong", "Top"],
    ).astype(str)

    return enriched
