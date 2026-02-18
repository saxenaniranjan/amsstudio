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


def infer_category(row: pd.Series) -> str:
    if "category" in row and str(row["category"]).strip().lower() not in {"", "unknown", "nan", "none"}:
        return str(row["category"]).strip().title()

    text_parts = [
        str(row.get("description", "")),
        str(row.get("service", "")),
        str(row.get("team", "")),
    ]
    return _match_by_keywords(" ".join(text_parts), CATEGORY_KEYWORDS, default="General")


def infer_business_function(row: pd.Series) -> str:
    text_parts = [
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


def derive_features(df: pd.DataFrame, reference_time: datetime | None = None) -> pd.DataFrame:
    if reference_time is None:
        reference_time = datetime.now()

    enriched = df.copy()

    enriched["category_derived"] = enriched.apply(infer_category, axis=1)
    enriched["business_function_derived"] = enriched.apply(infer_business_function, axis=1)

    if "resolved_at" not in enriched:
        enriched["resolved_at"] = pd.NaT
    if "created_at" not in enriched:
        enriched["created_at"] = pd.NaT

    delta_hours = (enriched["resolved_at"] - enriched["created_at"]).dt.total_seconds() / 3600
    enriched["mttr_hours"] = delta_hours

    fallback_mask = enriched["mttr_hours"].isna() & enriched["resolution_time_hours"].notna()
    enriched.loc[fallback_mask, "mttr_hours"] = enriched.loc[fallback_mask, "resolution_time_hours"]

    enriched["mttr_hours"] = enriched["mttr_hours"].clip(lower=0)

    end_time = enriched["resolved_at"].fillna(pd.Timestamp(reference_time))
    enriched["ticket_age_hours"] = (end_time - enriched["created_at"]).dt.total_seconds() / 3600
    enriched["ticket_age_hours"] = enriched["ticket_age_hours"].clip(lower=0)

    enriched["is_open"] = ~enriched["is_resolved"]

    enriched["sla_threshold_hours"] = enriched["priority"].map(SLA_THRESHOLD_HOURS).fillna(SLA_THRESHOLD_HOURS["Unknown"])

    enriched["is_sla_breached"] = np.where(
        enriched["is_resolved"],
        enriched["mttr_hours"] > enriched["sla_threshold_hours"],
        enriched["ticket_age_hours"] > enriched["sla_threshold_hours"],
    )

    open_ratio = (enriched["ticket_age_hours"] / enriched["sla_threshold_hours"]).replace([np.inf, -np.inf], np.nan)
    enriched["sla_risk"] = np.where(
        ~enriched["is_open"],
        "Closed",
        np.where(open_ratio >= 1.0, "High", np.where(open_ratio >= 0.6, "Medium", "Low")),
    )

    enriched["resolution_bucket"] = enriched["mttr_hours"].map(_resolution_bucket)

    team_score = _calculate_team_performance(enriched)
    enriched["team_performance_index"] = enriched["team"].map(team_score).fillna(50.0)

    enriched["team_performance_band"] = pd.cut(
        enriched["team_performance_index"],
        bins=[-np.inf, 40, 65, 80, np.inf],
        labels=["Needs Attention", "Average", "Strong", "Top"],
    ).astype(str)

    return enriched
