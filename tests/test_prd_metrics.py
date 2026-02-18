from __future__ import annotations

from datetime import datetime

import pandas as pd

from ticket_analytics.pipeline import run_ticket_pipeline
from ticket_analytics.prd_metrics import build_prd_cards, determine_dynamic_grain


def test_determine_dynamic_grain() -> None:
    assert determine_dynamic_grain(pd.Timestamp("2023-01-01"), pd.Timestamp("2026-12-31")) == "Y"
    assert determine_dynamic_grain(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-31")) == "Q"
    assert determine_dynamic_grain(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-05-31")) == "M"
    assert determine_dynamic_grain(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-31")) == "W"


def test_build_prd_cards(raw_ticket_df, reference_time: datetime) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)
    cards = build_prd_cards(enriched, reference_time=reference_time)

    expected_keys = {
        "data_completeness",
        "incident_volumetrics",
        "delivery_compliance",
        "efficiency",
        "quality",
        "performance",
    }
    assert expected_keys.issubset(cards.keys())

    assert cards["data_completeness"]["mandatory_columns_present_pct"] >= 0
    assert cards["incident_volumetrics"]["avg_inflow_rate"] >= 0
    assert cards["delivery_compliance"]["sla_adherence_pct"] >= 0
    assert cards["efficiency"]["mttr_hours"] >= 0
    assert cards["quality"]["recurring_issue_pct"] >= 0
    assert cards["performance"]["underperforming_teams_count"] >= 0
