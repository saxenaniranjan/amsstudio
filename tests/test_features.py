from __future__ import annotations

import pandas as pd

from ticket_analytics.features import derive_features
from ticket_analytics.preprocessing import preprocess_tickets


def test_derive_features_core_fields(raw_ticket_df, reference_time) -> None:
    pre = preprocess_tickets(raw_ticket_df)
    enriched = derive_features(pre, reference_time=reference_time)

    expected = {
        "category_derived",
        "business_function_derived",
        "mttr_hours",
        "ticket_age_hours",
        "resolution_bucket",
        "sla_threshold_hours",
        "is_sla_breached",
        "sla_risk",
        "team_performance_index",
        "team_performance_band",
    }
    assert expected.issubset(set(enriched.columns))

    inc2 = enriched.loc[enriched["ticket_id"] == "INC002"].iloc[0]
    assert round(float(inc2["mttr_hours"]), 2) == 11.0
    assert bool(inc2["is_sla_breached"]) is True

    inc1 = enriched.loc[enriched["ticket_id"] == "INC001"].iloc[0]
    assert inc1["category_derived"] in {"Network", "Access"}

    inc3 = enriched.loc[enriched["ticket_id"] == "INC003"].iloc[0]
    assert inc3["business_function_derived"] == "Finance"

    assert enriched["team_performance_index"].between(0, 100).all()


def test_open_ticket_sla_risk(raw_ticket_df, reference_time) -> None:
    pre = preprocess_tickets(raw_ticket_df)
    enriched = derive_features(pre, reference_time=reference_time)

    open_tickets = enriched[enriched["is_open"]]
    assert not open_tickets.empty
    assert set(open_tickets["sla_risk"].unique()).issubset({"Low", "Medium", "High"})


def test_mttr_falls_back_to_resolution_duration_when_timestamp_delta_is_zero(reference_time) -> None:
    raw = pd.DataFrame(
        {
            "Incident Number": ["INC9001"],
            "Issue Description": ["ERP database issue"],
            "Assignment Group": ["DBA Team"],
            "State": ["Closed"],
            "Priority": ["2"],
            "Business Service": ["ERP"],
            "Opened At": ["2026-01-01 10:00:00"],
            "Resolved At": ["2026-01-01 10:00:00"],
            "Resolution Duration": ["0 days 04:30:00"],
            "Reopened": [0],
        }
    )
    pre = preprocess_tickets(raw)
    enriched = derive_features(pre, reference_time=reference_time)

    assert round(float(enriched.iloc[0]["mttr_hours"]), 2) == 4.5


def test_mttr_is_unknown_when_no_delta_and_no_resolution_duration(reference_time) -> None:
    raw = pd.DataFrame(
        {
            "Incident Number": ["INC9002"],
            "Issue Description": ["Access issue"],
            "Assignment Group": ["Service Desk"],
            "State": ["Closed"],
            "Priority": ["3"],
            "Business Service": ["Identity"],
            "Opened At": ["2026-01-01 10:00:00"],
            "Resolved At": ["2026-01-01 10:00:00"],
            "Resolution Duration": [None],
            "Reopened": [0],
        }
    )
    pre = preprocess_tickets(raw)
    enriched = derive_features(pre, reference_time=reference_time)

    assert pd.isna(enriched.iloc[0]["mttr_hours"])
