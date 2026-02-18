from __future__ import annotations

import numpy as np

from ticket_analytics.preprocessing import parse_duration_to_hours, preprocess_tickets


def test_parse_duration_to_hours_variants() -> None:
    assert parse_duration_to_hours("2h") == 2
    assert parse_duration_to_hours("1d 6h") == 30
    assert parse_duration_to_hours("11:00") == 11
    assert round(parse_duration_to_hours("90m"), 2) == 1.5
    assert np.isnan(parse_duration_to_hours(None))


def test_preprocess_tickets_aliases_and_types(raw_ticket_df) -> None:
    output = preprocess_tickets(raw_ticket_df)

    expected_columns = {
        "ticket_id",
        "description",
        "team",
        "status",
        "priority",
        "service",
        "created_at",
        "resolved_at",
        "resolution_time_hours",
        "reopen_count",
        "is_resolved",
    }
    assert expected_columns.issubset(set(output.columns))

    assert output["priority"].tolist()[:4] == ["P1", "P2", "P3", "P2"]
    assert output["reopen_count"].dtype.kind in {"i", "u"}
    assert output["created_at"].notna().all()
    assert output["resolved_at"].isna().sum() == 2
    assert int(output["is_resolved"].sum()) == 4
