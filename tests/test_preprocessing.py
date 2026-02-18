from __future__ import annotations

import numpy as np
import pandas as pd

from ticket_analytics.preprocessing import parse_duration_to_hours, preprocess_tickets


def test_parse_duration_to_hours_variants() -> None:
    assert parse_duration_to_hours("2h") == 2
    assert parse_duration_to_hours("1d 6h") == 30
    assert parse_duration_to_hours("11:00") == 11
    assert round(parse_duration_to_hours("0 days 05:30:00"), 2) == 5.5
    assert round(parse_duration_to_hours(pd.Timedelta(hours=2, minutes=15)), 2) == 2.25
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


def test_preprocess_extended_aliases_and_duration_parsing() -> None:
    raw = pd.DataFrame(
        {
            "Incident Number": ["INC777"],
            "Issue Description": ["User cannot login to VPN"],
            "Assignment Group Name": ["Network Ops"],
            "Current Status": ["Closed"],
            "Priority Level": ["2"],
            "Business Service Name": ["VPN Gateway"],
            "Opened On": ["2026-01-01 09:00:00"],
            "Resolved On": ["2026-01-01 09:00:00"],
            "Resolution Age": ["0 days 06:00:00"],
            "Reopen Counter": [0],
        }
    )

    output = preprocess_tickets(raw)
    row = output.iloc[0]

    assert row["ticket_id"] == "INC777"
    assert row["team"] == "Network Ops"
    assert round(float(row["resolution_time_hours"]), 2) == 6.0
    assert bool(row["is_resolved"]) is True


def test_preprocess_includes_extended_optional_fields() -> None:
    raw = pd.DataFrame(
        {
            "Incident Number": ["INC2001"],
            "Short Description": ["VPN password reset needed"],
            "Assignment Group": ["Service Desk"],
            "Application": ["Identity"],
            "State": ["Open"],
            "Priority": ["3"],
            "Open Date": ["2026-02-01 09:00:00"],
            "Domain": ["Business Function"],
            "Sub Domain": ["IAM"],
            "Sub Category": ["Password Reset"],
            "Contact Type": ["Email"],
            "Customer": ["Acme"],
            "Environment": ["UAT"],
            "Company": ["Coforge"],
            "Department": ["IT"],
            "Requester Job Title": ["Analyst"],
        }
    )

    output = preprocess_tickets(raw)
    row = output.iloc[0]

    assert row["domain"] == "Business Function"
    assert row["sub_domain"] == "IAM"
    assert row["subcategory"] == "Password Reset"
    assert row["contact_channel"] == "Email"
    assert row["customer"] == "Acme"
    assert row["environment"] == "UAT"
    assert row["company"] == "Coforge"
    assert row["department"] == "IT"
    assert row["requester_job_title"] == "Analyst"
