from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest


@pytest.fixture
def reference_time() -> datetime:
    return datetime(2026, 1, 15, 12, 0, 0)


@pytest.fixture
def raw_ticket_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Incident ID": ["INC001", "INC002", "INC003", "INC004", "INC005", "INC006"],
            "Short Description": [
                "VPN access issue for new employee",
                "Laptop hardware failure",
                "Email outage in finance mailbox",
                "Database timeout on ERP query",
                "CRM application is very slow",
                "Password reset request",
            ],
            "Assignment Group": ["Network Team", "Desktop Team", "Messaging Team", "DBA Team", "App Support", "Service Desk"],
            "State": ["Closed", "Resolved", "Closed", "Open", "Closed", "Open"],
            "Severity": ["1", "2", "3", "2", "4", "3"],
            "Business Service": ["VPN", "Workstation", "Exchange", "ERP", "Salesforce", "Identity"],
            "Opened At": [
                "2026-01-01 08:00:00",
                "2026-01-02 10:00:00",
                "2026-01-03 09:00:00",
                "2026-01-04 08:00:00",
                "2026-01-05 07:30:00",
                "2026-01-06 11:15:00",
            ],
            "Resolved At": [
                "2026-01-01 11:00:00",
                "2026-01-02 21:00:00",
                "2026-01-04 15:00:00",
                None,
                "2026-01-05 09:30:00",
                None,
            ],
            "Resolution Duration": ["3h", "11:00", "1d 6h", None, "2", "1h"],
            "Reopened": [0, 1, 0, 0, 2, 0],
        }
    )
