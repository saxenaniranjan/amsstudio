from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from ticket_analytics.insights import build_insight_report, find_volume_anomalies
from ticket_analytics.pipeline import run_ticket_pipeline


def test_build_insight_report(raw_ticket_df, reference_time) -> None:
    enriched = run_ticket_pipeline(raw_ticket_df, reference_time=reference_time)
    report = build_insight_report(enriched)

    assert "metrics" in report
    assert "recommendations" in report
    assert "team_summary" in report
    assert report["metrics"]["total_tickets"] == len(enriched)
    assert len(report["recommendations"]) >= 1


def test_find_volume_anomalies_detects_spike() -> None:
    base = datetime(2026, 1, 1)
    rows = []
    for i in range(20):
        rows.append({"created_at": base + timedelta(days=i // 2)})
    for _ in range(20):
        rows.append({"created_at": base + timedelta(days=15)})

    df = pd.DataFrame(rows)
    anomalies = find_volume_anomalies(df)
    assert any(item["date"] == "2026-01-16" for item in anomalies)
