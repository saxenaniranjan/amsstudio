"""Pipeline orchestration for ticket analytics platform."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .features import derive_features
from .insights import build_insight_report
from .preprocessing import preprocess_tickets
from .query_engine import QueryResult, answer_query


def run_ticket_pipeline(
    raw_df: pd.DataFrame,
    reference_time: datetime | None = None,
    user_mapping: dict[str, str] | None = None,
    sla_threshold_hours: dict[str, float] | None = None,
) -> pd.DataFrame:
    preprocessed = preprocess_tickets(raw_df, user_mapping=user_mapping)
    enriched = derive_features(
        preprocessed,
        reference_time=reference_time,
        sla_threshold_hours=sla_threshold_hours,
    )
    return enriched


@dataclass
class TicketAnalysisSession:
    enriched_df: pd.DataFrame

    @classmethod
    def from_dataframe(
        cls,
        raw_df: pd.DataFrame,
        reference_time: datetime | None = None,
        user_mapping: dict[str, str] | None = None,
        sla_threshold_hours: dict[str, float] | None = None,
    ) -> "TicketAnalysisSession":
        return cls(
            run_ticket_pipeline(
                raw_df,
                reference_time=reference_time,
                user_mapping=user_mapping,
                sla_threshold_hours=sla_threshold_hours,
            )
        )

    @classmethod
    def from_file(
        cls,
        path: str,
        reference_time: datetime | None = None,
        user_mapping: dict[str, str] | None = None,
        sla_threshold_hours: dict[str, float] | None = None,
    ) -> "TicketAnalysisSession":
        file_lower = path.lower()
        if file_lower.endswith(".csv"):
            raw_df = pd.read_csv(path)
        else:
            raw_df = pd.read_excel(path)
        return cls.from_dataframe(
            raw_df,
            reference_time=reference_time,
            user_mapping=user_mapping,
            sla_threshold_hours=sla_threshold_hours,
        )

    def report(self) -> dict:
        return build_insight_report(self.enriched_df)

    def ask(self, query: str) -> QueryResult:
        return answer_query(query, self.enriched_df)
