"""Workspace and mapping helpers for the PRD flow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .constants import COLUMN_ALIASES, GLOBAL_FILTER_CANDIDATES, PRD_MANDATORY_FIELDS
from .preprocessing import normalize_column_name


@dataclass
class UploadHistoryEntry:
    file_name: str
    rows: int
    uploaded_at: str


def suggest_mapping_candidates(df: pd.DataFrame) -> dict[str, list[str]]:
    original_columns = [str(col) for col in df.columns]
    normalized_to_original = {normalize_column_name(col): str(col) for col in original_columns}
    normalized_cols = list(normalized_to_original.keys())
    mapping: dict[str, list[str]] = {}

    for canonical in PRD_MANDATORY_FIELDS:
        expected_tokens = set(canonical.split("_"))
        alias_set = {normalize_column_name(alias) for alias in COLUMN_ALIASES.get(canonical, [])}
        alias_tokens = set()
        for alias in alias_set:
            alias_tokens.update(alias.split("_"))
        scored: list[tuple[int, str]] = []
        for col in normalized_cols:
            tokens = set(col.split("_"))
            score = 0
            if col in alias_set:
                score += 100
            score += 10 * len(expected_tokens & tokens)
            score += 6 * len(alias_tokens & tokens)
            if any(alias in col or col in alias for alias in alias_set):
                score += 12
            if canonical in col:
                score += 25
            scored.append((score, col))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        mapping[canonical] = [
            normalized_to_original[item[1]]
            for item in scored
            if item[0] > 0
        ][:5]

    return mapping


def build_upload_history_entry(file_name: str, row_count: int) -> UploadHistoryEntry:
    return UploadHistoryEntry(file_name=file_name, rows=row_count, uploaded_at=datetime.now().isoformat())


def suggest_global_filters(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []

    candidates: list[str] = []
    for col in GLOBAL_FILTER_CANDIDATES:
        if col not in df.columns:
            continue
        unique_count = int(df[col].nunique(dropna=True))
        # PRD guidance: allow columns with manageable cardinality (~10-15% of rows).
        max_values = max(5, int(len(df) * 0.15))
        if 1 < unique_count <= max_values:
            candidates.append(col)

    return candidates
