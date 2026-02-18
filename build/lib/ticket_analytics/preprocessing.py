"""Data ingestion and preprocessing helpers for ITIL ticket dumps."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import COLUMN_ALIASES, PRIORITY_MAP, RESOLVED_STATUSES


def normalize_column_name(column: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(column).strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _canonical_alias_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            mapping[normalize_column_name(alias)] = canonical
    return mapping


def normalize_and_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [normalize_column_name(col) for col in result.columns]

    alias_to_canonical = _canonical_alias_map()
    renamed: dict[str, str] = {}
    existing = set(result.columns)
    for col in result.columns:
        canonical = alias_to_canonical.get(col)
        if canonical is None:
            continue
        if canonical == col:
            continue
        if canonical in existing or canonical in renamed.values():
            continue
        renamed[col] = canonical

    if renamed:
        result = result.rename(columns=renamed)

    return result


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_duration_to_hours(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().lower()
    if not text:
        return np.nan

    # hh:mm[:ss]
    if re.fullmatch(r"\d{1,3}:\d{2}(:\d{2})?", text):
        parts = [int(p) for p in text.split(":")]
        if len(parts) == 2:
            h, m = parts
            s = 0
        else:
            h, m, s = parts
        return h + (m / 60.0) + (s / 3600.0)

    # e.g. '1d 2h 30m'
    total_hours = 0.0
    matched = False
    for amount, unit in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*([a-z]+)", text):
        amount_f = float(amount)
        matched = True
        if unit.startswith("d"):
            total_hours += amount_f * 24
        elif unit.startswith("h"):
            total_hours += amount_f
        elif unit.startswith("m"):
            total_hours += amount_f / 60
        elif unit.startswith("s"):
            total_hours += amount_f / 3600

    if matched:
        return total_hours

    numeric = re.findall(r"[0-9]+(?:\.[0-9]+)?", text)
    if numeric:
        return float(numeric[0])

    return np.nan


def _standardize_priority(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown"
    key = normalize_column_name(str(value))
    return PRIORITY_MAP.get(key, str(value).strip().upper() if str(value).strip() else "Unknown")


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column not in result:
            result[column] = np.nan
    return result


def preprocess_tickets(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_and_alias_columns(raw_df)
    required_columns = [
        "ticket_id",
        "description",
        "team",
        "status",
        "priority",
        "service",
        "category",
        "created_at",
        "resolved_at",
        "resolution_time",
        "reopen_count",
    ]
    df = _ensure_columns(df, required_columns)

    if df["ticket_id"].isna().all():
        df["ticket_id"] = [f"TKT-{i + 1:06d}" for i in range(len(df))]
    else:
        df["ticket_id"] = df["ticket_id"].fillna(method="ffill")
        df["ticket_id"] = df["ticket_id"].fillna([f"TKT-{i + 1:06d}" for i in range(len(df))])

    for dt_col in ["created_at", "resolved_at", "updated_at", "closed_at"]:
        if dt_col in df:
            df[dt_col] = _safe_to_datetime(df[dt_col])

    if "resolved_at" in df and df["resolved_at"].isna().all() and "closed_at" in df:
        df["resolved_at"] = df["closed_at"]

    if "resolution_time" in df:
        df["resolution_time_hours"] = df["resolution_time"].map(parse_duration_to_hours)
    else:
        df["resolution_time_hours"] = np.nan

    for col in ["description", "team", "service", "category", "status"]:
        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace({"": "Unknown"})
        )

    df["priority"] = df["priority"].map(_standardize_priority)

    df["reopen_count"] = _coerce_numeric(df["reopen_count"]).fillna(0).astype(int)

    normalized_status = df["status"].astype(str).str.lower().str.replace(r"\s+", "_", regex=True)
    df["is_resolved"] = normalized_status.isin(RESOLVED_STATUSES) | df["resolved_at"].notna()

    return df
