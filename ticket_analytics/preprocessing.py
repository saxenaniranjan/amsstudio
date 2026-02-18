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


def _guess_canonical_column(column: str) -> str | None:
    tokens = set(column.split("_"))

    if "id" in tokens and {"ticket", "incident", "case", "task"} & tokens:
        return "ticket_id"
    if {"ticket", "incident", "request"} & tokens and "type" in tokens:
        return "ticket_type"
    if "assignment" in tokens and "group" in tokens:
        return "team"
    if {"support", "resolver"} & tokens and {"team", "group"} & tokens:
        return "team"
    if {"description", "title", "summary"} & tokens:
        return "description"
    if {"status", "state"} & tokens:
        return "status"
    if {"priority", "severity", "urgency"} & tokens:
        return "priority"
    if {"service", "application", "system"} & tokens:
        return "service"
    if "sub" in tokens and "category" in tokens:
        return "subcategory"
    if {"subcategory", "subcategory_name"} & tokens:
        return "subcategory"
    if "category" in tokens:
        return "category"
    if {"opened", "open", "created"} & tokens and {"at", "on", "date", "time", "datetime"} & tokens:
        return "created_at"
    if {"resolved", "closed", "completed", "completion"} & tokens and {
        "at",
        "on",
        "date",
        "time",
        "datetime",
    } & tokens:
        return "resolved_at"
    if "resolution" in tokens and {"time", "duration", "age"} & tokens:
        return "resolution_time"
    if {"reopen", "reopened", "reopens"} & tokens:
        return "reopen_count"
    if {"assignee", "assigned", "resolver", "owner"} & tokens:
        return "assignee"
    if {"response", "reply"} & tokens and "sla" in tokens:
        return "response_sla"
    if {"resolution", "resolve"} & tokens and "sla" in tokens:
        return "resolution_sla"
    if "cluster" in tokens or "tower" in tokens or "workstream" in tokens:
        return "cluster"
    if "domain" in tokens and "sub" in tokens:
        return "sub_domain"
    if "domain" in tokens:
        return "domain"
    if "channel" in tokens:
        return "contact_channel"
    if "customer" in tokens or "client" in tokens:
        return "customer"
    if "environment" in tokens or "env" in tokens:
        return "environment"
    if "company" in tokens or "organization" in tokens:
        return "company"
    if "department" in tokens or "dept" in tokens:
        return "department"
    if {"job", "title"} & tokens and {"requestor", "requester", "requested"} & tokens:
        return "requester_job_title"
    if "location" in tokens or "site" in tokens:
        return "location"
    return None


def _canonical_alias_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            mapping[normalize_column_name(alias)] = canonical
    return mapping


def normalize_and_alias_columns(df: pd.DataFrame, user_mapping: dict[str, str] | None = None) -> pd.DataFrame:
    result = df.copy()
    original_normalized = [normalize_column_name(col) for col in result.columns]
    result.columns = original_normalized

    # User mapping takes priority over aliases.
    if user_mapping:
        normalized_mapping: dict[str, str] = {}
        for source, target in user_mapping.items():
            source_norm = normalize_column_name(source)
            target_norm = normalize_column_name(target)
            normalized_mapping[source_norm] = target_norm
        rename_by_user = {
            source_col: normalized_mapping[source_col]
            for source_col in result.columns
            if source_col in normalized_mapping
        }
        if rename_by_user:
            result = result.rename(columns=rename_by_user)

    alias_to_canonical = _canonical_alias_map()
    renamed: dict[str, str] = {}
    existing = set(result.columns)
    for col in result.columns:
        canonical = alias_to_canonical.get(col)
        if canonical is None:
            canonical = _guess_canonical_column(col)
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
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    unresolved_mask = parsed.isna() & series.notna()
    if unresolved_mask.any():
        parsed_alt = pd.to_datetime(series[unresolved_mask], errors="coerce", utc=False, dayfirst=True)
        parsed.loc[unresolved_mask] = parsed_alt
    unresolved_mask = parsed.isna() & series.notna()
    if unresolved_mask.any():
        numeric = pd.to_numeric(series[unresolved_mask], errors="coerce")
        numeric = numeric[numeric.notna()]
        if not numeric.empty:
            excel_dates = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
            parsed.loc[excel_dates.index] = excel_dates
    return parsed


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_duration_to_hours(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan

    if isinstance(value, pd.Timedelta):
        return value.total_seconds() / 3600.0
    if isinstance(value, np.timedelta64):
        return pd.to_timedelta(value).total_seconds() / 3600.0

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().lower()
    if not text:
        return np.nan

    timedelta_value = pd.to_timedelta(text, errors="coerce")
    if not pd.isna(timedelta_value):
        return timedelta_value.total_seconds() / 3600.0

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


def preprocess_tickets(raw_df: pd.DataFrame, user_mapping: dict[str, str] | None = None) -> pd.DataFrame:
    df = normalize_and_alias_columns(raw_df, user_mapping=user_mapping)
    required_columns = [
        "ticket_id",
        "ticket_type",
        "description",
        "team",
        "status",
        "priority",
        "service",
        "category",
        "subcategory",
        "created_at",
        "resolved_at",
        "updated_at",
        "resolution_time",
        "response_sla",
        "resolution_sla",
        "reopen_count",
        "assignee",
        "closure_notes",
        "cluster",
        "domain",
        "sub_domain",
        "contact_channel",
        "customer",
        "environment",
        "company",
        "affected_system",
        "department",
        "requester_job_title",
        "location",
    ]
    df = _ensure_columns(df, required_columns)

    fallback_ids = pd.Series([f"TKT-{i + 1:06d}" for i in range(len(df))], index=df.index)
    df["ticket_id"] = df["ticket_id"].replace(r"^\s*$", np.nan, regex=True)
    df["ticket_id"] = df["ticket_id"].ffill().fillna(fallback_ids).astype(str)

    for dt_col in ["created_at", "resolved_at", "updated_at", "closed_at"]:
        if dt_col in df:
            df[dt_col] = _safe_to_datetime(df[dt_col])

    # Precedence from PRD: resolved time preferred; fallback to closed/update timestamps.
    if "closed_at" in df:
        no_resolved = df["resolved_at"].isna() & df["closed_at"].notna()
        df.loc[no_resolved, "resolved_at"] = df.loc[no_resolved, "closed_at"]

    if "resolution_time" in df:
        df["resolution_time_hours"] = df["resolution_time"].map(parse_duration_to_hours)
    else:
        df["resolution_time_hours"] = np.nan

    if df["created_at"].isna().any() and "updated_at" in df:
        missing_created = df["created_at"].isna() & df["updated_at"].notna()
        df.loc[missing_created, "created_at"] = df.loc[missing_created, "updated_at"]

    text_columns = [
        "ticket_type",
        "description",
        "team",
        "service",
        "category",
        "subcategory",
        "status",
        "assignee",
        "closure_notes",
        "cluster",
        "domain",
        "sub_domain",
        "contact_channel",
        "customer",
        "environment",
        "company",
        "affected_system",
        "department",
        "requester_job_title",
        "location",
    ]
    for col in text_columns:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})

    df["priority"] = df["priority"].map(_standardize_priority)
    df["reopen_count"] = _coerce_numeric(df["reopen_count"]).fillna(0).astype(int)

    normalized_status = df["status"].astype(str).str.lower().str.replace(r"\s+", "_", regex=True)
    if "updated_at" in df:
        inferred_resolved_mask = df["resolved_at"].isna() & normalized_status.isin(RESOLVED_STATUSES)
        df.loc[inferred_resolved_mask, "resolved_at"] = df.loc[inferred_resolved_mask, "updated_at"]

    df["is_resolved"] = normalized_status.isin(RESOLVED_STATUSES) | df["resolved_at"].notna()

    return df
