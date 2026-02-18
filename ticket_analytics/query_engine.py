"""Agentic query engine with planner, graph builder, and validator agents."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import Figure

from .constants import CHART_KEYWORDS, OPEN_STATUSES, PRIORITY_MAP, RESOLVED_STATUSES, SLA_THRESHOLD_HOURS
from .insights import build_insight_report
from .preprocessing import parse_duration_to_hours


@dataclass
class QueryResult:
    kind: str
    text: str
    data: pd.DataFrame | None = None
    chart: dict[str, Any] | None = None
    agent_trace: dict[str, Any] | None = None


@dataclass
class AgenticQueryIntent:
    request_kind: str
    chart_type: str
    metric: str
    dimension: str | None
    time_grain: str
    filters: dict[str, list[str]]
    lookback_value: int | None
    lookback_unit: str | None
    top_n: int | None
    source: str
    series_dimension: str | None = None


@dataclass
class BuildOutput:
    kind: str
    text: str
    data: pd.DataFrame | None
    figure: Figure | None
    chart_type: str | None
    chart_title: str | None
    filtered_rows: int
    total_rows: int
    applied_filters: dict[str, list[str]]
    date_window: dict[str, str | None]
    missing_requirements: list[str]
    data_unavailable: bool


_METRIC_LABELS = {
    "ticket_count": "Ticket Count",
    "avg_mttr_hours": "Average MTTR (hours)",
    "sla_compliance_pct": "SLA Compliance (%)",
    "breach_rate_pct": "SLA Breach Rate (%)",
    "breached_tickets": "Breached Tickets",
    "open_tickets": "Open Tickets",
    "team_performance_index": "Team Performance Index",
}

_METRIC_REQUIREMENTS = {
    "ticket_count": [],
    "avg_mttr_hours": ["mttr_hours"],
    "sla_compliance_pct": ["is_sla_breached"],
    "breach_rate_pct": ["is_sla_breached"],
    "breached_tickets": ["is_sla_breached"],
    "open_tickets": ["is_open"],
    "team_performance_index": ["team_performance_index"],
}

_LOOKBACK_UNITS = {
    "day": "D",
    "days": "D",
    "week": "W",
    "weeks": "W",
    "month": "M",
    "months": "M",
    "quarter": "Q",
    "quarters": "Q",
    "year": "Y",
    "years": "Y",
}

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = value.strip()
        if not token:
            continue
        marker = token.lower()
        if marker in seen:
            continue
        seen.add(marker)
        out.append(token)
    return out


def _matching_values_by_terms(series: pd.Series, terms: list[str]) -> list[str]:
    normalized_terms = [_normalize_text(term) for term in terms if _normalize_text(term)]
    if not normalized_terms:
        return []
    matches: list[str] = []
    for value in _value_candidates(series, max_values=400):
        token = _normalize_text(value)
        if not token:
            continue
        if any(term in token for term in normalized_terms):
            matches.append(value)
    return _dedupe_keep_order(matches)


def _contains_time_axis_phrase(query: str) -> bool:
    q = query.lower()
    phrase_patterns = [
        r"\bover(?:\s+all)?\s+time\b",
        r"\bacross\s+time\b",
        r"\ball\s+time\b",
        r"\btime\s+series\b",
        r"\btimeline\b",
        r"\bdaywise\b",
        r"\bweekwise\b",
        r"\bmonthwise\b",
        r"\btrend\b",
    ]
    return any(re.search(pattern, q) is not None for pattern in phrase_patterns)


def _extract_json_blob(text: str) -> Optional[dict[str, Any]]:
    candidate = text.strip()
    if not candidate:
        return None

    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?", "", candidate).strip()
        candidate = re.sub(r"```$", "", candidate).strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    chunk = candidate[start : end + 1]
    try:
        parsed = json.loads(chunk)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _detect_chart_type(query: str) -> str:
    lowered = query.lower()
    for keyword, chart in CHART_KEYWORDS.items():
        if keyword in lowered:
            return chart
    if any(x in lowered for x in ["graph", "plot", "visualize", "chart"]):
        return "bar"
    if any(x in lowered for x in ["trend", "over time", "timeline"]):
        return "line"
    return "bar"


def _detect_metric(query: str) -> str:
    q = query.lower()
    if any(
        k in q
        for k in [
            "sla compliance",
            "compliance %",
            "sla adherence",
            "adherence %",
            "within sla",
            "sla met",
            "met sla",
        ]
    ):
        return "sla_compliance_pct"
    if any(k in q for k in ["mttr", "mean time to resolve", "resolution time", "mean resolution time", "turnaround time"]):
        return "avg_mttr_hours"
    if any(k in q for k in ["breach rate", "sla breach rate", "sla breach %", "breach %"]):
        return "breach_rate_pct"
    if any(k in q for k in ["breached", "sla violations", "sla breach count", "breach count", "sla breach trend"]):
        return "breached_tickets"
    if any(k in q for k in ["open ticket", "open tickets", "backlog", "unresolved", "pending tickets", "aging tickets"]):
        return "open_tickets"
    if any(k in q for k in ["team performance", "performance index", "underperforming teams", "team efficiency"]):
        return "team_performance_index"
    return "ticket_count"


def _infer_time_grain(query: str, lookback_unit: str | None, lookback_value: int | None = None) -> str:
    q = query.lower()
    if "daily" in q or "per day" in q or "daywise" in q:
        return "D"
    if "hour" in q:
        return "D"
    if "weekly" in q or "per week" in q:
        return "W"
    if "monthly" in q or "per month" in q:
        return "M"
    if "quarter" in q:
        return "Q"
    if "year" in q:
        return "Y"
    if lookback_unit in {"day", "days"}:
        return "D"
    if lookback_unit in {"week", "weeks"}:
        if lookback_value is not None and lookback_value <= 2:
            return "D"
        return "W"
    if lookback_unit in {"month", "months"}:
        if lookback_value is not None and lookback_value <= 1:
            return "D"
        if lookback_value is not None and lookback_value <= 3:
            return "W"
        return "M"
    if lookback_unit in {"quarter", "quarters"}:
        if lookback_value is not None and lookback_value <= 1:
            return "W"
        return "Q"
    if lookback_unit in {"year", "years"}:
        if lookback_value is not None and lookback_value <= 2:
            return "M"
        return "Y"
    return "M"


def _detect_dimension(query: str, df: pd.DataFrame) -> str | None:
    q = query.lower()

    if "service desk" in q and "team" in df.columns:
        return "team"

    lookback_phrase = re.search(
        r"\b(?:last|past)\s+(?:\d+\s+)?(?:day|days|week|weeks|month|months|quarter|quarters|year|years)\b",
        q,
    )

    explicit_patterns = [
        (r"\b(by team|per team|team wise|assignment groups?|groupwise|by assignment group|per assignment group)\b", "team"),
        (r"\b(by category|per category|issue category|recurring category|category wise)\b", "category_derived"),
        (r"\b(by priority|per priority|priority wise|priority trend|priority breakdown)\b", "priority"),
        (r"\b(by status|per status|status wise)\b", "status"),
        (r"\b(by service|per service|service wise|by application|per application)\b", "service"),
        (r"\b(by function|per function|business function|function wise)\b", "business_function_derived"),
        (r"\b(by cluster|per cluster|cluster wise|by tower|tower wise|workstream wise)\b", "cluster"),
    ]

    matches: list[tuple[int, str]] = []
    if "service" in df.columns:
        service_match = re.search(r"\b(application|applications|app|apps)\b", q)
        if service_match:
            matches.append((service_match.start(), "service"))

    for pattern, column in explicit_patterns:
        if column not in df.columns:
            continue
        match = re.search(pattern, q)
        if match:
            matches.append((match.start(), column))

    if matches:
        matches.sort(key=lambda item: item[0])
        return matches[0][1]

    if _contains_time_axis_phrase(q):
        if "created_at" in df.columns:
            return "created_at"

    if lookback_phrase:
        if "created_at" in df.columns:
            return "created_at"

    for candidate in ["team", "service", "priority", "category_derived"]:
        if candidate in df.columns:
            return candidate

    return None


def _detect_series_dimension(query: str, df: pd.DataFrame, primary_dimension: str | None) -> str | None:
    q = query.lower()
    if not primary_dimension:
        return None

    if "priority" in df.columns and primary_dimension != "priority":
        priority_patterns = [
            r"\bacross all priorit(?:y|ies)\b",
            r"\bsplit by priority\b",
            r"\bpriority(?:\s|-)?wise\b",
            r"\bpriority labelled\b",
            r"\bpriority breakdown\b",
            r"\bcolor by priority\b",
            r"\bcolour by priority\b",
            r"\bdifferent colou?rs?\b",
            r"\bstacked\b",
        ]
        priority_values = re.findall(r"\bp\s*([1-4])\b", q)
        unique_priority_values = sorted(set(priority_values))
        has_priority_values = len(unique_priority_values) > 0
        comparative_phrase = re.search(r"\b(across|for|between|vs|versus|by)\b", q) is not None
        if (
            any(re.search(pattern, q) for pattern in priority_patterns)
            or (has_priority_values and "priority" in q)
            or (len(unique_priority_values) >= 2 and comparative_phrase)
        ):
            return "priority"

    if "status" in df.columns and primary_dimension != "status":
        status_patterns = [
            r"\bsplit by status\b",
            r"\bstatus(?:\s|-)?wise\b",
            r"\bstatus breakdown\b",
            r"\bcolor by status\b",
            r"\bcolour by status\b",
        ]
        if any(re.search(pattern, q) for pattern in status_patterns):
            return "status"

    if "team" in df.columns and primary_dimension != "team":
        team_patterns = [
            r"\bsplit by team\b",
            r"\bteam(?:\s|-)?wise\b",
            r"\bassignment group(?:\s|-)?wise\b",
            r"\bcolor by team\b",
            r"\bcolour by team\b",
        ]
        if any(re.search(pattern, q) for pattern in team_patterns):
            return "team"

    return None


def _extract_top_n(query: str) -> int | None:
    q = query.lower()

    def _to_int(token: str) -> int | None:
        token = token.strip().lower()
        if not token:
            return None
        if token.isdigit():
            return int(token)
        return _NUMBER_WORDS.get(token)

    patterns = [
        r"\btop\s+([a-z0-9]+)\b",
        r"\b(?:highest|largest|biggest|worst)\s+([a-z0-9]+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if not match:
            continue
        parsed = _to_int(match.group(1))
        if parsed is not None:
            return max(1, parsed)

    return None


def _extract_lookback(query: str) -> tuple[int | None, str | None]:
    q = query.lower()

    if re.search(r"\b(mtd|month to date)\b", q):
        return 1, "month"
    if re.search(r"\b(qtd|quarter to date)\b", q):
        return 1, "quarter"
    if re.search(r"\b(ytd|year to date)\b", q):
        return 1, "year"

    current = re.search(r"\b(this|current)\s+(day|week|month|quarter|year)\b", q)
    if current:
        return 1, current.group(2)

    match = re.search(r"(?:last|past)\s+(\d+)\s+(day|days|week|weeks|month|months|quarter|quarters|year|years)", q)
    if not match:
        singular = re.search(r"\b(?:last|past)\s+(day|week|month|quarter|year)\b", q)
        if singular:
            return 1, singular.group(1)
        return None, None
    return int(match.group(1)), match.group(2)


def _value_candidates(series: pd.Series, max_values: int = 40) -> list[str]:
    values = [str(v) for v in series.dropna().astype(str).unique().tolist() if str(v).strip()]
    values.sort(key=lambda x: (len(x), x))
    return values[:max_values]


def _add_filter_values(filters: dict[str, list[str]], column: str, values: list[str]) -> None:
    merged = filters.get(column, []) + values
    filters[column] = _dedupe_keep_order(merged)


def _extract_filters(query: str, df: pd.DataFrame) -> dict[str, list[str]]:
    qnorm = f" {_normalize_text(query)} "
    qlower = query.lower()
    filters: dict[str, list[str]] = {}

    priority_match = re.findall(r"\bp\s*([1-4])\b", qlower)
    if priority_match and "priority" in df.columns:
        _add_filter_values(filters, "priority", [f"P{value}" for value in priority_match])

    if "priority" in df.columns:
        descriptor_map = {
            "critical": "P1",
            "high": "P2",
            "medium": "P3",
            "moderate": "P3",
            "low": "P4",
            "minor": "P4",
        }
        semantic_priorities = [mapped for marker, mapped in descriptor_map.items() if re.search(rf"\b{marker}\b", qlower)]
        if semantic_priorities:
            _add_filter_values(filters, "priority", semantic_priorities)

        explicit_priority_level = re.findall(r"\b(?:sev(?:erity)?|priority)\s*[-_: ]*([1-4])\b", qlower)
        if explicit_priority_level:
            _add_filter_values(filters, "priority", [f"P{level}" for level in explicit_priority_level])

    explicit_patterns: list[tuple[str, str]] = [
        (r"(?:for|in|by)\s+(?:team|assignment group)\s+([a-z0-9 _\\-]+)", "team"),
        (r"(?:for|in|by)\s+(?:application|business service|service(?!\s+desk))\s+([a-z0-9 _\\-]+)", "service"),
        (r"(?:for|in|by)\s+priority\s+([a-z0-9 _\\-]+)", "priority"),
    ]
    for pattern, column in explicit_patterns:
        if column not in df.columns:
            continue
        match = re.search(pattern, qlower)
        if not match:
            continue
        captured = match.group(1).strip().replace(" tickets only", "").replace(" only", "")
        captured = re.split(
            r"\b(?:over|across|with|where|who|that|during|from|between|last|past|this|current)\b",
            captured,
            maxsplit=1,
        )[0]
        captured = re.sub(r"\s+", " ", captured)
        if not captured:
            continue

        parsed_values: list[str]
        if column == "priority":
            inferred: set[str] = {f"P{x}" for x in re.findall(r"\bp\s*([1-4])\b", captured, flags=re.IGNORECASE)}
            lowered = f" {captured.lower()} "
            descriptor_map = {
                "critical": "P1",
                "high": "P2",
                "medium": "P3",
                "moderate": "P3",
                "low": "P4",
                "minor": "P4",
            }
            for marker, mapped in descriptor_map.items():
                if f" {marker} " in lowered:
                    inferred.add(mapped)
            parsed_values = sorted(inferred) if inferred else [captured]
        else:
            parsed_values = [part.strip() for part in re.split(r"\s*(?:,|/|&| and )\s*", captured) if part.strip()]
            if not parsed_values:
                parsed_values = [captured]

        _add_filter_values(filters, column, parsed_values)

    if "ticket_type" in df.columns:
        type_keywords: list[tuple[list[str], list[str]]] = [
            (["incident", "inc"], [r"\bincident(?:s)?\b", r"\bincident ticket(?:s)?\b"]),
            (["service request", "sr"], [r"\bservice request(?:s)?\b", r"\bservice req(?:uest)?(?:s)?\b", r"\bsr\b"]),
            (["change request", "change", "crq"], [r"\bchange request(?:s)?\b", r"\bchange ticket(?:s)?\b", r"\bcrq\b"]),
            (["problem", "prb"], [r"\bproblem(?:s)?\b", r"\bproblem ticket(?:s)?\b", r"\bprb\b"]),
        ]
        semantic_type_terms: list[str] = []
        for terms, patterns in type_keywords:
            if any(re.search(pattern, qlower) for pattern in patterns):
                semantic_type_terms.extend(terms)
        if semantic_type_terms:
            ticket_type_matches = _matching_values_by_terms(df["ticket_type"], semantic_type_terms)
            if ticket_type_matches:
                _add_filter_values(filters, "ticket_type", ticket_type_matches)

    open_signals = [
        r"\bopen\b",
        r"\bunresolved\b",
        r"\bbacklog\b",
        r"\bon hold\b",
        r"\bin progress\b",
        r"\bpending\b",
        r"\breopened\b",
        r"\bactive\b",
    ]
    resolved_signals = [
        r"\bresolved\b",
        r"\bclosed\b",
        r"\bcompleted\b",
        r"\bfulfilled\b",
    ]
    has_open_signal = any(re.search(pattern, qlower) for pattern in open_signals)
    has_resolved_signal = any(re.search(pattern, qlower) for pattern in resolved_signals)
    if has_open_signal and not has_resolved_signal and "is_open" in df.columns:
        _add_filter_values(filters, "is_open", ["True"])
    if has_resolved_signal and not has_open_signal and "is_resolved" in df.columns:
        _add_filter_values(filters, "is_resolved", ["True"])

    if "status" in df.columns:
        status_terms: list[str] = []
        if re.search(r"\bon hold\b", qlower):
            status_terms.extend(["on hold", "hold"])
        if re.search(r"\bin progress\b", qlower):
            status_terms.extend(["in progress", "in_progress"])
        if re.search(r"\bpending\b", qlower):
            status_terms.append("pending")
        if re.search(r"\breopened\b", qlower):
            status_terms.append("reopened")
        if has_open_signal and not has_resolved_signal:
            status_terms.extend([*OPEN_STATUSES, "open"])
        if has_resolved_signal and not has_open_signal:
            status_terms.extend([*RESOLVED_STATUSES, "resolved", "closed"])
        if status_terms:
            status_matches = _matching_values_by_terms(df["status"], status_terms)
            if status_matches:
                _add_filter_values(filters, "status", status_matches)

    candidate_columns = [
        "team",
        "category_derived",
        "business_function_derived",
        "priority",
        "status",
        "service",
        "cluster",
        "domain",
        "sub_domain",
    ]

    for column in candidate_columns:
        if column not in df.columns:
            continue
        matches: list[str] = []
        for value in _value_candidates(df[column]):
            token = _normalize_text(value)
            if not token:
                continue
            if f" {token} " in qnorm:
                matches.append(value)
        if matches:
            _add_filter_values(filters, column, matches)

    return filters


def _merge_filters(primary: dict[str, list[str]], secondary: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for column in sorted(set(primary.keys()) | set(secondary.keys())):
        values: list[str] = []
        seen: set[str] = set()
        for candidate in primary.get(column, []) + secondary.get(column, []):
            value = str(candidate).strip()
            if not value:
                continue
            marker = value.lower()
            if marker in seen:
                continue
            seen.add(marker)
            values.append(value)
        if values:
            merged[column] = values
    return merged


def _detect_request_kind(query: str) -> str:
    lowered = query.lower()
    if any(k in lowered for k in ["recommend", "improve", "what should", "action", "suggest"]):
        return "text_recommendations"
    if any(k in lowered for k in ["graph", "chart", "plot", "visualize", "trend", "timeline"]):
        return "graph"
    if _contains_time_axis_phrase(lowered):
        return "graph"
    return "table"


def _has_trend_phrase(query: str) -> bool:
    return _contains_time_axis_phrase(query)


def _has_explicit_chart_type(query: str) -> bool:
    q = query.lower()
    return any(keyword in q for keyword in CHART_KEYWORDS)


def _has_explicit_dimension_phrase(query: str) -> bool:
    q = query.lower()
    patterns = [
        r"\b(by team|per team|team wise|assignment groups?|groupwise|by assignment group|per assignment group)\b",
        r"\b(by category|per category|issue category|recurring category|category wise)\b",
        r"\b(by priority|per priority|priority wise|priority trend|priority breakdown)\b",
        r"\b(by status|per status|status wise)\b",
        r"\b(by service|per service|service wise|by application|per application)\b",
        r"\b(by function|per function|business function|function wise)\b",
        r"\b(by cluster|per cluster|cluster wise|by tower|tower wise|workstream wise)\b",
    ]
    return any(re.search(pattern, q) is not None for pattern in patterns)


def _openai_client() -> tuple[Any | None, str | None]:
    # Keep automated tests deterministic and fast even when .env includes a live API key.
    if os.getenv("PYTEST_CURRENT_TEST"):
        return None, None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        model = os.getenv("TICKETX_LLM_MODEL", "gpt-4.1-mini")
        return client, model
    except Exception:
        return None, None


def llm_status() -> dict[str, Any]:
    model = os.getenv("TICKETX_LLM_MODEL", "gpt-4.1-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"enabled": False, "model": model, "reason": "OPENAI_API_KEY not configured"}
    try:
        import openai  # noqa: F401

        return {"enabled": True, "model": model, "reason": None}
    except Exception as exc:
        return {"enabled": False, "model": model, "reason": f"OpenAI client unavailable: {exc}"}


class IntentAgent:
    """Planner agent: converts user query into an execution intent."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.client, self.model = _openai_client()

    def plan(self, query: str) -> AgenticQueryIntent:
        llm_plan = self._plan_with_llm(query)
        if llm_plan is not None:
            return llm_plan
        return self._plan_with_rules(query)

    def _refine_with_rules(self, query: str, intent: AgenticQueryIntent) -> AgenticQueryIntent:
        lookback_value, lookback_unit = _extract_lookback(query)
        rule_filters = _extract_filters(query, self.df)
        rule_metric = _detect_metric(query)
        rule_dimension = _detect_dimension(query, self.df)
        rule_series_dimension = _detect_series_dimension(query, self.df, rule_dimension or intent.dimension)
        rule_chart_type = _detect_chart_type(query)
        rule_request_kind = _detect_request_kind(query)
        rule_top_n = _extract_top_n(query)
        explicit_chart_type = _has_explicit_chart_type(query)

        if rule_request_kind == "graph":
            intent.request_kind = "graph"

        if rule_metric != "ticket_count" and intent.metric == "ticket_count":
            intent.metric = rule_metric

        intent.filters = _merge_filters(intent.filters, rule_filters)

        if intent.lookback_value is None and lookback_value is not None:
            intent.lookback_value = lookback_value
        if intent.lookback_unit is None and lookback_unit is not None:
            intent.lookback_unit = lookback_unit

        if rule_dimension and rule_dimension != "created_at":
            intent.dimension = rule_dimension
        elif intent.request_kind == "graph" and rule_dimension == "created_at" and "created_at" in self.df.columns:
            if intent.dimension is None or intent.dimension == "created_at" or _has_trend_phrase(query):
                intent.dimension = "created_at"
        elif intent.dimension is None:
            intent.dimension = rule_dimension

        if intent.request_kind == "graph" and _has_trend_phrase(query) and "created_at" in self.df.columns:
            if rule_dimension and rule_dimension != "created_at" and intent.series_dimension is None:
                intent.series_dimension = rule_dimension
            intent.dimension = "created_at"

        if intent.request_kind == "graph":
            if explicit_chart_type:
                intent.chart_type = rule_chart_type
            elif rule_chart_type != "bar":
                intent.chart_type = rule_chart_type
            # Top-N category requests generally expect category comparison, not timeline.
            if intent.top_n and not _has_trend_phrase(query):
                intent.chart_type = "bar"
            if intent.dimension == "created_at" and intent.chart_type == "bar" and not explicit_chart_type:
                intent.chart_type = "line"

        if intent.top_n is None and rule_top_n is not None:
            intent.top_n = rule_top_n

        if intent.series_dimension is None and rule_series_dimension is not None:
            intent.series_dimension = rule_series_dimension

        if intent.series_dimension == intent.dimension:
            intent.series_dimension = None

        intent.time_grain = _infer_time_grain(query, intent.lookback_unit, intent.lookback_value)
        return intent

    def _schema_payload(self) -> dict[str, Any]:
        sample_values: dict[str, list[str]] = {}
        for column in self.df.columns:
            if self.df[column].dtype == "object" or str(self.df[column].dtype).startswith("category"):
                sample_values[column] = _value_candidates(self.df[column], max_values=10)
        return {
            "columns": list(self.df.columns),
            "sample_values": sample_values,
        }

    def _plan_with_llm(self, query: str) -> AgenticQueryIntent | None:
        if self.client is None or self.model is None:
            return None

        schema = self._schema_payload()
        prompt = {
            "query": query,
            "schema": schema,
            "instructions": {
                "output_format": "Return strict JSON only.",
                "request_kind": "graph | table | text_recommendations",
                "chart_type": "bar | line | scatter | pie | histogram | heatmap",
                "metric": "ticket_count | avg_mttr_hours | sla_compliance_pct | breach_rate_pct | breached_tickets | open_tickets | team_performance_index",
                "dimension": "existing column name or null",
                "series_dimension": "existing column name for color/stack split or null",
                "time_grain": "D | W | M | Q | Y",
                "filters": "dict of {column:[values]}",
                "lookback": "{value:int|null, unit:day|week|month|quarter|year|null}",
                "top_n": "int|null",
            },
        }

        try:
            response = self.client.responses.create(
                model=self.model,
                temperature=0,
                input=[
                    {
                        "role": "system",
                        "content": "You are an intent planner for ticket analytics queries. Return only JSON.",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(prompt),
                    },
                ],
            )
            parsed = _extract_json_blob(response.output_text)
            if not parsed:
                return None

            request_kind = str(parsed.get("request_kind", "table")).lower()
            chart_type = str(parsed.get("chart_type", "bar")).lower()
            metric = str(parsed.get("metric", "ticket_count"))
            dimension = parsed.get("dimension")
            if dimension is not None:
                dimension = str(dimension)
            time_grain = str(parsed.get("time_grain", "M")).upper()

            raw_filters = parsed.get("filters") if isinstance(parsed.get("filters"), dict) else {}
            filters: dict[str, list[str]] = {}
            for column, values in raw_filters.items():
                if column not in self.df.columns:
                    continue
                if not isinstance(values, list):
                    continue
                filters[column] = [str(v) for v in values if str(v).strip()]

            lookback = parsed.get("lookback") if isinstance(parsed.get("lookback"), dict) else {}
            lookback_value = lookback.get("value")
            lookback_unit = lookback.get("unit")
            if lookback_value is not None:
                lookback_value = int(lookback_value)
            if lookback_unit is not None:
                lookback_unit = str(lookback_unit).lower()

            top_n = parsed.get("top_n")
            if top_n is not None:
                top_n = int(top_n)

            intent = AgenticQueryIntent(
                request_kind=request_kind if request_kind in {"graph", "table", "text_recommendations"} else "table",
                chart_type=chart_type if chart_type in {"bar", "line", "scatter", "pie", "histogram", "heatmap"} else "bar",
                metric=metric if metric in _METRIC_LABELS else "ticket_count",
                dimension=dimension if (dimension in self.df.columns or dimension == "created_at") else None,
                time_grain=time_grain if time_grain in {"D", "W", "M", "Q", "Y"} else "M",
                filters=filters,
                lookback_value=lookback_value,
                lookback_unit=lookback_unit if lookback_unit in _LOOKBACK_UNITS else None,
                top_n=top_n,
                source="llm_intent_agent",
                series_dimension=(
                    str(parsed.get("series_dimension"))
                    if parsed.get("series_dimension") in self.df.columns and parsed.get("series_dimension") != dimension
                    else None
                ),
            )

            if intent.dimension is None:
                intent.dimension = _detect_dimension(query, self.df)
            return self._refine_with_rules(query, intent)
        except Exception:
            return None

    def _plan_with_rules(self, query: str) -> AgenticQueryIntent:
        lookback_value, lookback_unit = _extract_lookback(query)
        request_kind = _detect_request_kind(query)
        chart_type = _detect_chart_type(query)
        metric = _detect_metric(query)
        dimension = _detect_dimension(query, self.df)
        series_dimension = _detect_series_dimension(query, self.df, dimension)
        time_grain = _infer_time_grain(query, lookback_unit, lookback_value)
        filters = _extract_filters(query, self.df)
        top_n = _extract_top_n(query)
        explicit_chart_type = _has_explicit_chart_type(query)

        if request_kind == "graph" and _has_trend_phrase(query) and "created_at" in self.df.columns:
            if dimension and dimension != "created_at" and series_dimension is None:
                series_dimension = dimension
            dimension = "created_at"

        # If query is time-based and no explicit chart type is requested, prefer line.
        if request_kind == "graph" and dimension == "created_at" and chart_type == "bar" and not explicit_chart_type:
            chart_type = "line"

        return AgenticQueryIntent(
            request_kind=request_kind,
            chart_type=chart_type,
            metric=metric,
            dimension=dimension,
            time_grain=time_grain,
            filters=filters,
            lookback_value=lookback_value,
            lookback_unit=lookback_unit,
            top_n=top_n,
            source="rule_intent_agent",
            series_dimension=series_dimension if series_dimension != dimension else None,
        )


class GraphAgent:
    """Graph builder agent: augments data, slices, and constructs plotly outputs."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.original_df = df

    def build(self, intent: AgenticQueryIntent) -> BuildOutput:
        working = self._augment_dataframe(self.original_df)

        missing_requirements = [col for col in _METRIC_REQUIREMENTS[intent.metric] if col not in working.columns]
        if missing_requirements:
            no_data_text = (
                "Requested graph cannot be generated because required data is unavailable: "
                + ", ".join(missing_requirements)
            )
            if intent.request_kind == "graph":
                title = f"{_METRIC_LABELS[intent.metric]} ({intent.chart_type})"
                return self._build_no_data_output(
                    intent=intent,
                    text=no_data_text,
                    title=title,
                    reason="Missing required metric columns",
                    filtered_rows=0,
                    total_rows=int(len(working)),
                    applied_filters=intent.filters,
                    date_window={"start": None, "end": None},
                    missing_requirements=missing_requirements,
                )
            return BuildOutput(
                kind="text",
                text=no_data_text,
                data=None,
                figure=None,
                chart_type=None,
                chart_title=None,
                filtered_rows=0,
                total_rows=int(len(working)),
                applied_filters=intent.filters,
                date_window={"start": None, "end": None},
                missing_requirements=missing_requirements,
                data_unavailable=True,
            )

        filtered, applied_filters = self._apply_filters(working, intent.filters)
        filtered, date_window = self._apply_lookback(filtered, intent)

        if filtered.empty:
            no_data_text = "No data available for the requested graph after applying filters and date range."
            if intent.request_kind == "graph":
                title = f"{_METRIC_LABELS[intent.metric]} ({intent.chart_type})"
                return self._build_no_data_output(
                    intent=intent,
                    text=no_data_text,
                    title=title,
                    reason="No rows matched the current filters/date range",
                    filtered_rows=0,
                    total_rows=int(len(working)),
                    applied_filters=applied_filters,
                    date_window=date_window,
                    missing_requirements=[],
                )
            return BuildOutput(
                kind="text",
                text=no_data_text,
                data=None,
                figure=None,
                chart_type=None,
                chart_title=None,
                filtered_rows=0,
                total_rows=int(len(working)),
                applied_filters=applied_filters,
                date_window=date_window,
                missing_requirements=[],
                data_unavailable=True,
            )

        series_col = self._series_breakdown_column(filtered, intent)
        grouped = self._aggregate(filtered, intent, series_col=series_col)
        if grouped.empty:
            no_data_text = "No data available for the requested graph because the aggregated result is empty."
            if intent.request_kind == "graph":
                title = f"{_METRIC_LABELS[intent.metric]} ({intent.chart_type})"
                return self._build_no_data_output(
                    intent=intent,
                    text=no_data_text,
                    title=title,
                    reason="Grouped result has no usable points",
                    filtered_rows=int(len(filtered)),
                    total_rows=int(len(working)),
                    applied_filters=applied_filters,
                    date_window=date_window,
                    missing_requirements=[],
                )
            return BuildOutput(
                kind="text",
                text=no_data_text,
                data=None,
                figure=None,
                chart_type=None,
                chart_title=None,
                filtered_rows=int(len(filtered)),
                total_rows=int(len(working)),
                applied_filters=applied_filters,
                date_window=date_window,
                missing_requirements=[],
                data_unavailable=True,
            )

        metric_col = self._metric_output_column(intent.metric)
        x_col = "period" if intent.dimension == "created_at" else (intent.dimension or grouped.columns[0])
        color_col = series_col if series_col and series_col in grouped.columns else None

        chart_title = f"{_METRIC_LABELS[intent.metric]} by {x_col}"
        if color_col:
            chart_title = f"{chart_title} and {color_col}"
        figure = None

        if intent.request_kind == "graph":
            figure = self._build_figure(grouped, intent.chart_type, x_col, metric_col, chart_title, color_col=color_col)
            kind = "chart"
            text = (
                f"Generated {intent.chart_type} chart for {_METRIC_LABELS[intent.metric]} with {len(grouped)} data points. "
                f"Rows analyzed: {len(filtered)}."
            )
        else:
            kind = "table"
            text = (
                f"Computed {_METRIC_LABELS[intent.metric]} grouped by {x_col}. "
                f"Rows analyzed: {len(filtered)}."
            )

        return BuildOutput(
            kind=kind,
            text=text,
            data=grouped,
            figure=figure,
            chart_type=intent.chart_type if intent.request_kind == "graph" else None,
            chart_title=chart_title,
            filtered_rows=int(len(filtered)),
            total_rows=int(len(working)),
            applied_filters=applied_filters,
            date_window=date_window,
            missing_requirements=[],
            data_unavailable=False,
        )

    def _build_no_data_output(
        self,
        intent: AgenticQueryIntent,
        text: str,
        title: str,
        reason: str,
        filtered_rows: int,
        total_rows: int,
        applied_filters: dict[str, list[str]],
        date_window: dict[str, str | None],
        missing_requirements: list[str],
    ) -> BuildOutput:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data available<br><sup>{reason}</sup>",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 14},
        )
        fig.update_layout(title=title, xaxis={"visible": False}, yaxis={"visible": False})
        return BuildOutput(
            kind="chart",
            text=text,
            data=pd.DataFrame(),
            figure=fig,
            chart_type=intent.chart_type,
            chart_title=title,
            filtered_rows=filtered_rows,
            total_rows=total_rows,
            applied_filters=applied_filters,
            date_window=date_window,
            missing_requirements=missing_requirements,
            data_unavailable=True,
        )

    def _series_breakdown_column(self, df: pd.DataFrame, intent: AgenticQueryIntent) -> str | None:
        if intent.request_kind != "graph":
            return None

        if intent.series_dimension and intent.series_dimension in df.columns and intent.series_dimension != intent.dimension:
            return intent.series_dimension

        if intent.dimension != "created_at":
            return None

        candidates = [col for col, values in intent.filters.items() if len(values) > 1 and col in df.columns]
        if not candidates:
            return None

        preferred = ["priority", "team", "service", "category_derived", "business_function_derived", "cluster", "status"]
        for column in preferred:
            if column in candidates:
                return column
        return candidates[0]

    def _augment_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()

        for dt_col in ["created_at", "resolved_at", "updated_at"]:
            if dt_col in frame.columns:
                frame[dt_col] = pd.to_datetime(frame[dt_col], errors="coerce")

        if "priority" not in frame.columns and "severity" in frame.columns:
            mapped = frame["severity"].astype(str).str.strip().str.lower().map(PRIORITY_MAP)
            frame["priority"] = mapped.fillna("Unknown")

        if "is_resolved" not in frame.columns:
            status = frame.get("status", pd.Series(["" for _ in range(len(frame))], index=frame.index)).astype(str).str.lower()
            frame["is_resolved"] = status.isin(RESOLVED_STATUSES)
            if "resolved_at" in frame.columns:
                frame["is_resolved"] = frame["is_resolved"] | frame["resolved_at"].notna()

        if "is_open" not in frame.columns:
            frame["is_open"] = ~frame["is_resolved"].fillna(False)

        if "resolution_time_hours" not in frame.columns and "resolution_time" in frame.columns:
            frame["resolution_time_hours"] = frame["resolution_time"].map(parse_duration_to_hours)

        if "mttr_hours" not in frame.columns:
            frame["mttr_hours"] = pd.NA

        if "created_at" in frame.columns and "resolved_at" in frame.columns:
            delta = (frame["resolved_at"] - frame["created_at"]).dt.total_seconds() / 3600
            fill_mask = frame["mttr_hours"].isna() | (pd.to_numeric(frame["mttr_hours"], errors="coerce") <= 0)
            frame.loc[fill_mask, "mttr_hours"] = delta

        if "resolution_time_hours" in frame.columns:
            mttr = pd.to_numeric(frame["mttr_hours"], errors="coerce")
            fallback = (mttr.isna() | (mttr <= 0)) & (pd.to_numeric(frame["resolution_time_hours"], errors="coerce") > 0)
            frame.loc[fallback, "mttr_hours"] = frame.loc[fallback, "resolution_time_hours"]

        frame["mttr_hours"] = pd.to_numeric(frame["mttr_hours"], errors="coerce")

        if "ticket_age_hours" not in frame.columns and "created_at" in frame.columns:
            end_time = frame.get("resolved_at", pd.Series(pd.NaT, index=frame.index)).fillna(pd.Timestamp.now())
            frame["ticket_age_hours"] = (end_time - frame["created_at"]).dt.total_seconds() / 3600

        if "sla_threshold_hours" not in frame.columns:
            priority = frame.get("priority", pd.Series(["Unknown" for _ in range(len(frame))], index=frame.index)).astype(str)
            frame["sla_threshold_hours"] = priority.map(SLA_THRESHOLD_HOURS).fillna(SLA_THRESHOLD_HOURS["Unknown"])

        if "is_sla_breached" not in frame.columns:
            duration = frame["mttr_hours"].fillna(frame.get("ticket_age_hours", pd.Series(0, index=frame.index)))
            frame["is_sla_breached"] = duration > frame["sla_threshold_hours"]

        return frame

    def _apply_filters(self, df: pd.DataFrame, requested: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        filtered = df.copy()
        applied: dict[str, list[str]] = {}

        for column, values in requested.items():
            if column not in filtered.columns:
                continue
            if not values:
                continue

            column_series = filtered[column].astype(str)
            mask = pd.Series(False, index=filtered.index)
            matched_values: list[str] = []
            for raw in values:
                needle = str(raw).strip()
                if not needle:
                    continue
                exact = column_series.str.lower() == needle.lower()
                contains = column_series.str.contains(re.escape(needle), case=False, na=False)
                current = exact | contains
                if current.any():
                    matched_values.append(needle)
                mask = mask | current

            if matched_values:
                filtered = filtered[mask]
                applied[column] = sorted(set(matched_values))
            else:
                filtered = filtered[mask]
                applied[column] = [str(v) for v in values if str(v).strip()]

        return filtered, applied

    def _apply_lookback(self, df: pd.DataFrame, intent: AgenticQueryIntent) -> tuple[pd.DataFrame, dict[str, str | None]]:
        if not intent.lookback_value or not intent.lookback_unit:
            return df, {"start": None, "end": None}
        if "created_at" not in df.columns:
            return df, {"start": None, "end": None}

        dated = df.dropna(subset=["created_at"]).copy()
        if dated.empty:
            return dated, {"start": None, "end": None}

        end = pd.Timestamp(dated["created_at"].max())
        unit = intent.lookback_unit
        value = int(intent.lookback_value)

        if unit in {"day", "days"}:
            start = end - pd.Timedelta(days=value)
        elif unit in {"week", "weeks"}:
            start = end - pd.Timedelta(weeks=value)
        elif unit in {"month", "months"}:
            start = end - pd.DateOffset(months=value)
        elif unit in {"quarter", "quarters"}:
            start = end - pd.DateOffset(months=value * 3)
        elif unit in {"year", "years"}:
            start = end - pd.DateOffset(years=value)
        else:
            return dated, {"start": None, "end": None}

        sliced = dated[dated["created_at"] >= start]
        return sliced, {"start": start.isoformat(), "end": end.isoformat()}

    def _metric_output_column(self, metric: str) -> str:
        return {
            "ticket_count": "ticket_count",
            "avg_mttr_hours": "avg_mttr_hours",
            "sla_compliance_pct": "sla_compliance_pct",
            "breach_rate_pct": "breach_rate_pct",
            "breached_tickets": "breached_tickets",
            "open_tickets": "open_tickets",
            "team_performance_index": "team_performance_index",
        }.get(metric, "ticket_count")

    def _aggregate(self, df: pd.DataFrame, intent: AgenticQueryIntent, series_col: str | None = None) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        work = df.copy()
        if intent.dimension == "created_at":
            if "created_at" not in work.columns:
                return pd.DataFrame()
            work = work.dropna(subset=["created_at"])
            if work.empty:
                return pd.DataFrame()
            work["period"] = work["created_at"].dt.to_period(intent.time_grain).dt.to_timestamp()
            group_col = "period"
        else:
            group_col = intent.dimension
        group_cols: list[str] = [group_col] if group_col is not None else []
        if series_col and series_col in work.columns and series_col not in group_cols:
            group_cols.append(series_col)

        metric = intent.metric
        out_col = self._metric_output_column(metric)

        if group_col is None:
            if metric == "ticket_count":
                result = pd.DataFrame([{out_col: int(len(work))}])
            elif metric == "avg_mttr_hours":
                result = pd.DataFrame([{out_col: float(pd.to_numeric(work["mttr_hours"], errors="coerce").mean())}])
            elif metric == "sla_compliance_pct":
                result = pd.DataFrame([{out_col: float((1 - work["is_sla_breached"].mean()) * 100.0)}])
            elif metric == "breach_rate_pct":
                result = pd.DataFrame([{out_col: float(work["is_sla_breached"].mean() * 100.0)}])
            elif metric == "breached_tickets":
                result = pd.DataFrame([{out_col: int(work["is_sla_breached"].sum())}])
            elif metric == "open_tickets":
                result = pd.DataFrame([{out_col: int(work["is_open"].sum())}])
            elif metric == "team_performance_index":
                result = pd.DataFrame([{out_col: float(work["team_performance_index"].mean())}])
            else:
                result = pd.DataFrame([{out_col: int(len(work))}])
            return result

        if group_col not in work.columns:
            return pd.DataFrame()

        if metric == "ticket_count":
            result = work.groupby(group_cols, dropna=False).size().rename(out_col).reset_index()
        elif metric == "avg_mttr_hours":
            result = (
                work.groupby(group_cols, dropna=False)["mttr_hours"]
                .mean()
                .rename(out_col)
                .reset_index()
            )
        elif metric == "sla_compliance_pct":
            result = (
                work.groupby(group_cols, dropna=False)["is_sla_breached"]
                .mean()
                .rsub(1.0)
                .mul(100.0)
                .rename(out_col)
                .reset_index()
            )
        elif metric == "breach_rate_pct":
            result = (
                work.groupby(group_cols, dropna=False)["is_sla_breached"]
                .mean()
                .mul(100.0)
                .rename(out_col)
                .reset_index()
            )
        elif metric == "breached_tickets":
            result = (
                work.groupby(group_cols, dropna=False)["is_sla_breached"]
                .sum()
                .rename(out_col)
                .reset_index()
            )
        elif metric == "open_tickets":
            result = work.groupby(group_cols, dropna=False)["is_open"].sum().rename(out_col).reset_index()
        elif metric == "team_performance_index":
            result = (
                work.groupby(group_cols, dropna=False)["team_performance_index"]
                .mean()
                .rename(out_col)
                .reset_index()
            )
        else:
            result = work.groupby(group_cols, dropna=False).size().rename(out_col).reset_index()

        if out_col in result.columns:
            result[out_col] = pd.to_numeric(result[out_col], errors="coerce").round(2)
            if metric == "avg_mttr_hours":
                result = result.dropna(subset=[out_col])

        if group_col == "period":
            sort_cols: list[str] = ["period"]
            if series_col and series_col in result.columns:
                sort_cols.append(series_col)
            result = result.sort_values(sort_cols)
        else:
            if intent.top_n:
                if series_col and series_col in result.columns:
                    primary_rank = (
                        result.groupby(group_col, dropna=False)[out_col]
                        .mean()
                        .sort_values(ascending=False)
                        .head(intent.top_n)
                        .reset_index()
                    )
                    top_values = set(primary_rank[group_col].tolist())
                    result = result[result[group_col].isin(top_values)]
                    result = result.merge(
                        primary_rank.rename(columns={out_col: "_rank_value"}),
                        on=group_col,
                        how="left",
                    )
                    sort_cols = ["_rank_value", group_col]
                    if series_col in result.columns:
                        sort_cols.append(series_col)
                    result = result.sort_values(sort_cols, ascending=[False, True, True]).drop(columns=["_rank_value"])
                else:
                    result = result.sort_values(out_col, ascending=False).head(intent.top_n)
            else:
                result = result.sort_values(out_col, ascending=False)

        return result

    def _build_figure(
        self,
        data: pd.DataFrame,
        chart_type: str,
        x_col: str,
        y_col: str,
        title: str,
        color_col: str | None = None,
    ) -> Figure:
        color = color_col if color_col and color_col in data.columns else None
        if chart_type == "line":
            return px.line(data, x=x_col, y=y_col, color=color, markers=True, title=title)
        if chart_type == "scatter":
            return px.scatter(data, x=x_col, y=y_col, color=color, title=title)
        if chart_type == "pie":
            return px.pie(data, names=x_col, values=y_col, title=title)
        if chart_type == "histogram":
            return px.histogram(data, x=x_col, y=y_col if y_col in data.columns else None, color=color, title=title)
        if chart_type == "heatmap":
            return px.density_heatmap(data, x=x_col, y=y_col, title=title)
        fig = px.bar(data, x=x_col, y=y_col, color=color, title=title)
        if color is not None:
            fig.update_layout(barmode="stack")
        return fig


class ValidatorAgent:
    """Validator agent: verifies graph/query output quality and alignment."""

    def __init__(self) -> None:
        self.client, self.model = _openai_client()

    def validate(self, query: str, intent: AgenticQueryIntent, output: BuildOutput) -> dict[str, Any]:
        if output.data_unavailable and intent.request_kind == "graph" and output.kind == "chart" and output.figure is not None:
            return {"is_valid": True, "issues": [], "llm_validator_used": False}

        issues: list[str] = []

        if output.data_unavailable:
            issues.append("Data unavailable for requested graph.")

        if intent.request_kind == "graph":
            if output.kind != "chart":
                issues.append("Expected chart output but chart was not generated.")
            if output.figure is None:
                issues.append("Plotly figure is missing from graph output.")

        if output.filtered_rows == 0 and not output.data_unavailable:
            issues.append("No rows left after filter application.")

        if intent.metric == "avg_mttr_hours" and output.data is not None:
            if "avg_mttr_hours" in output.data.columns and output.data["avg_mttr_hours"].notna().sum() == 0:
                issues.append("MTTR metric has no usable values.")

        llm_report = self._validate_with_llm(query, intent, output)
        if llm_report and isinstance(llm_report.get("issues"), list):
            issues.extend([str(i) for i in llm_report["issues"] if str(i).strip()])

        deduped = []
        seen = set()
        for issue in issues:
            if issue in seen:
                continue
            deduped.append(issue)
            seen.add(issue)

        return {
            "is_valid": len(deduped) == 0,
            "issues": deduped,
            "llm_validator_used": bool(llm_report),
        }

    def _validate_with_llm(self, query: str, intent: AgenticQueryIntent, output: BuildOutput) -> dict[str, Any] | None:
        if self.client is None or self.model is None or output.data is None or output.data.empty:
            return None

        preview = output.data.head(8).to_dict("records")
        payload = {
            "query": query,
            "intent": asdict(intent),
            "output_kind": output.kind,
            "rows": output.filtered_rows,
            "data_preview": preview,
        }

        try:
            response = self.client.responses.create(
                model=self.model,
                temperature=0,
                input=[
                    {
                        "role": "system",
                        "content": "You are a validator agent. Return only JSON: {is_valid:boolean, issues:string[]}",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(payload),
                    },
                ],
            )
            parsed = _extract_json_blob(response.output_text)
            if not parsed:
                return None
            return {
                "is_valid": bool(parsed.get("is_valid", True)),
                "issues": parsed.get("issues", []) if isinstance(parsed.get("issues"), list) else [],
            }
        except Exception:
            return None


def _clarification_prompt(query: str, intent: AgenticQueryIntent) -> str | None:
    q = query.lower()

    if intent.request_kind != "graph":
        return None

    if intent.dimension is None:
        return (
            "I need one clarification before generating the graph: which dimension should I use "
            "(for example team, priority, service, category, or time)?"
        )

    if intent.metric == "ticket_count" and any(
        token in q for token in ["sla", "compliance", "adherence", "mttr", "breach", "performance", "efficiency"]
    ):
        return (
            "I need one clarification before generating the graph: should I plot SLA compliance, SLA breach rate, "
            "MTTR, or ticket count?"
        )

    if (
        not _has_explicit_dimension_phrase(query)
        and not _has_trend_phrase(query)
        and not intent.lookback_value
        and not intent.top_n
    ):
        return (
            "I need one clarification before generating the graph: which dimension should I use "
            "(for example team, priority, service, category, or time)?"
        )

    if intent.dimension == "created_at" and intent.top_n and not _has_trend_phrase(query):
        return (
            "I need one clarification: do you want a trend over time for top segments, or a top-segment comparison "
            "for the selected period?"
        )

    return None


def _repair_intent_if_possible(intent: AgenticQueryIntent, output: BuildOutput, df: pd.DataFrame) -> AgenticQueryIntent | None:
    if output.data_unavailable:
        return None

    repaired = AgenticQueryIntent(**asdict(intent))
    changed = False

    if intent.request_kind == "graph" and intent.dimension is None:
        if "created_at" in df.columns:
            repaired.dimension = "created_at"
            changed = True
        elif "team" in df.columns:
            repaired.dimension = "team"
            changed = True

    if intent.request_kind == "graph" and intent.chart_type not in {"bar", "line", "scatter", "pie", "histogram", "heatmap"}:
        repaired.chart_type = "bar"
        changed = True

    if intent.metric == "avg_mttr_hours" and "mttr_hours" not in df.columns and "resolution_time_hours" in df.columns:
        repaired.metric = "ticket_count"
        changed = True

    if repaired.series_dimension == repaired.dimension:
        repaired.series_dimension = None
        changed = True

    if changed:
        repaired.source = f"{intent.source}+repair"
        return repaired
    return None


def _summary_when_no_query(df: pd.DataFrame) -> QueryResult:
    report = build_insight_report(df)
    metrics = report["metrics"]
    text = (
        f"Dataset has {metrics['total_tickets']} tickets, average MTTR {metrics['avg_mttr_hours']}h, "
        f"and SLA breach rate {round(metrics['breach_rate'] * 100, 2)}%."
    )
    return QueryResult(kind="text", text=text, agent_trace={"source": "summary"})


def _recommendation_query(df: pd.DataFrame) -> QueryResult:
    report = build_insight_report(df)
    recs = report["recommendations"]
    body = "\n".join([f"- {rec}" for rec in recs])
    return QueryResult(kind="text", text=f"Recommended actions:\n{body}", agent_trace={"source": "recommendation"})


def answer_query(query: str, df: pd.DataFrame) -> QueryResult:
    text = query.strip()
    if not text:
        return _summary_when_no_query(df)

    lowered = text.lower()
    if any(k in lowered for k in ["recommend", "improve", "what should", "action", "suggest"]):
        return _recommendation_query(df)

    planner = IntentAgent(df)
    intent = planner.plan(text)
    clarification = _clarification_prompt(text, intent)
    if clarification:
        return QueryResult(
            kind="clarification",
            text=clarification,
            agent_trace={"intent": asdict(intent), "source": "clarification_agent", "llm_status": llm_status()},
        )

    graph_agent = GraphAgent(df)
    output = graph_agent.build(intent)

    validator = ValidatorAgent()
    validation = validator.validate(text, intent, output)

    repaired = False
    if not validation["is_valid"] and not output.data_unavailable:
        repaired_intent = _repair_intent_if_possible(intent, output, df)
        if repaired_intent is not None:
            retry_output = graph_agent.build(repaired_intent)
            retry_validation = validator.validate(text, repaired_intent, retry_output)
            if retry_validation["is_valid"] or (retry_output.data_unavailable and output.data is None):
                output = retry_output
                validation = retry_validation
                intent = repaired_intent
                repaired = True

    suffix = "Validation passed."
    if not validation["is_valid"]:
        suffix = "Validation flagged: " + "; ".join(validation["issues"])

    result_text = f"{output.text} {suffix}".strip()

    chart_payload: dict[str, Any] | None = None
    if output.figure is not None:
        chart_payload = {
            "type": "plotly_figure",
            "figure": output.figure,
            "title": output.chart_title,
            "chart_type": output.chart_type,
        }

    return QueryResult(
        kind=output.kind,
        text=result_text,
        data=output.data,
        chart=chart_payload,
        agent_trace={
            "intent": asdict(intent),
            "validation": validation,
            "llm_status": llm_status(),
            "applied_filters": output.applied_filters,
            "date_window": output.date_window,
            "filtered_rows": output.filtered_rows,
            "total_rows": output.total_rows,
            "missing_requirements": output.missing_requirements,
            "repaired": repaired,
        },
    )
