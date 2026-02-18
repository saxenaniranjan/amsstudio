"""Constants and keyword maps for ticket analytics."""

from __future__ import annotations

from typing import Dict, List

COLUMN_ALIASES: Dict[str, List[str]] = {
    "ticket_id": ["ticket_id", "incident_id", "number", "id", "ticket_number", "incident_number"],
    "description": ["description", "short_description", "summary", "details", "incident_description"],
    "team": ["team", "assignment_group", "resolver_group", "support_group", "group"],
    "status": ["status", "state", "ticket_status", "incident_state"],
    "priority": ["priority", "severity", "impact_priority", "urgency"],
    "service": ["service", "business_service", "application", "system", "service_name"],
    "category": ["category", "incident_category", "type", "issue_type"],
    "created_at": ["created_at", "opened_at", "opened", "created", "open_time", "start_time"],
    "resolved_at": ["resolved_at", "closed_at", "resolved", "closed", "end_time", "resolution_time_stamp"],
    "updated_at": ["updated_at", "last_updated", "modified_at"],
    "resolution_time": ["resolution_time", "time_to_resolve", "ttr", "mttr", "resolution_duration"],
    "reopen_count": ["reopen_count", "reopened", "reopens", "re_open_count"],
    "assignee": ["assignee", "assigned_to", "owner", "agent"],
}

RESOLVED_STATUSES = {
    "resolved",
    "closed",
    "completed",
    "done",
    "fulfilled",
}

OPEN_STATUSES = {
    "new",
    "open",
    "in_progress",
    "pending",
    "assigned",
    "on_hold",
}

PRIORITY_MAP = {
    "1": "P1",
    "p1": "P1",
    "critical": "P1",
    "sev1": "P1",
    "severity1": "P1",
    "2": "P2",
    "p2": "P2",
    "high": "P2",
    "sev2": "P2",
    "severity2": "P2",
    "3": "P3",
    "p3": "P3",
    "medium": "P3",
    "moderate": "P3",
    "sev3": "P3",
    "severity3": "P3",
    "4": "P4",
    "p4": "P4",
    "low": "P4",
    "minor": "P4",
    "sev4": "P4",
    "severity4": "P4",
}

SLA_THRESHOLD_HOURS = {
    "P1": 4,
    "P2": 8,
    "P3": 24,
    "P4": 72,
    "Unknown": 24,
}

CATEGORY_KEYWORDS = {
    "Network": ["network", "vpn", "latency", "bandwidth", "dns", "firewall"],
    "Access": ["access", "password", "login", "mfa", "permission", "unlock"],
    "Hardware": ["laptop", "desktop", "keyboard", "mouse", "printer", "hardware"],
    "Software": ["software", "application", "app", "crash", "bug", "upgrade", "install"],
    "Email": ["email", "outlook", "mailbox", "smtp", "exchange"],
    "Database": ["database", "sql", "oracle", "postgres", "query", "db"],
    "Security": ["security", "malware", "phishing", "vulnerability", "breach"],
    "Service Request": ["request", "onboard", "offboard", "new account", "provision"],
    "Performance": ["slow", "performance", "timeout", "hang", "response time"],
}

BUSINESS_FUNCTION_KEYWORDS = {
    "IT Operations": ["infra", "infrastructure", "ops", "network", "server", "platform"],
    "Engineering": ["engineering", "dev", "code", "release", "deployment", "ci/cd"],
    "Finance": ["invoice", "payment", "expense", "finance", "erp", "sap"],
    "HR": ["hr", "payroll", "benefits", "employee", "onboarding"],
    "Sales": ["crm", "salesforce", "lead", "opportunity", "quote"],
    "Customer Support": ["customer", "support", "cs", "helpdesk", "case"],
    "Security": ["security", "soar", "siem", "soc", "threat"],
    "Operations": ["supply", "logistics", "warehouse", "operations", "fulfillment"],
}

CHART_KEYWORDS = {
    "bar": "bar",
    "column": "bar",
    "line": "line",
    "trend": "line",
    "time series": "line",
    "pie": "pie",
    "donut": "pie",
    "histogram": "histogram",
    "distribution": "histogram",
    "scatter": "scatter",
    "heatmap": "heatmap",
}
