"""
Tool profiles for scenario-based agent comparisons.

Each profile lists allowed tool names. Agents read TOOL_PROFILE (or SCENARIO_ROLE
for multi-agent scenarios) and only receive matching OpenAI tool definitions.
"""
from __future__ import annotations

import os
import re
from typing import Any

from code.tools.agent_tools import TOOL_DEFINITIONS as _ALL_TOOL_DEFINITIONS
from code.tools import agent_tools

# Profile name -> allowed tool names
TOOL_PROFILES: dict[str, list[str]] = {
    # Scenario A: unified telemetry (no K8s API)
    "telemetry_multimodal": [
        "search_logs",
        "search_traces",
        "search_metrics",
        "query_clickhouse",
        "run_remediation",
    ],
    # Scenario B: signal specialists
    "logs_only": ["search_logs", "query_clickhouse", "run_remediation"],
    "traces_only": ["search_traces", "run_remediation"],
    "metrics_only": ["search_metrics", "run_remediation"],
    # Scenario C: domain specialists
    "hardware_metrics": ["search_metrics", "query_clickhouse", "run_remediation"],
    "platform_k8s": [
        "get_pod_status",
        "get_pod_logs",
        "get_events",
        "restart_deployment",
        "scale_deployment",
        "delete_pod",
        "run_remediation",
    ],
    "application_telemetry": ["search_logs", "search_traces", "run_remediation"],
    # Backward compatibility: all tools (pre-refactor agents)
    "full": [
        "query_clickhouse",
        "search_logs",
        "search_traces",
        "search_metrics",
        "run_remediation",
        "get_pod_status",
        "get_pod_logs",
        "get_events",
        "restart_deployment",
        "scale_deployment",
        "delete_pod",
    ],
}

PROFILE_PROMPTS: dict[str, str] = {
    "telemetry_multimodal": (
        "You analyze logs, traces, and metrics from ClickHouse only. "
        "Do NOT assume Kubernetes pod status — use telemetry tools. "
        "Identify the failing microservice from observability signals."
    ),
    "logs_only": (
        "You are a logs-only specialist. Use search_logs and query_clickhouse. "
        "Report whether logs indicate a failure and which service is affected."
    ),
    "traces_only": (
        "You are a traces-only specialist. Use search_traces to find latency errors, "
        "failed spans, or slow services since the fault window."
    ),
    "metrics_only": (
        "You are a metrics-only specialist. Use search_metrics for SLI anomalies "
        "(latency, errors, saturation) since the fault window."
    ),
    "hardware_metrics": (
        "You are a hardware/host metrics specialist. Use search_metrics and query_clickhouse "
        "for node-level signals: CPU, memory, disk, network saturation. "
        "Do NOT use Kubernetes API tools."
    ),
    "platform_k8s": (
        "You are a Kubernetes platform specialist. Use get_pod_status, get_events, get_pod_logs, "
        "and remediation tools (restart_deployment, scale_deployment, delete_pod). "
        "Do NOT query ClickHouse logs/traces/metrics."
    ),
    "application_telemetry": (
        "You are an application-layer specialist. Use search_logs and search_traces only. "
        "Focus on microservice errors and request failures. Do NOT use Kubernetes API tools."
    ),
    "full": "You have full telemetry and Kubernetes tools.",
}


def active_profile_name() -> str:
    """TOOL_PROFILE env, else SCENARIO_ROLE mapped via caller, else 'full'."""
    explicit = (os.environ.get("TOOL_PROFILE") or "").strip()
    if explicit:
        return explicit
    role = (os.environ.get("SCENARIO_ROLE") or "").strip()
    role_map = {
        "logs": "logs_only",
        "traces": "traces_only",
        "metrics": "metrics_only",
        "hardware": "hardware_metrics",
        "platform": "platform_k8s",
        "application": "application_telemetry",
    }
    if role in role_map:
        return role_map[role]
    return "full"


def allowed_tool_names(profile: str | None = None) -> list[str]:
    name = profile or active_profile_name()
    if name not in TOOL_PROFILES:
        raise ValueError(f"Unknown tool profile: {name}. Known: {list(TOOL_PROFILES)}")
    return list(TOOL_PROFILES[name])


def resolve_tool_definitions(profile: str | None = None) -> list[dict[str, Any]]:
    allowed = set(allowed_tool_names(profile))
    return [
        t
        for t in _ALL_TOOL_DEFINITIONS
        if t.get("function", {}).get("name") in allowed
    ]


def system_prompt_for_profile(profile: str | None = None, base_prompt: str = "") -> str:
    name = profile or active_profile_name()
    hint = PROFILE_PROMPTS.get(name, "")
    role = (os.environ.get("SCENARIO_ROLE") or "").strip()
    role_line = f"\nYour role in this run: **{role}**.\n" if role else ""
    tools_line = f"\nAllowed tools: {', '.join(allowed_tool_names(name))}.\n"
    return f"{base_prompt.strip()}\n{role_line}{tools_line}{hint}\n".strip()


def execute_tool_gated(name: str, arguments: dict[str, Any], ctx: dict[str, Any]) -> str:
    """Execute tool only if permitted by active profile."""
    profile = ctx.get("tool_profile") or active_profile_name()
    allowed = set(allowed_tool_names(profile))
    if name not in allowed:
        return (
            f"Error: tool '{name}' is not available in profile '{profile}'. "
            f"Allowed: {sorted(allowed)}"
        )
    return agent_tools.execute_tool(name, arguments, ctx)
