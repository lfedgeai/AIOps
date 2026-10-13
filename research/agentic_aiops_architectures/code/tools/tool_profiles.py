"""
Tool profiles for scenario-based agent comparisons.

Each profile lists allowed tool names. Agents read TOOL_PROFILE (or SCENARIO_ROLE
for multi-agent scenarios) and only receive matching OpenAI tool definitions.
"""
from __future__ import annotations

import os
from typing import Any

from code.tools.agent_tools import TOOL_DEFINITIONS as _ALL_TOOL_DEFINITIONS
from code.tools import agent_tools
from code.tools.context_engineering import (
    STANDARD_TELEMETRY_TOOLS,
    context_tool_names,
    profile_has_telemetry,
)
from code.tools.rag_policy import in_rag_phase, required_rag_tools

# Profile name -> allowed tool names
TOOL_PROFILES: dict[str, list[str]] = {
    "telemetry_only": [
        "search_logs",
        "search_traces",
        "search_metrics",
        "query_clickhouse",
        "log_action",
    ],
    "telemetry_k8s": [
        "search_logs",
        "search_traces",
        "search_metrics",
        "query_clickhouse",
        "get_pod_status",
        "get_pod_logs",
        "get_events",
        "restart_deployment",
        "scale_deployment",
        "delete_pod",
        "log_action",
    ],
    "logs_only": ["search_logs", "query_clickhouse", "log_action"],
    "traces_only": ["search_traces", "log_action"],
    "metrics_only": ["search_metrics", "log_action"],
    "hardware_metrics": ["search_metrics", "query_clickhouse", "log_action"],
    "platform_k8s": [
        "get_pod_status",
        "get_pod_logs",
        "get_events",
        "restart_deployment",
        "scale_deployment",
        "delete_pod",
        "log_action",
    ],
    "application_telemetry": ["search_logs", "search_traces", "log_action"],
    "full": [
        "query_clickhouse",
        "search_logs",
        "search_traces",
        "search_metrics",
        "log_action",
        "get_pod_status",
        "get_pod_logs",
        "get_events",
        "restart_deployment",
        "scale_deployment",
        "delete_pod",
    ],
}

PROFILE_PROMPTS: dict[str, str] = {
    "telemetry_only": (
        "You analyze logs, traces, and metrics from ClickHouse only. "
        "You do NOT have Kubernetes API access. "
        "Use compare_telemetry to see what changed after fault injection at T0."
    ),
    "telemetry_k8s": (
        "You have full telemetry AND Kubernetes cluster access. "
        "Use compare_telemetry, search_logs, get_pod_status, and get_events to diagnose. "
        "When you find a fault, FIX it using restart_deployment, scale_deployment, or delete_pod. "
        "You MUST call K8s remediation tools when a fault is found — do not just suggest."
    ),
    "logs_only": (
        "You are a logs-only specialist. Use search_logs, query_clickhouse, and compare_telemetry."
    ),
    "traces_only": "You are a traces-only specialist. Use search_traces.",
    "metrics_only": "You are a metrics-only specialist. Use search_metrics.",
    "hardware_metrics": (
        "You are a hardware/host metrics specialist. Use search_metrics and query_clickhouse."
    ),
    "platform_k8s": (
        "You are a Kubernetes platform specialist. Use get_pod_status, get_events, get_pod_logs, "
        "and remediation tools. Do NOT query ClickHouse logs/traces/metrics."
    ),
    "application_telemetry": (
        "You are an application-layer specialist. Use search_logs and search_traces."
    ),
    "full": "You have full telemetry and Kubernetes tools.",
}

REMEDIATION_VERIFY_RULES = """
## Verify remediation worked

After calling restart_deployment, scale_deployment, or delete_pod:
- Re-check the affected service with your available read tools.
- Do NOT finish with detected=true or claim success until verification shows recovery.
- Use log_action to record what you verified and the outcome.
"""


def active_profile_name() -> str:
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
    base = list(TOOL_PROFILES[name])
    for t in context_tool_names():
        if t not in base:
            base.append(t)
    if profile_has_telemetry(name):
        for t in STANDARD_TELEMETRY_TOOLS:
            if t not in base:
                base.append(t)
    return base


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
    ctx_c = (os.environ.get("CONTEXT_C") or "0").strip()
    if ctx_c != "0":
        rag_tools = ", ".join(required_rag_tools())
        ctx_line = (
            f"\nContext corpora enabled (CONTEXT_C): {ctx_c}.\n"
            f"Round 0 MUST start with a call to the enabled RAG tool(s): {rag_tools}.\n"
            "Other tools unlock after each required RAG tool has been called.\n"
        )
    else:
        ctx_line = ""
    tools_line = f"\nAllowed tools: {', '.join(allowed_tool_names(name))}.\n"
    verify_block = ""
    remediation_tools = {"restart_deployment", "scale_deployment", "delete_pod"}
    if remediation_tools.intersection(allowed_tool_names(name)):
        verify_block = f"\n{REMEDIATION_VERIFY_RULES.strip()}\n"
    return f"{base_prompt.strip()}\n{role_line}{ctx_line}{tools_line}{hint}{verify_block}\n".strip()


def execute_tool_gated(name: str, arguments: dict[str, Any], ctx: dict[str, Any]) -> str:
    profile = ctx.get("tool_profile") or active_profile_name()
    tool_log = ctx.get("tool_calls_log") or []
    if in_rag_phase(tool_log):
        pending = required_rag_tools()
        called = {tc.get("name") for tc in tool_log}
        still = [t for t in pending if t not in called]
        if still and name not in still:
            return (
                f"Error: call `{still[0]}` first (CONTEXT_C RAG phase). "
                f"Pending RAG tools: {still}"
            )
    allowed = set(allowed_tool_names(profile))
    if name not in allowed:
        return (
            f"Error: tool '{name}' is not available in profile '{profile}'. "
            f"Allowed: {sorted(allowed)}"
        )
    return agent_tools.execute_tool(name, arguments, ctx)
