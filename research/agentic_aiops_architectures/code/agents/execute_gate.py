"""Shared diagnose→execute gate and remediation list normalization.

When remediation write tools are available and the model claims detected=true,
forbid finishing with final JSON until at least one write tool was called
(or the model explicitly declares it cannot remediate).

Identical for every matrix agent (fairness). Disable with AGENT_EXECUTE_GATE=0.
"""
from __future__ import annotations

import os
from typing import Any

from code.tools.tool_profiles import allowed_tool_names

REMEDIATION_WRITE_TOOLS = frozenset(
    {
        "restart_deployment",
        "scale_deployment",
        "delete_pod",
        "patch_deployment",
    }
)

_CANNOT_REMEDIATE_MARKERS = (
    "cannot remediate",
    "cannot_remediate",
    "unable to remediate",
    "no remediation tool",
    "remediation tools unavailable",
    "cannot fix",
)

EXECUTE_GATE_NUDGE = (
    "[execute-gate]\n"
    "You reported detected=true but have not called a Kubernetes remediation tool yet.\n"
    "You MUST call one of: scale_deployment, restart_deployment, delete_pod "
    "(choose the action that matches the fault — e.g. replicas=0 → scale_deployment).\n"
    "Do NOT output final JSON until after a remediation tool call "
    "(or state clearly in suggested_remediations that you cannot remediate)."
)


def execute_gate_enabled() -> bool:
    raw = (os.environ.get("AGENT_EXECUTE_GATE") or "1").strip().lower()
    return raw not in ("0", "off", "false", "no", "none")


def flatten_suggested_remediations(value: Any) -> list[str]:
    """Flatten nested lists/tuples into a list of strings (harness-safe)."""
    out: list[str] = []

    def walk(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, str):
            out.append(node)
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
            return
        out.append(str(node))

    walk(value)
    return out


def remediation_tools_available(profile: str | None = None) -> bool:
    try:
        names = set(allowed_tool_names(profile))
    except Exception:
        return False
    return bool(names & REMEDIATION_WRITE_TOOLS)


def has_remediation_write(tool_ctx: dict[str, Any]) -> bool:
    for tc in tool_ctx.get("tool_calls_log") or []:
        if (tc.get("name") or "") in REMEDIATION_WRITE_TOOLS:
            return True
    for w in tool_ctx.get("remediation_writes") or []:
        if isinstance(w, dict) and (w.get("tool") or "") in REMEDIATION_WRITE_TOOLS:
            return True
    return False


def declares_cannot_remediate(parsed: dict[str, Any]) -> bool:
    parts = flatten_suggested_remediations(parsed.get("suggested_remediations"))
    text = " ".join(parts).lower()
    return any(m in text for m in _CANNOT_REMEDIATE_MARKERS)


def maybe_block_final_for_execute_gate(
    parsed: dict[str, Any],
    tool_ctx: dict[str, Any],
    messages: list[dict[str, Any]],
) -> bool:
    """
    If True, a user nudge was appended — caller must continue the tool loop
    (do not finalize).
    """
    if not execute_gate_enabled():
        return False
    if not isinstance(parsed, dict):
        return False
    if not parsed.get("detected"):
        return False
    profile = tool_ctx.get("tool_profile")
    if not remediation_tools_available(profile if isinstance(profile, str) else None):
        return False
    if has_remediation_write(tool_ctx):
        return False
    if declares_cannot_remediate(parsed):
        return False
    # Cap nudges so a stubborn model cannot burn the whole iteration budget
    nudges = int(tool_ctx.get("execute_gate_nudges") or 0)
    max_nudges = int(os.environ.get("AGENT_EXECUTE_GATE_MAX_NUDGES") or "3")
    if nudges >= max_nudges:
        return False
    tool_ctx["execute_gate_nudges"] = nudges + 1
    messages.append({"role": "user", "content": EXECUTE_GATE_NUDGE})
    print(
        f"[execute_gate] blocked final JSON (detected=true, no rem write yet) "
        f"nudge={nudges + 1}/{max_nudges}",
        flush=True,
    )
    return True
