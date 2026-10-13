"""Shared remediation timing fields propagated from tool_ctx to agent output."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def finalize_remediation_timings(tool_ctx: dict[str, Any], result: dict[str, Any]) -> None:
    """Copy remediation timestamps into agent loop result; set end time on agent exit."""
    if tool_ctx.get("remediation_start_time") and not tool_ctx.get("remediation_end_time"):
        tool_ctx["remediation_end_time"] = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
    for key in (
        "remediation_start_time",
        "remediation_end_time",
        "remediation_executed_time",
        "detection_timestamp",
        "remediation_writes",
    ):
        if tool_ctx.get(key) is not None:
            result[key] = tool_ctx[key]


def remediation_into_signals(llm_result: dict[str, Any], signals: dict[str, Any]) -> None:
    """Merge remediation timing from LLM loop result into DetectionResult.signals."""
    for key in (
        "remediation_start_time",
        "remediation_end_time",
        "remediation_executed_time",
        "detection_timestamp",
        "remediation_writes",
    ):
        if llm_result.get(key) is not None:
            signals[key] = llm_result[key]


_REMEDIATION_SIGNAL_KEYS = (
    "remediation_start_time",
    "remediation_end_time",
    "remediation_executed_time",
    "detection_timestamp",
    "remediation_writes",
)


def parse_agent_signals(agent_result: dict[str, Any] | None) -> dict[str, Any]:
    """Extract signals dict from harness agent subprocess result."""
    if not agent_result:
        return {}
    signals: dict[str, Any] = {}
    raw = agent_result.get("signals")
    if isinstance(raw, dict) and raw:
        signals = dict(raw)
    else:
        out = agent_result.get("agent_output_raw") or ""
        if out:
            try:
                import json

                parsed = json.loads(out)
                sig = parsed.get("signals")
                if isinstance(sig, dict):
                    signals = dict(sig)
            except Exception:
                pass
    for key in _REMEDIATION_SIGNAL_KEYS:
        if agent_result.get(key) is not None and key not in signals:
            signals[key] = agent_result[key]
    return signals
