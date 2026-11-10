"""Shared forced-RAG integration for agent tool loops."""
from __future__ import annotations

import json
import sys
import uuid
from typing import Any, Callable

from code.tools.rag_policy import (
    default_rag_arguments,
    forced_rag_tool_for_round,
    in_rag_phase,
    pending_rag_tools,
)
from code.tools.tool_profiles import active_profile_name, resolve_tool_definitions


ExecuteToolFn = Callable[[str, dict[str, Any], dict[str, Any]], str]


def resolve_tools_for_round(
    round_idx: int,
    tool_calls_log: list[dict[str, Any]],
    profile: str | None = None,
) -> list[dict[str, Any]]:
    """During RAG phase expose only the forced tool; otherwise full profile."""
    forced = forced_rag_tool_for_round(round_idx, tool_calls_log)
    all_defs = resolve_tool_definitions(profile or active_profile_name())
    if forced:
        return [t for t in all_defs if t.get("function", {}).get("name") == forced]
    return all_defs


def rag_phase_user_nudge(
    messages: list[dict[str, Any]],
    round_idx: int,
    tool_calls_log: list[dict[str, Any]],
) -> None:
    """Prepend a user nudge before a forced-RAG LLM round (idempotent per round)."""
    forced = forced_rag_tool_for_round(round_idx, tool_calls_log)
    if not forced:
        return
    marker = f"[forced-rag-round-{round_idx}:{forced}]"
    if messages and messages[-1].get("role") == "user" and marker in str(messages[-1].get("content", "")):
        return
    args = default_rag_arguments(forced)
    messages.append({
        "role": "user",
        "content": (
            f"{marker}\n"
            f"You MUST call `{forced}` first with query "
            f'"{args["query"]}" before any telemetry or Kubernetes tools.'
        ),
    })


def execute_forced_rag(
    tool_ctx: dict[str, Any],
    messages: list[dict[str, Any]],
    execute_fn: ExecuteToolFn,
    *,
    truncate_fn: Callable[[str], str] | None = None,
    mlflow_preview_fn: Callable[[str], str] | None = None,
) -> str | None:
    """
    Run the next pending RAG tool from the harness when the model did not.
    Returns the forced tool name if executed, else None.
    """
    pending = pending_rag_tools(tool_ctx.get("tool_calls_log", []))
    if not pending:
        return None
    forced = pending[0]
    args = default_rag_arguments(forced)
    # IMPORTANT: file=sys.stderr — a plain print() would corrupt the agent's
    # JSON stdout protocol (harness parses stdout for the final result JSON).
    print(f"[rag_force] harness executing forced {forced}", file=sys.stderr, flush=True)
    raw = execute_fn(forced, args, tool_ctx)
    preview = (mlflow_preview_fn or (lambda x: x))(raw)
    if truncate_fn:
        raw = truncate_fn(raw)
    tool_ctx.setdefault("tool_calls_log", []).append({
        "name": forced,
        "arguments": args,
        "result_preview": preview,
        "forced_by_harness": True,
    })
    call_id = f"forced-{uuid.uuid4().hex[:12]}"
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": forced, "arguments": json.dumps(args)},
        }],
    })
    messages.append({"role": "tool", "tool_call_id": call_id, "content": raw})
    return forced


def filter_prompt_tool_calls(
    parsed: list[dict[str, Any]],
    round_idx: int,
    tool_calls_log: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """During RAG phase, keep only the forced RAG tool call if present."""
    forced = forced_rag_tool_for_round(round_idx, tool_calls_log)
    if not forced:
        return parsed
    allowed = [tc for tc in parsed if tc.get("name") == forced]
    return allowed


def block_final_json_during_rag_phase(tool_calls_log: list[dict[str, Any]]) -> bool:
    return in_rag_phase(tool_calls_log)
