"""Forced RAG tool policy based on CONTEXT_C (round 0+ until each corpus is queried)."""
from __future__ import annotations

import os
from typing import Any

from code.tools.context_engineering import context_tool_names


def required_rag_tools() -> list[str]:
    """RAG tools enabled for this run (empty when CONTEXT_C=0)."""
    return context_tool_names()


def rag_tools_called(tool_calls_log: list[dict[str, Any]]) -> set[str]:
    rag = set(required_rag_tools())
    return {str(tc.get("name")) for tc in tool_calls_log if tc.get("name") in rag}


def pending_rag_tools(tool_calls_log: list[dict[str, Any]]) -> list[str]:
    """Required RAG tools not yet invoked, in CONTEXT_C order."""
    called = rag_tools_called(tool_calls_log)
    return [t for t in required_rag_tools() if t not in called]


def in_rag_phase(tool_calls_log: list[dict[str, Any]]) -> bool:
    return bool(pending_rag_tools(tool_calls_log))


def forced_rag_tool_for_round(
    round_idx: int,
    tool_calls_log: list[dict[str, Any]],
) -> str | None:
    """
    Tool to force on this round while RAG phase is active.

    Round 0 forces the first pending corpus tool; each subsequent corpus is
    forced on the next round after the prior one succeeds (round 1 for C2 of
    CONTEXT_C=1,4, etc.). If the model skips a forced call, the same tool stays
    forced on later rounds until it runs.
    """
    pending = pending_rag_tools(tool_calls_log)
    if not pending:
        return None
    required = required_rag_tools()
    next_index = len(required) - len(pending)
    if round_idx >= next_index:
        return pending[0]
    return None


def default_rag_query(tool_name: str) -> str:
    """Default search query when the harness executes a forced RAG call."""
    variant = (
        os.environ.get("HARNESS_FAULT_VARIANT")
        or os.environ.get("FAULT_VARIANT")
        or "cart"
    ).strip()
    fault = (os.environ.get("HARNESS_FAULT_FLAG") or os.environ.get("FAULT_FLAG") or "").strip()
    parts = [variant, "service"]
    if "scale" in fault:
        parts.extend(["deployment", "replicas", "kubernetes"])
    elif "config" in fault:
        parts.extend(["configuration", "environment", "deployment"])
    elif "kill" in fault or "pod" in fault:
        parts.extend(["kubernetes", "pod", "deployment"])
    if tool_name == "rag_search_source":
        parts.append("implementation")
    elif tool_name == "rag_search_docs":
        parts.append("documentation")
    elif tool_name == "rag_search_architecture":
        parts.extend(["architecture", "topology"])
    elif tool_name == "rag_search_dependencies":
        parts.extend(["dependencies", "manifest"])
    return " ".join(parts)


def default_rag_arguments(tool_name: str) -> dict[str, Any]:
    return {"query": default_rag_query(tool_name), "limit": 5}


def openai_tool_choice_for_round(
    round_idx: int,
    tool_calls_log: list[dict[str, Any]],
) -> str | dict[str, Any]:
    forced = forced_rag_tool_for_round(round_idx, tool_calls_log)
    if forced:
        return {"type": "function", "function": {"name": forced}}
    return "auto"
