"""Parse tool-call intents embedded in assistant text (models that skip native tool_calls)."""
from __future__ import annotations

import json
import re
from typing import Any


_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _normalize_call(obj: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("function_name")
    if not name and isinstance(obj.get("function"), dict):
        name = obj["function"].get("name")
    if not name or not isinstance(name, str):
        return None
    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters")
    if args is None and isinstance(obj.get("function"), dict):
        args = obj["function"].get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"raw": args}
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    return {"name": name, "arguments": args}


def parse_content_tool_calls(content: str | None) -> list[dict[str, Any]]:
    """
    Extract tool calls from assistant content when the API left tool_calls empty.

    Supports:
    - <tool_call>{"name":..., "arguments":...}</tool_call>
    - JSON object with name/function_name + arguments/parameters
    - fenced ```json blocks with the same shapes
    """
    if not content or not str(content).strip():
        return []
    text = str(content).strip()
    out: list[dict[str, Any]] = []

    for m in _TOOL_CALL_TAG.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        norm = _normalize_call(obj)
        if norm:
            out.append(norm)

    if out:
        return out

    candidates: list[str] = [text]
    for m in _FENCE.finditer(text):
        candidates.append(m.group(1))

    for cand in candidates:
        # Try whole string, then first {...} slice
        blobs = [cand]
        start = cand.find("{")
        end = cand.rfind("}")
        if start >= 0 and end > start:
            blobs.append(cand[start : end + 1])
        for blob in blobs:
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, list):
                for item in obj:
                    norm = _normalize_call(item) if isinstance(item, dict) else None
                    if norm:
                        out.append(norm)
            else:
                norm = _normalize_call(obj)
                if norm:
                    out.append(norm)
            if out:
                return out
    return out
