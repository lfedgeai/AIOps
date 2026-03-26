#!/usr/bin/env python3
"""
DeepSeek R1 Agent: AIOps failure detection using prompt-based tool calling.

Uses deepseek-r1-distill-qwen-14b via LiteLLM.  This reasoning model does NOT
support native function/tool calling, so tools are described in the system
prompt and the model emits tool calls as ``<tool_call>`` blocks or JSON code
fences which we parse, execute, and inject as ``<tool_result>``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.agents.mlflow_agent_logging import (
    log_agent_trace_error_to_mlflow,
    log_agent_trace_to_mlflow,
    serialize_messages_for_mlflow,
)
from code.agents.mlflow_config import mlflow_tracking_uri
from code.tools.agent_tools import TOOL_DEFINITIONS, execute_tool

import requests

CLICKHOUSE_HTTP = os.environ.get("CLICKHOUSE_HTTP", "http://127.0.0.1:8123")

DEEPSEEK_API_BASE = os.environ.get(
    "DEEPSEEK_API_BASE",
    "https://litellm-prod.apps.maas.redhatworkshops.io/v1",
)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-r1-distill-qwen-14b")
MAX_TOOL_ITERATIONS = int(os.environ.get("MAX_TOOL_ITERATIONS", "8"))


@dataclass
class DetectionResult:
    detected: bool
    first_alert_time: str | None
    confidence: float
    suggested_root_cause: str | None
    suggested_remediations: list[str]
    signals: dict[str, Any]
    llm_invoked: bool = False


def _build_tool_descriptions() -> str:
    """Render TOOL_DEFINITIONS into a text block the LLM can read."""
    parts: list[str] = []
    for td in TOOL_DEFINITIONS:
        fn = td.get("function", {})
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        param_lines = []
        for pname, pspec in params.items():
            req = " (required)" if pname in required else ""
            param_lines.append(f"    - {pname}: {pspec.get('type', 'any')} — {pspec.get('description', '')}{req}")
        parts.append(f"### {name}\n{desc}\nParameters:\n" + "\n".join(param_lines))
    return "\n\n".join(parts)


SYSTEM_PROMPT = f"""You are an AIOps expert analyzing observability data from a microservices application (OTEL demo on OpenShift).

## CRITICAL: You MUST call tools before answering

You have NO prior knowledge of the system state. You MUST use tools to gather real data BEFORE making any detection decision. Any answer without tool calls first is WRONG.

## How to call tools

Emit a <tool_call> block with a JSON object containing "name" and "arguments":
<tool_call>{{"name": "search_logs", "arguments": {{"query": "error", "since_ts": "2026-03-24T00:00:00Z"}}}}</tool_call>

You may call multiple tools in one message — use one <tool_call> block per tool.
The system will reply with <tool_result> blocks containing tool output.

{_build_tool_descriptions()}

## Workflow (follow exactly)

Step 1: Call search_logs with query="error" and the since_ts provided.
Step 2: Call search_traces and search_metrics to get more context.
Step 3: If needed, use query_clickhouse for custom SQL.
Step 4: Analyze ALL gathered data.
Step 5: If failure detected, call run_remediation with steps.
Step 6: ONLY AFTER using tools, output your final JSON (no markdown fences):
   {{"detected": true|false, "confidence": 0.0-1.0, "suggested_root_cause": "service name or null", "suggested_remediations": ["step1", "step2"]}}

## Rules
- You MUST call at least search_logs before giving a final answer
- detected=true only when actual data shows errors or latency anomalies
- detected=false when data is sparse or normal
- Do NOT wrap final JSON in code fences
- Do NOT answer without calling tools first
"""

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
_CODE_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(\{[^`]*?\})\s*```",
    re.DOTALL,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks that DeepSeek R1 models emit."""
    return _THINK_RE.sub("", text).strip()


def _chat_completions(messages: list[dict]) -> dict:
    """Call hosted LLM chat/completions API (no tools parameter — prompt-based)."""
    base = (DEEPSEEK_API_BASE or "").rstrip("/")
    url = f"{base}/chat/completions" if base else None
    if not url:
        return {}

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 4096,
    }
    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract tool calls from <tool_call> tags OR markdown code blocks with {"name":..., "arguments":...}."""
    calls: list[dict[str, Any]] = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            name = obj.get("name", "")
            args = obj.get("arguments", {})
            if name:
                calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
        except json.JSONDecodeError:
            pass
    if calls:
        return calls
    for m in _CODE_BLOCK_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            if "name" in obj and "arguments" in obj:
                name = obj["name"]
                args = obj["arguments"]
                if name:
                    calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
        except json.JSONDecodeError:
            pass
    return calls


def _try_parse_final_json(text: str) -> dict[str, Any] | None:
    """Try to extract the final detection JSON from model output."""
    clean = _strip_think(text)
    clean = _TOOL_CALL_RE.sub("", clean).strip()
    # Also check inside code blocks for detection JSON
    for m in _CODE_BLOCK_RE.finditer(clean):
        try:
            obj = json.loads(m.group(1))
            if "detected" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    # Strip code blocks that had tool calls (not detection JSON) and search remaining text
    clean = _CODE_BLOCK_RE.sub("", clean).strip()
    if not clean:
        return None
    json_match = re.search(r'\{[^{}]*"detected"\s*:', clean, re.DOTALL)
    if not json_match:
        return None
    candidate = clean[json_match.start():]
    depth = 0
    end = 0
    for i, ch in enumerate(candidate):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == 0:
        return None
    try:
        return json.loads(candidate[:end])
    except json.JSONDecodeError:
        return None


def run_agentic_loop(
    ch_http: str,
    since_ts: str,
    mlflow_run_id: str | None = None,
) -> dict[str, Any]:
    tool_ctx: dict[str, Any] = {
        "ch_http": ch_http,
        "since_ts": since_ts,
        "mlflow_run_id": mlflow_run_id or os.environ.get("MLFLOW_RUN_ID"),
        "remediation_steps": [],
        "tool_calls_log": [],
        "thinking_log": [],
        "llm_rounds": [],
    }

    user_msg = (
        f"Analyze telemetry since {since_ts}. "
        "Use the tools (search_logs, search_traces, search_metrics) to gather data first, "
        "then determine if there is a failure and suggest remediations. "
        "Start by searching logs for errors."
    )

    def _mlflow_preview(text: str, max_chars: int = 12_000) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 60] + "\n... [truncated for MLflow]"

    def _log_to_mlflow(
        prompts: dict | None = None,
        tool_calls: list | None = None,
        thinking: list | None = None,
    ) -> None:
        run_id = tool_ctx.get("mlflow_run_id") or os.environ.get("MLFLOW_RUN_ID")
        if not run_id:
            return
        uri = mlflow_tracking_uri()
        prompts_payload: dict[str, Any] = {
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_msg,
            "since_ts": since_ts,
            "openai_model": DEEPSEEK_MODEL,
            "openai_api_base": (DEEPSEEK_API_BASE or "").rstrip("/"),
            "registered_tool_names": [t.get("function", {}).get("name") for t in TOOL_DEFINITIONS],
            "conversation_messages": serialize_messages_for_mlflow(messages),
        }
        if prompts:
            prompts_payload.update(prompts)
        tc_log = tool_calls if tool_calls is not None else tool_ctx.get("tool_calls_log", [])
        th_log = thinking if thinking is not None else tool_ctx.get("thinking_log", [])
        try:
            log_agent_trace_to_mlflow(
                run_id=run_id,
                tracking_uri=uri,
                prompts=prompts_payload,
                tool_calls=list(tc_log),
                thinking=list(th_log) if th_log else None,
                llm_rounds=list(tool_ctx.get("llm_rounds", [])),
            )
        except Exception as e:
            print(f"[agent] MLflow logging failed: {type(e).__name__}: {e}", file=sys.stderr)
            try:
                log_agent_trace_error_to_mlflow(
                    run_id=run_id,
                    tracking_uri=uri,
                    error_payload={
                        "phase": "mlflow_logging_error",
                        "error": str(e),
                        "type": type(e).__name__,
                        "traceback": traceback.format_exc(),
                    },
                )
            except Exception as inner:
                print(
                    f"[agent] MLflow error-path log also failed: {type(inner).__name__}: {inner}",
                    file=sys.stderr,
                )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    _log_to_mlflow()

    for round_idx in range(MAX_TOOL_ITERATIONS):
        req_messages_snapshot = serialize_messages_for_mlflow(messages)
        try:
            resp = _chat_completions(messages)
        except requests.RequestException as e:
            err_msg = str(e)
            if "401" in err_msg:
                err_msg = "API key invalid or expired. Set DEEPSEEK_API_KEY."
            tool_ctx["llm_rounds"].append({
                "round": round_idx + 1,
                "model": DEEPSEEK_MODEL,
                "request_messages": req_messages_snapshot,
                "api_error": err_msg,
            })
            _log_to_mlflow(
                tool_calls=tool_ctx.get("tool_calls_log"),
                thinking=tool_ctx.get("thinking_log"),
            )
            return {
                "detected": False,
                "confidence": 0.0,
                "suggested_root_cause": None,
                "suggested_remediations": tool_ctx.get("remediation_steps", []),
                "llm_invoked": True,
                "llm_error": err_msg,
            }

        choice = resp.get("choices", [{}])[0] or {}
        msg = choice.get("message", {})
        raw_content = msg.get("content") or ""

        visible_content = _strip_think(raw_content)
        if visible_content:
            tool_ctx["thinking_log"].append({
                "step": len(tool_ctx["thinking_log"]) + 1,
                "content": visible_content[:100_000],
            })

        messages.append({"role": "assistant", "content": raw_content})

        parsed_tool_calls = _parse_tool_calls(visible_content)

        has_used_tools = len(tool_ctx.get("tool_calls_log", [])) > 0

        if not parsed_tool_calls:
            parsed_json = _try_parse_final_json(raw_content)
            if parsed_json and "detected" in parsed_json and has_used_tools:
                remediations = tool_ctx.get("remediation_steps") or parsed_json.get("suggested_remediations", [])
                tool_ctx["llm_rounds"].append({
                    "round": round_idx + 1,
                    "model": DEEPSEEK_MODEL,
                    "request_messages": req_messages_snapshot,
                    "assistant_content": visible_content[:100_000],
                    "note": "final_detection_json",
                })
                _log_to_mlflow(
                    tool_calls=tool_ctx.get("tool_calls_log"),
                    thinking=tool_ctx.get("thinking_log"),
                )
                return {
                    "detected": bool(parsed_json.get("detected", False)),
                    "confidence": float(parsed_json.get("confidence", 0.0)),
                    "suggested_root_cause": parsed_json.get("suggested_root_cause"),
                    "suggested_remediations": remediations if remediations else list(parsed_json.get("suggested_remediations", [])),
                    "llm_invoked": True,
                }

            tool_ctx["llm_rounds"].append({
                "round": round_idx + 1,
                "model": DEEPSEEK_MODEL,
                "request_messages": req_messages_snapshot,
                "assistant_content": visible_content[:100_000],
                "note": "no_tool_calls" + ("_premature_json_ignored" if (parsed_json and not has_used_tools) else ""),
            })

            if round_idx < MAX_TOOL_ITERATIONS - 1:
                if has_used_tools:
                    nudge = (
                        "You have gathered data with the tools. Now provide your FINAL conclusion as a single JSON object.\n"
                        "Do NOT wrap it in code fences. Output ONLY the JSON:\n"
                        '{"detected": true_or_false, "confidence": 0.0-1.0, "suggested_root_cause": "service or null", "suggested_remediations": ["step1", ...]}'
                    )
                else:
                    nudge = (
                        "You MUST call tools first. Do NOT give a final answer until you have gathered real data.\n"
                        f"Call search_logs now:\n"
                        f'<tool_call>{{"name": "search_logs", "arguments": {{"query": "error", "since_ts": "{tool_ctx["since_ts"]}"}}}}</tool_call>\n'
                        "Copy and emit the <tool_call> block above (you may adjust the query)."
                    )
                messages.append({"role": "user", "content": nudge})
                print(f"[agent] round {round_idx+1}: no tool calls, nudging ({'conclude' if has_used_tools else 'call tools'})", file=sys.stderr, flush=True)
                continue

            _log_to_mlflow(
                tool_calls=tool_ctx.get("tool_calls_log"),
                thinking=tool_ctx.get("thinking_log"),
            )
            return {
                "detected": False,
                "confidence": 0.0,
                "suggested_root_cause": None,
                "suggested_remediations": tool_ctx.get("remediation_steps", []),
                "llm_invoked": True,
            }

        tool_exec_log: list[dict[str, Any]] = []
        result_parts: list[str] = []
        for tc in parsed_tool_calls:
            name = tc["name"]
            args = tc["arguments"]
            print(f"[agent] tool call: {name}({json.dumps(args)[:200]})", file=sys.stderr, flush=True)
            result = execute_tool(name, args, tool_ctx)
            tool_ctx["tool_calls_log"].append({
                "name": name,
                "arguments": args,
                "result_preview": _mlflow_preview(str(result)),
            })
            tool_exec_log.append({
                "name": name,
                "arguments": args,
                "result_preview": _mlflow_preview(str(result)),
            })
            result_parts.append(
                f'<tool_result name="{name}">\n{result[:30_000]}\n</tool_result>'
            )

        tool_ctx["llm_rounds"].append({
            "round": round_idx + 1,
            "model": DEEPSEEK_MODEL,
            "request_messages": req_messages_snapshot,
            "assistant_content": visible_content[:100_000],
            "parsed_tool_calls": [{"name": tc["name"], "arguments": tc["arguments"]} for tc in parsed_tool_calls],
            "tool_executions": tool_exec_log,
        })

        messages.append({"role": "user", "content": "\n\n".join(result_parts)})

        _log_to_mlflow(
            tool_calls=tool_ctx.get("tool_calls_log"),
            thinking=tool_ctx.get("thinking_log"),
        )

    _log_to_mlflow(
        tool_calls=tool_ctx.get("tool_calls_log"),
        thinking=tool_ctx.get("thinking_log"),
    )
    return {
        "detected": False,
        "confidence": 0.0,
        "suggested_root_cause": None,
        "suggested_remediations": tool_ctx.get("remediation_steps", []),
        "llm_invoked": True,
    }


def run_detection(
    ch_http: str,
    since_ts: str,
    mlflow_run_id: str | None = None,
) -> DetectionResult:
    llm_result = run_agentic_loop(ch_http, since_ts, mlflow_run_id)
    detected = llm_result.get("detected", False)
    confidence = llm_result.get("confidence", 0.0)
    suggested_root_cause = llm_result.get("suggested_root_cause")
    suggested_remediations = llm_result.get("suggested_remediations", [])
    llm_invoked = llm_result.get("llm_invoked", False)

    signals: dict[str, Any] = {}
    first_alert_time = datetime.now(timezone.utc).isoformat() if detected else None

    return DetectionResult(
        detected=detected,
        first_alert_time=first_alert_time,
        confidence=confidence,
        suggested_root_cause=suggested_root_cause,
        suggested_remediations=suggested_remediations,
        signals=signals,
        llm_invoked=llm_invoked,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Hosted LLM AIOps Agent (prompt-based tool calling)")
    ap.add_argument("--since", required=True, help="ISO8601 timestamp for detection window start")
    ap.add_argument("--json", action="store_true", help="Output JSON to stdout")
    args = ap.parse_args()

    try:
        ch_http = os.environ.get("CLICKHOUSE_HTTP", CLICKHOUSE_HTTP)
        mlflow_run_id = os.environ.get("MLFLOW_RUN_ID")
        result = run_detection(ch_http, args.since, mlflow_run_id)

        if args.json:
            print(json.dumps(asdict(result), indent=2), flush=True)
        else:
            print(f"detected={result.detected} confidence={result.confidence:.2f}")
            if result.suggested_root_cause:
                print(f"root_cause={result.suggested_root_cause}")
            for r in result.suggested_remediations:
                print(f"  - {r}")

        return 0 if result.detected else 1
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        if args.json:
            fatal = DetectionResult(
                detected=False,
                first_alert_time=None,
                confidence=0.0,
                suggested_root_cause=None,
                suggested_remediations=[],
                signals={"agent_fatal": type(e).__name__, "detail": str(e)[:2000]},
                llm_invoked=False,
            )
            print(json.dumps(asdict(fatal), indent=2), flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
