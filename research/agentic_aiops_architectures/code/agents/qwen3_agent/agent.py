#!/usr/bin/env python3
"""
Qwen3 Agent: AIOps failure detection using native tool calling.

Uses Qwen3-14B via LiteLLM/OpenAI-compatible API with native function calling
support.  Tools: query_clickhouse, search_logs, search_traces,
search_metrics, run_remediation.
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
    serialize_tool_calls_for_mlflow,
)
from code.agents.mlflow_config import mlflow_tracking_uri
from code.tools.agent_tools import TOOL_DEFINITIONS, execute_tool

CLICKHOUSE_HTTP = os.environ.get("CLICKHOUSE_HTTP", "http://127.0.0.1:8123")

QWEN3_API_BASE = os.environ.get(
    "QWEN3_API_BASE",
    "https://litellm-prod.apps.maas.redhatworkshops.io/v1",
)
QWEN3_API_KEY = os.environ.get("QWEN3_API_KEY", "")
QWEN3_MODEL = os.environ.get("QWEN3_MODEL", "qwen3-14b")
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


SYSTEM_PROMPT = """You are an AIOps expert analyzing observability data from a microservices application (OTEL demo).

You have access to tools to query ClickHouse: query_clickhouse, search_logs, search_traces, search_metrics, and run_remediation.

Your task: determine if there is a failure or anomaly requiring remediation, and suggest remediation steps.

Process:
1. Use search_logs, search_traces, search_metrics to gather telemetry. The since_ts parameter is the detection window start (ISO8601).
2. Analyze the data.
3. When you have your conclusion, respond with a JSON object (no markdown):
   {"detected": true|false, "confidence": 0.0-1.0, "suggested_root_cause": "service or null", "suggested_remediations": ["step1", "step2", ...]}
4. If detected=true, call run_remediation with your suggested steps before giving the final JSON.

Rules:
- detected=true if meaningful errors or latency indicate a fault
- detected=false if data is sparse or normal
- confidence: 0.5=uncertain, 0.9=high
- suggested_remediations: 2-5 concrete, actionable steps
"""


def _truncate_tool_result(text: str, max_chars: int = 14_000) -> str:
    if not text or len(text) <= max_chars:
        return text
    return text[: max_chars - 80] + "\n... [truncated for context size] ..."


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
        f"Analyze telemetry since {since_ts}. Use the tools to search logs, traces, "
        "and metrics. Determine if there is a failure and suggest remediations. "
        "Respond with JSON when done."
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
            "openai_model": QWEN3_MODEL,
            "openai_api_base": (QWEN3_API_BASE or "").rstrip("/"),
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

    try:
        from openai import OpenAI
    except ImportError:
        _log_to_mlflow(prompts={"agent_note": "openai package not installed; LLM not invoked"})
        return {"detected": False, "confidence": 0.0, "suggested_root_cause": None, "suggested_remediations": [], "llm_invoked": False}

    client_kwargs: dict[str, Any] = {
        "api_key": QWEN3_API_KEY or "nokey",
        "base_url": (QWEN3_API_BASE or "").rstrip("/"),
    }
    try:
        import httpx
        client_kwargs["timeout"] = httpx.Timeout(connect=15.0, read=180.0, write=120.0, pool=10.0)
    except ImportError:
        client_kwargs["timeout"] = 180.0
    client = OpenAI(**client_kwargs)

    for round_idx in range(MAX_TOOL_ITERATIONS):
        print(
            f"[qwen3_agent] LLM round {round_idx + 1}/{MAX_TOOL_ITERATIONS} (model={QWEN3_MODEL})",
            file=sys.stderr,
            flush=True,
        )
        req_messages_snapshot = serialize_messages_for_mlflow(messages)

        try:
            response = client.chat.completions.create(
                model=QWEN3_MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as e:
            tool_ctx["llm_rounds"].append({
                "round": round_idx + 1,
                "model": QWEN3_MODEL,
                "request_messages": req_messages_snapshot,
                "api_error": str(e),
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
                "llm_error": str(e),
            }

        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
        api_tool_calls = getattr(msg, "tool_calls", None) or []
        if content.strip():
            tool_ctx["thinking_log"].append({"step": len(tool_ctx["thinking_log"]) + 1, "content": content})

        def _append_llm_round(
            *,
            tool_executions: list[dict[str, Any]] | None = None,
            note: str | None = None,
        ) -> None:
            entry: dict[str, Any] = {
                "round": round_idx + 1,
                "model": QWEN3_MODEL,
                "request_messages": req_messages_snapshot,
                "assistant_content": (content or "")[:100_000],
                "native_tool_calls_from_api": serialize_tool_calls_for_mlflow(api_tool_calls),
            }
            if tool_executions is not None:
                entry["tool_executions"] = tool_executions
            if note:
                entry["note"] = note
            tool_ctx["llm_rounds"].append(entry)

        if api_tool_calls:
            tc_serialized: list[dict[str, Any]] = []
            for tc in api_tool_calls:
                tid = tc.id if hasattr(tc, "id") else tc.get("id", "")
                fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
                name = fn.name if hasattr(fn, "name") else fn.get("name", "")
                args_str = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", "{}")
                tc_serialized.append(
                    {"id": tid, "type": "function", "function": {"name": name, "arguments": args_str}}
                )
            messages.append({
                "role": "assistant",
                "content": content if content else None,
                "tool_calls": tc_serialized,
            })

            tool_exec_log: list[dict[str, Any]] = []
            for tc in api_tool_calls:
                fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
                name = fn.name if hasattr(fn, "name") else fn.get("name", "")
                args_str = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}
                print(f"[qwen3_agent] tool {name} (native)", file=sys.stderr, flush=True)
                result = _truncate_tool_result(execute_tool(name, args, tool_ctx))
                tool_ctx["tool_calls_log"].append({
                    "name": name,
                    "arguments": args,
                    "result_preview": _mlflow_preview(str(result)),
                })
                tool_exec_log.append(
                    {"name": name, "arguments": args, "result_preview": _mlflow_preview(str(result))}
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id if hasattr(tc, "id") else tc.get("id", ""),
                    "content": result,
                })
            _append_llm_round(tool_executions=tool_exec_log)
            _log_to_mlflow(
                tool_calls=tool_ctx.get("tool_calls_log"),
                thinking=tool_ctx.get("thinking_log"),
            )
            continue

        raw_content = (msg.content or "").strip()
        if raw_content:
            parse_content = re.sub(r"^```(?:json)?\s*", "", raw_content)
            parse_content = re.sub(r"\s*```\s*$", "", parse_content)
            try:
                parsed = json.loads(parse_content)
                if not isinstance(parsed, dict):
                    raise json.JSONDecodeError("expected object", parse_content, 0)
                remediations = tool_ctx.get("remediation_steps") or parsed.get("suggested_remediations", [])
                _append_llm_round(note="final_detection_json")
                _log_to_mlflow(
                    tool_calls=tool_ctx.get("tool_calls_log"),
                    thinking=tool_ctx.get("thinking_log"),
                )
                return {
                    "detected": bool(parsed.get("detected", False)),
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "suggested_root_cause": parsed.get("suggested_root_cause") or None,
                    "suggested_remediations": remediations if remediations else list(parsed.get("suggested_remediations", [])),
                    "llm_invoked": True,
                }
            except json.JSONDecodeError:
                _append_llm_round(note="non_json_assistant_text_nudge_follows")
                _log_to_mlflow(
                    tool_calls=tool_ctx.get("tool_calls_log"),
                    thinking=tool_ctx.get("thinking_log"),
                )
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": "Reply with only the final JSON object (no markdown): "
                    '{"detected": true|false, "confidence": 0.0-1.0, "suggested_root_cause": "...", "suggested_remediations": ["..."]}',
                })
                continue

        _append_llm_round(note="empty_or_unparsed_assistant_message")
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
    ap = argparse.ArgumentParser(description="Qwen3 AIOps Agent (native tool calling)")
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
