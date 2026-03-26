#!/usr/bin/env python3
"""
Reference Agent: LLM-based AIOps failure detection using tool-calling (agentic loop).

Uses an OpenAI-compatible API (Ollama, vLLM, etc.) with tools: query_clickhouse,
search_logs, search_traces, search_metrics, run_remediation. The LLM iteratively
calls tools to gather telemetry, then returns detection + remediations.
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

# Ensure project root is on path for code.tools
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
# Use 127.0.0.1 by default: "localhost" can resolve to IPv6 (::1) while Ollama often listens on IPv4 only → long hangs.
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:11434/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "qwen2.5")
MAX_TOOL_ITERATIONS = int(os.environ.get("MAX_TOOL_ITERATIONS", "8"))
# Per-request HTTP timeout (seconds) — avoids hanging forever if Ollama stalls
OPENAI_TIMEOUT_SEC = float(os.environ.get("OPENAI_TIMEOUT_SEC", "180"))


@dataclass
class DetectionResult:
    """Result of one agent invocation."""
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


def _ollama_style_tool_followups(base_url: str) -> bool:
    """
    Ollama's OpenAI-compatible /v1/chat often hangs on transcripts that mix assistant.tool_calls with role=tool.
    Use plain assistant text plus user-role tool outputs instead (same pattern as inline-tool parsing).
    """
    ex = os.environ.get("OLLAMA_STYLE_TOOL_FOLLOWUPS", "").strip().lower()
    if ex in ("0", "false", "no", "off"):
        return False
    if ex in ("1", "true", "yes", "on"):
        return True
    u = (base_url or "").lower()
    return ":11434" in u or "localhost:11434" in u or "127.0.0.1:11434" in u


def _truncate_tool_result(text: str, max_chars: int = 14_000) -> str:
    """Keep LLM context small so local models (Ollama) don't stall on huge tool payloads."""
    if not text or len(text) <= max_chars:
        return text
    return text[: max_chars - 80] + "\n... [truncated for context size] ..."


def _extract_inline_tool_calls_from_content(content: str) -> list[dict[str, Any]]:
    """Parse {\"name\": \"...\", \"arguments\": {...}} blobs from model text (Ollama often sets tool_calls=None)."""
    if not content or '"name"' not in content:
        return []
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    i = 0
    n = len(content)
    while i < n:
        if content[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(content[i:])
            if isinstance(obj, dict) and "name" in obj:
                args = obj.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                found.append({"name": str(obj["name"]), "arguments": args})
            i += end
        except json.JSONDecodeError:
            i += 1
    return found


def run_agentic_loop(
    ch_http: str,
    since_ts: str,
    mlflow_run_id: str | None = None,
) -> dict[str, Any]:
    """Run the agentic tool-calling loop; return parsed detection result."""
    tool_ctx: dict[str, Any] = {
        "ch_http": ch_http,
        "since_ts": since_ts,
        "mlflow_run_id": mlflow_run_id or os.environ.get("MLFLOW_RUN_ID"),
        "remediation_steps": [],
        "tool_calls_log": [],
        "thinking_log": [],  # assistant message content (reasoning) for MLflow
        "llm_rounds": [],  # per API call: request messages + assistant + tool executions (MLflow)
    }

    user_msg = f"""Analyze telemetry since {since_ts}. Use the tools to search logs, traces, and metrics. Determine if there is a failure and suggest remediations. Respond with JSON when done."""

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
            "openai_model": OPENAI_MODEL,
            "openai_api_base": (OPENAI_API_BASE or "").rstrip("/"),
            "registered_tool_names": [t.get("function", {}).get("name") for t in TOOL_DEFINITIONS],
            # Full conversation as sent to the LLM on the latest turn (includes tool results).
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

    _log_to_mlflow()  # Prompts + initial conversation before LLM import (captures run even if deps fail)

    try:
        from openai import OpenAI
    except ImportError:
        _log_to_mlflow(prompts={"agent_note": "openai package not installed; LLM not invoked"})
        return {"detected": False, "confidence": 0.0, "suggested_root_cause": None, "suggested_remediations": [], "llm_invoked": False}

    base_url = (OPENAI_API_BASE or "").rstrip("/")
    client_kwargs: dict[str, Any] = {"api_key": OPENAI_API_KEY or "ollama"}
    if base_url:
        client_kwargs["base_url"] = base_url
    try:
        import httpx

        client_kwargs["timeout"] = httpx.Timeout(
            connect=15.0,
            read=OPENAI_TIMEOUT_SEC,
            write=120.0,
            pool=10.0,
        )
    except ImportError:
        client_kwargs["timeout"] = OPENAI_TIMEOUT_SEC
    client = OpenAI(**client_kwargs)

    for round_idx in range(MAX_TOOL_ITERATIONS):
        print(
            f"[ollama_qwen2] LLM round {round_idx + 1}/{MAX_TOOL_ITERATIONS} (model={OPENAI_MODEL})",
            file=sys.stderr,
            flush=True,
        )
        req_messages_snapshot = serialize_messages_for_mlflow(messages)

        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as e:
            tool_ctx["llm_rounds"].append(
                {
                    "round": round_idx + 1,
                    "model": OPENAI_MODEL,
                    "request_messages": req_messages_snapshot,
                    "api_error": str(e),
                }
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
                "model": OPENAI_MODEL,
                "request_messages": req_messages_snapshot,
                "assistant_content": (content or "")[:100_000],
                "native_tool_calls_from_api": serialize_tool_calls_for_mlflow(api_tool_calls),
            }
            if tool_executions is not None:
                entry["tool_executions"] = tool_executions
            if note:
                entry["note"] = note
            tool_ctx["llm_rounds"].append(entry)

        # Native OpenAI-style tool_calls: strict API uses assistant.tool_calls + role=tool; Ollama needs user follow-ups.
        if api_tool_calls:
            use_ollama_followups = _ollama_style_tool_followups(base_url)
            if use_ollama_followups:
                planned: list[str] = []
                for tc in api_tool_calls:
                    fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
                    name = fn.name if hasattr(fn, "name") else fn.get("name", "")
                    args_str = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", "{}")
                    planned.append(f"{name}({args_str})")
                asst_body = (content or "").strip()
                suffix = "[Tool calls requested: " + "; ".join(planned) + "]"
                asst_body = f"{asst_body}\n\n{suffix}" if asst_body else suffix
                messages.append({"role": "assistant", "content": asst_body})
            else:
                tc_serialized: list[dict[str, Any]] = []
                for tc in api_tool_calls:
                    tid = tc.id if hasattr(tc, "id") else tc.get("id", "")
                    fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
                    name = fn.name if hasattr(fn, "name") else fn.get("name", "")
                    args_str = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", "{}")
                    tc_serialized.append(
                        {"id": tid, "type": "function", "function": {"name": name, "arguments": args_str}}
                    )
                messages.append(
                    {
                        "role": "assistant",
                        "content": content if content else None,
                        "tool_calls": tc_serialized,
                    }
                )
            tool_exec_log: list[dict[str, Any]] = []
            for tc in api_tool_calls:
                name = tc.function.name if hasattr(tc.function, "name") else tc.get("function", {}).get("name", "")
                args_str = tc.function.arguments if hasattr(tc.function, "arguments") else tc.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}
                tag = "native" if not use_ollama_followups else "native→ollama-compat"
                print(f"[ollama_qwen2] tool {name} ({tag})", file=sys.stderr, flush=True)
                result = _truncate_tool_result(execute_tool(name, args, tool_ctx))
                tool_ctx["tool_calls_log"].append(
                    {
                        "name": name,
                        "arguments": args,
                        "result_preview": _mlflow_preview(str(result)),
                    }
                )
                tool_exec_log.append(
                    {"name": name, "arguments": args, "result_preview": _mlflow_preview(str(result))}
                )
                if use_ollama_followups:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"[Tool `{name}` output]\n{result}",
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id if hasattr(tc, "id") else tc.get("id", ""),
                            "content": result,
                        }
                    )
            _append_llm_round(tool_executions=tool_exec_log)
            _log_to_mlflow(
                tool_calls=tool_ctx.get("tool_calls_log"),
                thinking=tool_ctx.get("thinking_log"),
            )
            continue

        raw_content = (msg.content or "").strip()
        # Ollama: tool JSON in message body; role=tool + empty tool_calls confuses many servers — use user follow-ups.
        inline_tools = _extract_inline_tool_calls_from_content(raw_content)
        if inline_tools:
            messages.append({"role": "assistant", "content": content})
            tool_exec_log = []
            for inv in inline_tools:
                name = inv.get("name", "")
                args = inv.get("arguments") or {}
                print(f"[ollama_qwen2] tool {name} (inline)", file=sys.stderr, flush=True)
                result = _truncate_tool_result(execute_tool(name, args, tool_ctx))
                tool_ctx["tool_calls_log"].append(
                    {
                        "name": name,
                        "arguments": args,
                        "result_preview": _mlflow_preview(str(result)),
                    }
                )
                tool_exec_log.append(
                    {"name": name, "arguments": args, "result_preview": _mlflow_preview(str(result))}
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"[Tool `{name}` output]\n{result}",
                    }
                )
            _append_llm_round(tool_executions=tool_exec_log, note="inline_tool_json_in_assistant_text")
            _log_to_mlflow(
                tool_calls=tool_ctx.get("tool_calls_log"),
                thinking=tool_ctx.get("thinking_log"),
            )
            continue

        parse_content = raw_content
        if parse_content:
            parse_content = re.sub(r"^```(?:json)?\s*", "", parse_content)
            parse_content = re.sub(r"\s*```\s*$", "", parse_content)
            try:
                parsed = json.loads(parse_content)
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
                # Model still reasoning; keep conversation and ask another round.
                _append_llm_round(note="non_json_assistant_text_nudge_follows")
                _log_to_mlflow(
                    tool_calls=tool_ctx.get("tool_calls_log"),
                    thinking=tool_ctx.get("thinking_log"),
                )
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": "Reply with only the final JSON object (no markdown): "
                        '{"detected": true|false, "confidence": 0.0-1.0, "suggested_root_cause": "...", "suggested_remediations": ["..."]}',
                    }
                )
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
    """Run failure detection via agentic tool-calling loop."""
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
    ap = argparse.ArgumentParser(description="LLM-based AIOps Agent (agentic tool-calling)")
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
