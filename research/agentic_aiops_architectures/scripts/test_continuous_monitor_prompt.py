#!/usr/bin/env python3
"""Test whether qwen3 keeps polling when prompted to continuously monitor."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MONITOR_PROMPT = """You are continuously monitoring system health. Do NOT stop after one check.

Repeat until you find a fault or exhaust useful checks:
1. Call search_logs, search_traces, and/or search_metrics (since_ts below).
2. If unhealthy → output final JSON with detected=true and stop.
3. If still healthy → call MORE tools with different queries and check again.
4. Only output final JSON with detected=false after multiple thorough checks.

Keep monitoring — do not answer with JSON after only one tool round.

since_ts: {since_ts}
"""


def _run(label: str, extra_env: dict[str, str], since: str) -> dict:
    env = os.environ.copy()
    env.update(extra_env)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CLICKHOUSE_HTTP", "http://127.0.0.1:8123")
    env["MAX_TOOL_ITERATIONS"] = "10"
    env["TOOL_PROFILE"] = "telemetry_multimodal"
    if label == "monitor":
        env["AGENT_USER_PROMPT_OVERRIDE"] = MONITOR_PROMPT.replace("{since_ts}", since)

    py = os.environ.get("HARNESS_PYTHON") or sys.executable
    agent = ROOT / "code" / "agents" / "qwen3_agent" / "agent.py"
    print(f"\n=== {label.upper()} ===", file=sys.stderr)
    t0 = time.monotonic()
    r = subprocess.run(
        [py, "-u", str(agent), "--since", since, "--json"],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
        cwd=str(ROOT),
    )
    elapsed = time.monotonic() - t0
    print(f"exit={r.returncode} elapsed={elapsed:.1f}s", file=sys.stderr)
    if r.stderr:
        for line in r.stderr.splitlines():
            if "qwen3_agent" in line or "tool " in line:
                print(f"  {line}", file=sys.stderr)
    parsed = None
    if r.stdout.strip():
        try:
            start, end = r.stdout.find("{"), r.stdout.rfind("}")
            parsed = json.loads(r.stdout[start : end + 1])
        except json.JSONDecodeError:
            pass
    rounds = 0
    tools = 0
    if parsed:
        sig = parsed.get("signals") or {}
        ai = sig.get("ai_metrics") or {}
        rounds = int(ai.get("ai_rounds") or 0)
        tools = int(ai.get("ai_total_tool_calls") or 0)
        rds = sig.get("ai_metrics_rounds") or []
        for rd in rds:
            print(
                f"  round {rd.get('round')}: tools={rd.get('tool_count')} "
                f"llm_latency={rd.get('llm_latency_sec')}s",
                file=sys.stderr,
            )
    return {
        "label": label,
        "elapsed_seconds": round(elapsed, 1),
        "exit_code": r.returncode,
        "llm_rounds": rounds,
        "tool_calls": tools,
        "detected": parsed.get("detected") if parsed else None,
        "agent_error": (r.stderr or "")[:500] if r.returncode == 2 else None,
    }


def main() -> int:
    if (ROOT / "config" / ".env").exists():
        for line in (ROOT / "config" / ".env").read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    since = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
    baseline = _run("baseline", {}, since)
    monitor = _run("monitor", {}, since)

    summary = {
        "since_ts": since,
        "baseline": baseline,
        "continuous_monitor_prompt": monitor,
        "conclusion": {
            "agent_has_time_based_self_polling": False,
            "agent_exits_after_one_subprocess": True,
            "monitor_prompt_more_rounds_than_baseline": monitor["llm_rounds"] > baseline["llm_rounds"],
            "explanation": (
                "Agents loop LLM+tool rounds inside one run (max MAX_TOOL_ITERATIONS), "
                "then emit final JSON and exit. They do not sleep and re-query over wall-clock time. "
                "Harness must re-invoke the agent for repeated checks."
            ),
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
