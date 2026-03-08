#!/usr/bin/env python3
"""
LLM-based AIOps Agent: failure detection and remediation suggestions.

Queries ClickHouse for telemetry, sends context to an LLM for analysis,
and returns structured detection + remediation output.
Designed for harness integration: MTTD/MTTR evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import requests

# Defaults: assume port-forward or in-cluster
CLICKHOUSE_HTTP = os.environ.get("CLICKHOUSE_HTTP", "http://localhost:8123")

# LLM config: OpenAI-compatible API (OpenAI, Ollama, vLLM, etc.)
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "")  # e.g. http://localhost:11434/v1
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "llama3.2")


@dataclass
class DetectionResult:
    """Result of one agent invocation."""
    detected: bool
    first_alert_time: str | None  # ISO8601
    confidence: float
    suggested_root_cause: str | None
    suggested_remediations: list[str]
    signals: dict[str, Any]


def query_clickhouse(ch_http: str, sql: str) -> str:
    """Run SQL against ClickHouse HTTP interface. Returns raw response text."""
    try:
        r = requests.post(ch_http, data=sql, timeout=30)
        r.raise_for_status()
        return (r.text or "").strip()
    except Exception as e:
        return f"Error: {e}"


def fetch_telemetry_context(ch_http: str, since_ts: str, max_log_lines: int = 30) -> dict[str, Any]:
    """Fetch logs, trace stats, and error summary from ClickHouse."""
    ctx: dict[str, Any] = {}

    # Error log count
    out = query_clickhouse(
        ch_http,
        f"SELECT count() FROM otel.otel_logs WHERE SeverityText = 'ERROR' AND Timestamp >= '{since_ts}'",
    )
    try:
        ctx["error_log_count"] = int(out.split("\n")[0] or 0)
    except (ValueError, IndexError):
        ctx["error_log_count"] = 0

    # High-latency span count (duration > 5s)
    ns_5s = 5_000_000_000
    out = query_clickhouse(
        ch_http,
        f"SELECT count() FROM otel.otel_traces WHERE Timestamp >= '{since_ts}' AND Duration > {ns_5s}",
    )
    try:
        ctx["high_latency_span_count"] = int(out.split("\n")[0] or 0)
    except (ValueError, IndexError):
        ctx["high_latency_span_count"] = 0

    # Sample recent error logs (Timestamp, ServiceName, Body)
    sql = f"""
    SELECT formatDateTime(Timestamp, '%Y-%m-%d %H:%M:%S') as ts, ServiceName, Body
    FROM otel.otel_logs
    WHERE SeverityText = 'ERROR' AND Timestamp >= '{since_ts}'
    ORDER BY Timestamp DESC
    LIMIT {max_log_lines}
    FORMAT TabSeparated
    """
    out = query_clickhouse(ch_http, sql)
    lines = [ln for ln in out.split("\n") if ln and not ln.startswith("Error:")]
    ctx["recent_errors"] = []
    for ln in lines[:max_log_lines]:
        parts = ln.split("\t", 2)
        if len(parts) >= 3:
            ctx["recent_errors"].append({"ts": parts[0], "service": parts[1], "message": parts[2][:500]})
        elif len(parts) == 2:
            ctx["recent_errors"].append({"ts": parts[0], "service": parts[1], "message": ""})

    # Top error services
    sql = f"""
    SELECT ServiceName, count() as cnt
    FROM otel.otel_logs
    WHERE SeverityText = 'ERROR' AND Timestamp >= '{since_ts}'
    GROUP BY ServiceName
    ORDER BY cnt DESC
    LIMIT 10
    FORMAT TabSeparated
    """
    out = query_clickhouse(ch_http, sql)
    ctx["top_error_services"] = []
    for ln in out.split("\n"):
        if ln and not ln.startswith("Error:"):
            parts = ln.split("\t")
            if len(parts) >= 2:
                try:
                    ctx["top_error_services"].append({"service": parts[0], "count": int(parts[1])})
                except ValueError:
                    pass

    # Slow spans summary (service, count, max_duration_ms)
    sql = f"""
    SELECT ServiceName, count() as cnt, max(Duration)/1000000 as max_ms
    FROM otel.otel_traces
    WHERE Timestamp >= '{since_ts}' AND Duration > {ns_5s}
    GROUP BY ServiceName
    ORDER BY cnt DESC
    LIMIT 5
    FORMAT TabSeparated
    """
    out = query_clickhouse(ch_http, sql)
    ctx["slow_spans"] = []
    for ln in out.split("\n"):
        if ln and not ln.startswith("Error:"):
            parts = ln.split("\t")
            if len(parts) >= 3:
                try:
                    ctx["slow_spans"].append(
                        {"service": parts[0], "count": int(parts[1]), "max_ms": float(parts[2])}
                    )
                except (ValueError, IndexError):
                    pass

    return ctx


def call_llm(context: dict[str, Any]) -> dict[str, Any]:
    """Call LLM with telemetry context; return parsed detection result."""
    try:
        from openai import OpenAI
    except ImportError:
        return {"detected": False, "confidence": 0.0, "suggested_root_cause": None, "suggested_remediations": []}

    base_url = OPENAI_API_BASE or None
    if base_url and not base_url.endswith("/"):
        base_url = base_url.rstrip("/")

    client_kwargs: dict[str, Any] = {"api_key": OPENAI_API_KEY or "ollama"}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    prompt = f"""You are an AIOps expert analyzing observability data from a microservices application (OTEL demo).

Analyze the following telemetry and determine if there is a failure or anomaly requiring remediation.

TELEMETRY DATA (since {context.get('_since_ts', 'unknown')}):
- Error log count: {context.get('error_log_count', 0)}
- High-latency span count (>5s): {context.get('high_latency_span_count', 0)}
- Top error services: {json.dumps(context.get('top_error_services', []))}
- Slow spans by service: {json.dumps(context.get('slow_spans', []))}
- Recent error log samples:
{json.dumps(context.get('recent_errors', [])[:15], indent=2)}

Respond with a JSON object only (no markdown, no explanation outside JSON):
{{
  "detected": true or false,
  "confidence": 0.0 to 1.0,
  "suggested_root_cause": "service or component name or null",
  "suggested_remediations": ["step 1", "step 2", ...]
}}

Rules:
- detected=true if there are meaningful errors or latency issues indicating a fault
- detected=false if data is sparse, normal, or no clear failure pattern
- confidence reflects how sure you are (0.5 = uncertain, 0.9 = high confidence)
- suggested_remediations: 2-5 concrete, actionable steps
"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = (response.choices[0].message.content or "").strip()
    except Exception as e:
        return {
            "detected": False,
            "confidence": 0.0,
            "suggested_root_cause": None,
            "suggested_remediations": [f"LLM call failed: {e}"],
        }

    # Parse JSON from response (handle markdown code blocks)
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```\s*$", "", content)
    try:
        parsed = json.loads(content)
        return {
            "detected": bool(parsed.get("detected", False)),
            "confidence": float(parsed.get("confidence", 0.0)),
            "suggested_root_cause": parsed.get("suggested_root_cause") or None,
            "suggested_remediations": list(parsed.get("suggested_remediations", [])),
        }
    except json.JSONDecodeError:
        return {
            "detected": False,
            "confidence": 0.0,
            "suggested_root_cause": None,
            "suggested_remediations": [f"Failed to parse LLM response: {content[:200]}"],
        }


def run_detection(
    ch_http: str,
    since_ts: str,
    max_log_lines: int = 30,
) -> DetectionResult:
    """
    Run failure detection: fetch telemetry from ClickHouse, ask LLM, return DetectionResult.
    """
    context = fetch_telemetry_context(ch_http, since_ts, max_log_lines)
    context["_since_ts"] = since_ts

    signals = {
        "log_error_count": context.get("error_log_count", 0),
        "high_latency_span_count": context.get("high_latency_span_count", 0),
        "top_error_services": [s.get("service", "") for s in context.get("top_error_services", [])],
    }

    llm_result = call_llm(context)

    detected = llm_result.get("detected", False)
    confidence = llm_result.get("confidence", 0.0)
    suggested_root_cause = llm_result.get("suggested_root_cause")
    suggested_remediations = llm_result.get("suggested_remediations", [])

    # Fallback when LLM unavailable or returned no remediations: use threshold-based detection
    if not suggested_remediations and (signals["log_error_count"] >= 5 or signals["high_latency_span_count"] >= 3):
        detected = True
        confidence = max(confidence, 0.5)
        suggested_root_cause = suggested_root_cause or (
            signals["top_error_services"][0] if signals["top_error_services"] else None
        )
        suggested_remediations = [
            "Check service health and restart if needed",
            "Review logs for details",
        ]

    first_alert_time = datetime.now(timezone.utc).isoformat() if detected else None

    return DetectionResult(
        detected=detected,
        first_alert_time=first_alert_time,
        confidence=confidence,
        suggested_root_cause=suggested_root_cause,
        suggested_remediations=suggested_remediations,
        signals=signals,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM-based AIOps Agent: detect failures, suggest remediations")
    ap.add_argument("--since", required=True, help="ISO8601 timestamp for detection window start")
    ap.add_argument("--max-log-lines", type=int, default=30, help="Max error log lines to send to LLM")
    ap.add_argument("--json", action="store_true", help="Output JSON to stdout")
    args = ap.parse_args()

    ch_http = os.environ.get("CLICKHOUSE_HTTP", CLICKHOUSE_HTTP)
    result = run_detection(ch_http, args.since, max_log_lines=args.max_log_lines)

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print(f"detected={result.detected} confidence={result.confidence:.2f}")
        if result.suggested_root_cause:
            print(f"root_cause={result.suggested_root_cause}")
        for r in result.suggested_remediations:
            print(f"  - {r}")

    return 0 if result.detected else 1


if __name__ == "__main__":
    sys.exit(main())
