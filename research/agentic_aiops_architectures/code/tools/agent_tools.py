"""
Shared agent tools for LLM-based AIOps: ClickHouse queries, search, and remediation logging.

Used by all agents (ollama_qwen2, deepseek_agent, qwen3_agent, llama_scout_agent) for the agentic tool-calling loop.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import requests

CLICKHOUSE_HTTP = os.environ.get("CLICKHOUSE_HTTP", "http://127.0.0.1:8123")


def _since_for_clickhouse(since_ts: str) -> str:
    """Convert ISO8601 to ClickHouse-friendly format (YYYY-MM-DD HH:MM:SS)."""
    try:
        dt = datetime.fromisoformat(since_ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return since_ts


def query_clickhouse(sql: str, ch_http: str | None = None) -> str:
    """Run arbitrary SQL against ClickHouse HTTP interface. Returns raw response text."""
    url = ch_http or CLICKHOUSE_HTTP
    try:
        r = requests.post(url, data=sql, timeout=30)
        r.raise_for_status()
        return (r.text or "").strip()
    except Exception as e:
        return f"Error: {e}"


def search_logs(
    query: str,
    since_ts: str,
    limit: int = 20,
    ch_http: str | None = None,
) -> str:
    """Search OTEL logs by Body text. Returns matching log entries (Timestamp, ServiceName, Body)."""
    url = ch_http or CLICKHOUSE_HTTP
    since_sql = _since_for_clickhouse(since_ts)
    # Escape single quotes for SQL string literal
    q_esc = query.replace("'", "''")
    sql = f"""
    SELECT formatDateTime(Timestamp, '%Y-%m-%d %H:%M:%S') as ts, ServiceName, Body
    FROM otel.otel_logs
    WHERE Timestamp >= '{since_sql}' AND positionCaseInsensitive(Body, '{q_esc}') > 0
    ORDER BY Timestamp DESC
    LIMIT {limit}
    FORMAT TabSeparated
    """
    out = query_clickhouse(sql, url)
    if out.startswith("Error:"):
        return out
    lines = [ln for ln in out.split("\n") if ln]
    if not lines:
        return "No matching logs found."
    results = []
    for ln in lines:
        parts = ln.split("\t", 2)
        if len(parts) >= 3:
            results.append({"ts": parts[0], "service": parts[1], "message": parts[2][:500]})
    return json.dumps(results, indent=2)


def search_traces(
    service_name: str | None = None,
    since_ts: str | None = None,
    min_duration_ms: float | None = None,
    limit: int = 20,
    ch_http: str | None = None,
) -> str:
    """Search OTEL traces. Filter by service, time range, min duration. Returns trace summary."""
    url = ch_http or CLICKHOUSE_HTTP
    conditions = ["1=1"]
    if since_ts:
        since_sql = _since_for_clickhouse(since_ts)
        conditions.append(f"Timestamp >= '{since_sql}'")
    if service_name:
        conditions.append(f"ServiceName = '{service_name.replace(chr(39), chr(39)+chr(39))}'")
    if min_duration_ms is not None:
        ns = int(min_duration_ms * 1_000_000)
        conditions.append(f"Duration > {ns}")
    where = " AND ".join(conditions)
    sql = f"""
    SELECT formatDateTime(Timestamp, '%Y-%m-%d %H:%M:%S') as ts, ServiceName, Duration/1000000 as duration_ms
    FROM otel.otel_traces
    WHERE {where}
    ORDER BY Timestamp DESC
    LIMIT {limit}
    FORMAT TabSeparated
    """
    out = query_clickhouse(sql, url)
    if out.startswith("Error:"):
        return out
    lines = [ln for ln in out.split("\n") if ln]
    if not lines:
        return "No matching traces found."
    results = []
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) >= 3:
            try:
                results.append({"ts": parts[0], "service": parts[1], "duration_ms": float(parts[2])})
            except ValueError:
                pass
    return json.dumps(results, indent=2)


def search_metrics_data(
    metric_name: str | None = None,
    since_ts: str | None = None,
    limit: int = 50,
    ch_http: str | None = None,
) -> str:
    """Search OTEL metrics. otel_metrics_gauge/sum use TimeUnix; otel_metrics may use Timestamp."""
    url = ch_http or CLICKHOUSE_HTTP
    since_sql = _since_for_clickhouse(since_ts) if since_ts else "1970-01-01 00:00:00"
    # otel_metrics_gauge/sum have TimeUnix (DateTime64); otel_metrics if exists may use Timestamp
    tables_and_time_col = [
        ("otel.otel_metrics_gauge", "TimeUnix"),
        ("otel.otel_metrics_sum", "TimeUnix"),
        ("otel.otel_metrics", "Timestamp"),
    ]
    for table, time_col in tables_and_time_col:
        try:
            if metric_name:
                sql = f"""
                SELECT * FROM {table}
                WHERE {time_col} >= toDateTime64('{since_sql}', 3) AND MetricName = '{metric_name.replace("'", "''")}'
                ORDER BY {time_col} DESC LIMIT {limit}
                FORMAT TabSeparatedWithNames
                """
            else:
                sql = f"""
                SELECT * FROM {table}
                WHERE {time_col} >= toDateTime64('{since_sql}', 3)
                ORDER BY {time_col} DESC LIMIT {limit}
                FORMAT TabSeparatedWithNames
                """
            out = query_clickhouse(sql, url)
            if out.startswith("Error:"):
                continue
            return out if out else f"Empty result from {table}"
        except Exception:
            continue
    return "No metrics tables found (tried otel_metrics_gauge, otel_metrics_sum, otel_metrics)."


def run_remediation(
    steps: list[str],
    mlflow_run_id: str | None = None,
    accumulated: list[str] | None = None,
) -> str:
    """
    Log remediation steps. If MLFLOW_RUN_ID is set (harness passes it), log to MLflow.
    Accumulates steps in the provided list for the agent's final suggested_remediations.
    """
    if accumulated is not None:
        accumulated.extend(steps)
    run_id = mlflow_run_id or os.environ.get("MLFLOW_RUN_ID")
    all_steps = (accumulated or []) if accumulated else steps
    if run_id and all_steps:
        try:
            import mlflow

            uri = (
                os.environ.get("MLFLOW_TRACKING_URI")
                or os.environ.get("MLFLOW_TRACKING_URL")
                or ""
            ).strip()
            if not uri:
                from code.agents.mlflow_config import mlflow_tracking_uri

                uri = mlflow_tracking_uri()
            mlflow.set_tracking_uri(uri)
            with mlflow.start_run(run_id=run_id):
                mlflow.log_param("remediation_steps", json.dumps(all_steps))
            return f"Logged {len(all_steps)} remediation step(s) to MLflow."
        except Exception as e:
            return f"MLflow log failed: {e}"
    return f"Captured {len(steps)} remediation step(s) (MLflow logging skipped—no run_id)."


# OpenAI-style tool definitions for the LLM
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "query_clickhouse",
            "description": "Run arbitrary SQL against ClickHouse. Use for custom telemetry queries. Tables: otel.otel_logs, otel.otel_traces, otel.otel_metrics (or otel_metrics_gauge/sum).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "Valid ClickHouse SQL query."},
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_logs",
            "description": "Search OTEL logs by text in Body. Returns matching log entries with timestamp, service, message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to search for in log Body (case-insensitive)."},
                    "since_ts": {"type": "string", "description": "ISO8601 timestamp for search window start."},
                    "limit": {"type": "integer", "description": "Max results (default 20).", "default": 20},
                },
                "required": ["query", "since_ts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_traces",
            "description": "Search OTEL traces by service, time range, or min duration. Returns trace summary with ts, service, duration_ms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string", "description": "Filter by ServiceName."},
                    "since_ts": {"type": "string", "description": "ISO8601 timestamp for search window start."},
                    "min_duration_ms": {"type": "number", "description": "Only traces with duration >= this (ms)."},
                    "limit": {"type": "integer", "description": "Max results (default 20).", "default": 20},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_metrics",
            "description": "Search OTEL metrics by name and time range. Returns metric data points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_name": {"type": "string", "description": "Filter by metric name."},
                    "since_ts": {"type": "string", "description": "ISO8601 timestamp for search window start."},
                    "limit": {"type": "integer", "description": "Max results (default 50).", "default": 50},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_remediation",
            "description": "Log remediation steps (for now, logs to MLflow; does not execute actions). Call this when you have determined remediation steps to apply.",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of remediation steps to log.",
                    },
                },
                "required": ["steps"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict[str, Any], ctx: dict[str, Any]) -> str:
    """Execute a tool by name with given arguments. ctx: {ch_http, since_ts, mlflow_run_id}."""
    ch_http = ctx.get("ch_http") or CLICKHOUSE_HTTP
    since_ts = ctx.get("since_ts", "")
    mlflow_run_id = ctx.get("mlflow_run_id")

    if name == "query_clickhouse":
        return query_clickhouse(arguments.get("sql", ""), ch_http)
    if name == "search_logs":
        return search_logs(
            arguments.get("query", ""),
            arguments.get("since_ts", since_ts),
            arguments.get("limit", 20),
            ch_http,
        )
    if name == "search_traces":
        return search_traces(
            arguments.get("service_name"),
            arguments.get("since_ts", since_ts),
            arguments.get("min_duration_ms"),
            arguments.get("limit", 20),
            ch_http,
        )
    if name == "search_metrics":
        return search_metrics_data(
            metric_name=arguments.get("metric_name"),
            since_ts=arguments.get("since_ts") or since_ts,
            limit=arguments.get("limit", 50),
            ch_http=ch_http,
        )
    if name == "run_remediation":
        steps = arguments.get("steps", [])
        if isinstance(steps, str):
            steps = [steps]
        accumulated = ctx.get("remediation_steps")
        return run_remediation(steps, mlflow_run_id, accumulated)
    return f"Unknown tool: {name}"
