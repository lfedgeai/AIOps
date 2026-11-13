"""Bounded tool-loop agent — full telemetry query access for detect/RCA.

Purpose
-------
Eval-harness-parity investigation: instead of a single-shot prompt with
pre-digested signals, the model gets a tool surface and a round budget:

  search_logs(query, service_name, since_ts, limit)
  search_traces(service_name, since_ts, min_duration_ms, limit)
  search_metrics(metric_name, service_name, since_ts, limit)
  list_services(since_ts, limit)
  count_errors_by_service(since_ts)        ← fleet-wide error summary
  get_deployment_config(name, namespace)   ← K8s spec inspection

Loop: model emits a tool call (JSON) → harness executes → result fed back →
repeat, until the model returns the final decision schema or the round budget
is exhausted (then it is forced to answer).

How it connects
---------------
- ``slots/detector.py`` and ``slots/rca.py`` run their model through
  ``ToolLoopAgent.run`` with their stage schema.
- Tools execute against the runtime's OWN ClickHouse + kubeconfig.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, ValidationError

from runtime_harness.providers.llm import LLMProvider, ProviderError
from runtime_harness.schemas import extract_json

DEFAULT_MAX_ITERATIONS = 8

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "search_logs",
        "description": "Search OTEL logs. Filter by Body text query and/or ServiceName; since_ts is ISO8601.",
        "args": {
            "query": "optional text to match in log bodies",
            "service_name": "optional service filter",
            "since_ts": "optional ISO timestamp (default: 30 minutes ago)",
            "limit": "max rows (default 20)",
        },
    },
    {
        "name": "search_traces",
        "description": "Search OTEL traces. Filter by service, min duration (ms); since_ts is ISO8601.",
        "args": {
            "service_name": "optional service filter",
            "since_ts": "optional ISO timestamp",
            "min_duration_ms": "optional minimum span duration",
            "limit": "max rows (default 20)",
        },
    },
    {
        "name": "search_metrics",
        "description": "Search OTEL metrics (gauge/sum). Filter by metric name and/or service.",
        "args": {
            "metric_name": "optional metric name",
            "service_name": "optional service filter",
            "since_ts": "optional ISO timestamp",
            "limit": "max rows (default 50)",
        },
    },
    {
        "name": "list_services",
        "description": "List applications seen in telemetry (ServiceName) with signal counts.",
        "args": {"since_ts": "optional ISO timestamp (default: 2 hours ago)", "limit": "default 50"},
    },
    {
        "name": "count_errors_by_service",
        "description": "Fleet-wide error counts per service since a timestamp (error logs AND failed spans) — the fastest way to find WHICH service is failing.",
        "args": {"since_ts": "optional ISO timestamp (default: 30 minutes ago)"},
    },
    {
        "name": "last_activity_by_service",
        "description": "Last telemetry activity (max timestamp) per service. A service whose last activity is minutes older than the fleet's is likely DOWN (e.g. scaled to zero).",
        "args": {},
    },
    {
        "name": "get_deployment_config",
        "description": "Read a Kubernetes deployment spec: image, replicas, per-container env. Omit namespace to target the workload namespace.",
        "args": {"name": "deployment name", "namespace": "optional — omit for workload namespace"},
    },
]


def _default_since(minutes: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


def build_tool_executor(
    ch_client, workload_namespace: str = "otel-demo"
) -> Callable[[str, dict[str, Any]], str]:
    """Wire tool names to the runtime's own ClickHouse + kubeconfig."""

    def count_errors_by_service(since_ts: Optional[str] = None) -> str:
        from runtime_harness.telemetry.clickhouse import _esc

        try:
            dt = datetime.fromisoformat((since_ts or _default_since(30)).replace("Z", "+00:00"))
            since_sql = dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, AttributeError):
            since_sql = _default_since(30)[:19].replace("T", " ")
        sql = (
            f"SELECT service, sum(errors) FROM ("
            f"SELECT ServiceName AS service, count() AS errors FROM {ch_client.database}.otel_logs "
            f"WHERE Timestamp >= '{since_sql}' AND ("
            f"positionCaseInsensitive(SeverityText, 'ERROR') > 0 "
            f"OR positionCaseInsensitive(Body, 'error') > 0) AND ServiceName != '' "
            f"GROUP BY ServiceName "
            f"UNION ALL "
            f"SELECT ServiceName, count() FROM {ch_client.database}.otel_traces "
            f"WHERE Timestamp >= '{since_sql}' AND StatusCode = 'Error' AND ServiceName != '' "
            f"GROUP BY ServiceName) GROUP BY service ORDER BY sum(errors) DESC LIMIT 25 "
            f"FORMAT TabSeparated"
        )
        out = ch_client.query(sql)
        if out.startswith("Error:"):
            return out
        rows = []
        for ln in out.split("\n"):
            if ln.strip():
                parts = ln.split("\t")
                if len(parts) == 2:
                    rows.append({"service": parts[0], "errors": int(parts[1])})
        return json.dumps({"since": since_sql, "errors_by_service": rows}, indent=2)

    def last_activity_by_service() -> str:
        sql = (
            f"SELECT service, max(ts) FROM ("
            f"SELECT ServiceName AS service, max(Timestamp) AS ts FROM {ch_client.database}.otel_logs "
            f"WHERE Timestamp >= now() - INTERVAL 2 HOUR AND ServiceName != '' GROUP BY ServiceName "
            f"UNION ALL "
            f"SELECT ServiceName, max(Timestamp) FROM {ch_client.database}.otel_traces "
            f"WHERE Timestamp >= now() - INTERVAL 2 HOUR AND ServiceName != '' GROUP BY ServiceName) "
            f"GROUP BY service ORDER BY max(ts) DESC LIMIT 30 FORMAT TabSeparated"
        )
        out = ch_client.query(sql)
        if out.startswith("Error:"):
            return out
        rows = []
        for ln in out.split("\n"):
            if ln.strip():
                parts = ln.split("\t")
                if len(parts) == 2:
                    rows.append({"service": parts[0], "last_activity": parts[1][:19]})
        return json.dumps(
            {"now": _default_since(0), "services": rows,
             "note": "services with old last_activity vs the fleet are likely down"},
            indent=2,
        )

    def get_deployment_config(name: str, namespace: Optional[str] = None) -> str:
        ns = namespace or workload_namespace
        try:
            from kubernetes import client as k8s_client, config as k8s_config

            try:
                k8s_config.load_incluster_config()
            except Exception:  # noqa: BLE001
                k8s_config.load_kube_config()
            dep = k8s_client.AppsV1Api().read_namespaced_deployment(name, ns)
            containers = [
                {
                    "name": c.name,
                    "image": c.image,
                    "env": [{"name": e.name, "value": e.value} for e in (c.env or [])],
                }
                for c in dep.spec.template.spec.containers
            ]
            return json.dumps(
                {
                    "name": name,
                    "namespace": ns,
                    "replicas": dep.spec.replicas,
                    "ready_replicas": dep.status.ready_replicas or 0,
                    "containers": containers,
                },
                indent=2,
            )
        except Exception as exc:  # noqa: BLE001
            body = str(exc)
            if "404" in body or "not found" in body.lower():
                return (
                    f"Error: no deployment '{name}' in namespace '{ns}' — the workload "
                    f"namespace is '{workload_namespace}'."
                )
            return f"Error: {exc}"

    from runtime_harness.telemetry import clickhouse as ch

    def execute(name: str, args: dict[str, Any]) -> str:
        args = args or {}
        try:
            if name == "search_logs":
                return ch.search_logs(
                    query=args.get("query") or "",
                    service_name=args.get("service_name"),
                    since_ts=args.get("since_ts"),
                    limit=int(args.get("limit", 20)),
                    client=ch_client,
                )
            if name == "search_traces":
                return ch.search_traces(
                    service_name=args.get("service_name"),
                    since_ts=args.get("since_ts"),
                    min_duration_ms=args.get("min_duration_ms"),
                    limit=int(args.get("limit", 20)),
                    client=ch_client,
                )
            if name == "search_metrics":
                return ch.search_metrics(
                    metric_name=args.get("metric_name"),
                    service_name=args.get("service_name"),
                    since_ts=args.get("since_ts"),
                    limit=int(args.get("limit", 50)),
                    client=ch_client,
                )
            if name == "list_services":
                return ch.list_services(
                    since_ts=args.get("since_ts"), limit=int(args.get("limit", 50)), client=ch_client
                )
            if name == "count_errors_by_service":
                return count_errors_by_service(since_ts=args.get("since_ts"))
            if name == "last_activity_by_service":
                return last_activity_by_service()
            if name == "get_deployment_config":
                return get_deployment_config(
                    name=str(args.get("name", "")), namespace=args.get("namespace")
                )
            return f"Error: unknown tool '{name}'"
        except Exception as exc:  # noqa: BLE001 — feed errors back to the model
            return f"Error: {exc}"

    return execute


def _tool_manual() -> str:
    lines = ["Available tools (call by responding with ONLY a JSON object):",
             '{"tool": "<name>", "args": {<arguments>}}', "", "Tools:"]
    for t in TOOL_SPECS:
        args = ", ".join(f"{k}: {v}" for k, v in t["args"].items())
        lines.append(f"- {t['name']}: {t['description']} Args: {args}")
    return "\n".join(lines)


class ToolLoopAgent:
    """Runs one model through bounded investigation rounds."""

    def __init__(
        self,
        provider: LLMProvider,
        execute: Callable[[str, dict[str, Any]], str],
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ):
        self.provider = provider
        self.execute = execute
        self.max_iterations = max_iterations

    def run(
        self,
        agent_id: str,
        system: str,
        task: str,
        schema: Type[BaseModel],
    ) -> tuple[BaseModel, list[dict[str, Any]]]:
        """Investigate via tools; return (validated decision, tool trace)."""
        messages: list[dict[str, str]] = [
            {
                "role": "user",
                "content": (
                    f"{task}\n\n{_tool_manual()}\n\n"
                    f"Investigate with the tools. When you have enough evidence, stop "
                    f"calling tools and respond with ONLY your final JSON decision "
                    f"(no tool call). You have at most {self.max_iterations} tool rounds."
                ),
            }
        ]
        trace: list[dict[str, Any]] = []
        for _ in range(self.max_iterations):
            content = self.provider._chat_messages(agent_id, system, messages)
            obj = extract_json(content)
            if obj is not None and "tool" in obj and isinstance(obj.get("tool"), str):
                name = obj["tool"]
                args = obj.get("args") or {}
                result = self.execute(name, args)
                trace.append({"tool": name, "args": args, "result_len": len(result)})
                degraded = result.startswith("Error:") or '"source": "stub"' in result[:300]
                note = (
                    "\n[note: this tool is DEGRADED — backend unreachable. Do not "
                    "retry it; conclude with the best available evidence.]"
                    if degraded
                    else ""
                )
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"<tool_result>\n{result[:6000]}\n</tool_result>{note}\n\n"
                            "Continue investigating, or respond with ONLY your final "
                            "JSON decision."
                        ),
                    }
                )
                continue
            # no tool call → final answer
            decision = self._validate(content, schema)
            return decision, trace
        # budget exhausted — force a final answer; models sometimes keep
        # emitting tool calls, so re-ask firmly and tolerate a few failures
        force = (
            "Investigation budget exhausted. DO NOT call any more tools. Respond "
            "with ONLY your final JSON decision now — nothing else."
        )
        messages.append({"role": "user", "content": force})
        for _ in range(3):
            content = self.provider._chat_messages(agent_id, system, messages)
            obj = extract_json(content)
            if obj is not None and "tool" not in obj:
                return self._validate(content, schema), trace
            messages.append({"role": "assistant", "content": content[:2000]})
            messages.append({"role": "user", "content": force})
        raise ProviderError(
            f"{agent_id}: model did not produce a final decision within budget "
            f"({self.max_iterations} rounds + 3 forced finals)"
        )

    @staticmethod
    def _validate(content: str, schema: Type[BaseModel]) -> BaseModel:
        obj = extract_json(content)
        if obj is None:
            raise ProviderError(f"no JSON object in final output: {content[:200]}")
        try:
            return schema.model_validate(obj)
        except ValidationError as exc:
            raise ProviderError(f"final output failed schema validation: {exc}") from exc


__all__ = ["ToolLoopAgent", "build_tool_executor", "TOOL_SPECS"]
