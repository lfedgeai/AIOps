"""MCP server — eval-harness-parity tool surface over the runtime's telemetry.

Purpose
-------
Expose the same tool categories the evaluation harness validated:
  reads : list_services, search_logs, search_metrics, search_traces,
          get_deployment_config
  writes: scale_deployment, restart_deployment, delete_pod, patch_deployment
          (GATED — refused unless AIOPS_RT_ENABLE_WRITE_TOOLS=1; the harness,
          never the model, owns execution)

Namespace discipline (Oct-26 lesson): schemas say "omit to target the
workload namespace"; 404-style errors are self-healing messages.

How it connects
---------------
- Telemetry tools delegate to ``telemetry/clickhouse.py`` (the runtime's OWN
  ClickHouse instance).
- K8s tools use the ``kubernetes`` python client (kubeconfig / in-cluster).
- Started via ``aiops-runtime mcp`` (see ``cli.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Optional

from runtime_harness.telemetry import clickhouse as ch

WRITE_TOOLS_ENABLED = os.environ.get("AIOPS_RT_ENABLE_WRITE_TOOLS", "") == "1"
WORKLOAD_NAMESPACE = os.environ.get("AIOPS_RT_NAMESPACE", "otel-demo")


def _k8s():
    from kubernetes import client as k8s_client, config as k8s_config

    try:
        k8s_config.load_incluster_config()
    except Exception:  # noqa: BLE001
        k8s_config.load_kube_config()
    return k8s_client


def _guard_writes() -> Optional[str]:
    if WRITE_TOOLS_ENABLED:
        return None
    return (
        "Error: write tools are disabled. The harness (never the model) owns "
        "execution. Set AIOPS_RT_ENABLE_WRITE_TOOLS=1 to enable, and only with "
        "dry_run/HIL policy in place."
    )


def build_mcp():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "mcp package required: pip install 'mcp>=1.9.0' (or pip install -e '.[mcp]')"
        ) from exc

    ch_url = os.environ.get("CLICKHOUSE_RT_HTTP") or os.environ.get(
        "CLICKHOUSE_HTTP", "http://127.0.0.1:38124"
    )
    database = os.environ.get("CLICKHOUSE_DATABASE", "otel")
    client = ch.ClickHouseClient(base_url=ch_url, database=database)

    mcp = FastMCP(
        name="aiops-runtime-telemetry",
        instructions=(
            "Read-only OTEL telemetry tools backed by the runtime's own ClickHouse, "
            "plus Kubernetes inspection. Use list_services to discover adopted "
            "applications from signals, then search_logs / search_metrics / "
            "search_traces / get_deployment_config for investigation. Do not assume "
            "a fixed service inventory. Write tools are harness-gated."
        ),
    )

    # ------------------------------------------------------------------
    # Telemetry reads (own ClickHouse)
    # ------------------------------------------------------------------

    @mcp.tool(name="list_services")
    def list_services(since_ts: Optional[str] = None, limit: int = 50) -> str:
        """List applications self-adopted from OTEL telemetry (ServiceName).

        Prefer this over hard-coding workload names.
        """
        return ch.list_services(since_ts=since_ts, limit=limit, client=client)

    @mcp.tool(name="search_logs")
    def search_logs(
        query: str = "",
        since_ts: Optional[str] = None,
        service_name: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        """Search OTEL logs (Body text + optional ServiceName)."""
        return ch.search_logs(
            query=query, since_ts=since_ts, service_name=service_name, limit=limit, client=client
        )

    @mcp.tool(name="search_traces")
    def search_traces(
        service_name: Optional[str] = None,
        since_ts: Optional[str] = None,
        min_duration_ms: Optional[float] = None,
        limit: int = 20,
    ) -> str:
        """Search OTEL traces (service, time, min duration filters)."""
        return ch.search_traces(
            service_name=service_name,
            since_ts=since_ts,
            min_duration_ms=min_duration_ms,
            limit=limit,
            client=client,
        )

    @mcp.tool(name="search_metrics")
    def search_metrics(
        metric_name: Optional[str] = None,
        since_ts: Optional[str] = None,
        service_name: Optional[str] = None,
        limit: int = 50,
    ) -> str:
        """Search OTEL metrics (gauge/sum tables)."""
        return ch.search_metrics(
            metric_name=metric_name,
            since_ts=since_ts,
            service_name=service_name,
            limit=limit,
            client=client,
        )

    # ------------------------------------------------------------------
    # Kubernetes reads (eval-harness parity: config discoverability)
    # ------------------------------------------------------------------

    @mcp.tool(name="get_deployment_config")
    def get_deployment_config(
        name: str,
        namespace: Optional[str] = None,
    ) -> str:
        """Read a deployment's live spec: image, replicas, per-container env.

        Namespace: omit to target the workload namespace
        (""" + WORKLOAD_NAMESPACE + """). This is how spec-level faults
        (e.g. a corrupt env var) become discoverable.
        """
        ns = namespace or WORKLOAD_NAMESPACE
        try:
            apps = _k8s().AppsV1Api()
            dep = apps.read_namespaced_deployment(name, ns)
            containers = [
                {
                    "name": c.name,
                    "image": c.image,
                    "env": [
                        {"name": e.name, "value": e.value} for e in (c.env or [])
                    ],
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
                    f"Error: no deployment '{name}' in namespace '{ns}' — "
                    f"the workload namespace is '{WORKLOAD_NAMESPACE}'. Retry "
                    f"with namespace omitted or set to '{WORKLOAD_NAMESPACE}'."
                )
            return f"Error: {exc}"

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Kubernetes writes — GATED (harness owns execution; shared with the
    # pipeline's Direct remediation mode via write_tools.py)
    # ------------------------------------------------------------------

    @mcp.tool(name="scale_deployment")
    def scale_deployment(name: str, replicas: int, namespace: Optional[str] = None) -> str:
        """Scale a deployment to `replicas`. Write tool — harness-gated."""
        refused = _guard_writes()
        if refused:
            return refused
        from runtime_harness.write_tools import execute_write

        return execute_write("scale_deployment", {"name": name, "replicas": replicas}, namespace)

    @mcp.tool(name="restart_deployment")
    def restart_deployment(name: str, namespace: Optional[str] = None) -> str:
        """Rollout-restart a deployment. Write tool — harness-gated."""
        refused = _guard_writes()
        if refused:
            return refused
        from runtime_harness.write_tools import execute_write

        return execute_write("restart_deployment", {"name": name}, namespace)

    @mcp.tool(name="delete_pod")
    def delete_pod(name: str, namespace: Optional[str] = None) -> str:
        """Delete a pod (controller recreates it). Write tool — harness-gated."""
        refused = _guard_writes()
        if refused:
            return refused
        from runtime_harness.write_tools import execute_write

        return execute_write("delete_pod", {"name": name}, namespace)

    @mcp.tool(name="patch_deployment")
    def patch_deployment(
        name: str,
        set_env: Optional[dict[str, str]] = None,
        remove_env: Optional[list[str]] = None,
        namespace: Optional[str] = None,
    ) -> str:
        """Patch a deployment's container env (read-modify-write, rolling update).

        Guardrails (Oct-26 Run B, validated): G1 set_env only overrides
        existing vars; G2 no-op patches return Error (no exec credit);
        G3 409-Conflict retried with backoff.

        Write tool — harness-gated.
        """
        refused = _guard_writes()
        if refused:
            return refused
        from runtime_harness.write_tools import execute_write

        return execute_write(
            "patch_deployment",
            {"name": name, "set_env": set_env or {}, "remove_env": remove_env or []},
            namespace,
        )

    return mcp


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="aiops-runtime-mcp")
    parser.add_argument("--transport", choices=("stdio", "sse", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args(argv)

    mcp = build_mcp()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
