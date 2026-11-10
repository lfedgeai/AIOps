#!/usr/bin/env python3
"""Patch the OTEL Collector ConfigMap to export logs/traces/metrics to ClickHouse.

Adds a ``clickhouse`` exporter to every pipeline and removes the demo exporters
that are known to fail on a minimal OpenShift cluster (OpenSearch, dead Jaeger
OTLP, Prometheus OTLP HTTP). Safe to re-run.

Configuration (env or flags):
  OTEL_DEMO_NAMESPACE          default otel-demo
  OTEL_COLLECTOR_CONFIGMAP     default otel-collector
  CLICKHOUSE_COLLECTOR_ENDPOINT default tcp://clickhouse.<AIOPS_NAMESPACE>.svc.cluster.local:9000
  CLICKHOUSE_DATABASE          default otel

Requires the ``kubernetes`` and ``pyyaml`` Python packages and a valid
kubeconfig or in-cluster service account.
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

DROP_FROM_PIPELINES = {
    # demo exporters that are dead or too noisy on a minimal cluster
    "logs": {"opensearch", "clickhouse/rt"},
    "metrics": {"otlphttp/prometheus", "clickhouse/rt"},
    "traces": {"otlp", "clickhouse/rt"},
}


def _k8s_client():
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client


def main() -> int:
    aiops_ns = os.environ.get("AIOPS_NAMESPACE") or "agentic-aiops"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--namespace", default=os.environ.get("OTEL_DEMO_NAMESPACE") or "otel-demo")
    ap.add_argument("--configmap", default=os.environ.get("OTEL_COLLECTOR_CONFIGMAP") or "otel-collector")
    ap.add_argument(
        "--endpoint",
        default=os.environ.get("CLICKHOUSE_COLLECTOR_ENDPOINT")
        or f"tcp://clickhouse.{aiops_ns}.svc.cluster.local:9000?dial_timeout=10s",
    )
    ap.add_argument("--database", default=os.environ.get("CLICKHOUSE_DATABASE") or "otel")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    client = _k8s_client()
    core = client.CoreV1Api()
    cm = core.read_namespaced_config_map(args.configmap, args.namespace)
    raw = (cm.data or {}).get("relay")
    if not raw:
        print(f"[collector] ConfigMap {args.namespace}/{args.configmap} has no 'relay' key", file=sys.stderr)
        return 1

    cfg = yaml.safe_load(raw) or {}
    cfg.setdefault("exporters", {})["clickhouse"] = {
        "endpoint": args.endpoint,
        "database": args.database,
        "create_schema": True,
        "timeout": "10s",
    }

    pipelines = cfg.setdefault("service", {}).setdefault("pipelines", {})
    for name, drop in DROP_FROM_PIPELINES.items():
        if name not in pipelines:
            continue
        exporters = [e for e in (pipelines[name].get("exporters") or []) if e not in drop]
        if "clickhouse" not in exporters:
            exporters.append("clickhouse")
        if "debug" in exporters:
            exporters = [e for e in exporters if e != "debug"] + ["debug"]
        pipelines[name]["exporters"] = exporters

    telemetry = cfg.setdefault("service", {}).setdefault("telemetry", {})
    metrics = telemetry.setdefault("metrics", {})
    metrics["level"] = "basic"
    metrics.pop("readers", None)

    rendered = yaml.dump(cfg, default_flow_style=False, sort_keys=False)
    if args.dry_run:
        print(rendered)
        return 0

    core.patch_namespaced_config_map(
        args.configmap,
        args.namespace,
        {"data": {"relay": rendered}},
    )
    print(f"[collector] patched {args.namespace}/{args.configmap} -> {args.endpoint} db={args.database}")
    print("[collector] restart the collector: oc rollout restart deploy/otel-collector -n " + args.namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
