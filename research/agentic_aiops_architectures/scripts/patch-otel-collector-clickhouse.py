#!/usr/bin/env python3
"""Patch otel-collector ConfigMap for AIOps harness telemetry.

- Ensure ClickHouse exporter (agentic-aiops) on traces/metrics/logs
- Drop dead/demo exporters that thrash the collector on this SNO:
  OpenSearch (SCC-blocked, never schedules), Jaeger OTLP (scaled to 0),
  Prometheus OTLP HTTP (WAL / connection refused)
- Soften self-telemetry so the collector does not HTTPS-loop to its own :4318
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import yaml

NS = "otel-demo"
CM = "otel-collector"

# Dead exporters that cause permanent export errors / probe timeouts
DROP_FROM_PIPELINES = {
    "logs": {"opensearch"},
    "metrics": {"otlphttp/prometheus"},
    "traces": {"otlp"},  # jaeger-collector (deploy scaled to 0)
}


def main() -> int:
    result = subprocess.run(
        ["oc", "get", "configmap", CM, "-n", NS, "-o", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    cm = json.loads(result.stdout)
    config = yaml.safe_load(cm["data"]["relay"])

    endpoint = os.environ.get(
        "CLICKHOUSE_COLLECTOR_ENDPOINT",
        "tcp://clickhouse.agentic-aiops.svc.cluster.local:9000?dial_timeout=10s",
    )
    config.setdefault("exporters", {})["clickhouse"] = {
        "endpoint": endpoint,
        "database": "otel",
        "create_schema": True,
        "timeout": "10s",
    }

    pipelines = config.setdefault("service", {}).setdefault("pipelines", {})
    for name, drop in DROP_FROM_PIPELINES.items():
        if name not in pipelines:
            continue
        exporters = list(pipelines[name].get("exporters") or [])
        exporters = [e for e in exporters if e not in drop]
        if "clickhouse" not in exporters:
            exporters.append("clickhouse")
        # Keep debug last for local troubleshooting without blocking export
        if "debug" in exporters:
            exporters = [e for e in exporters if e != "debug"] + ["debug"]
        pipelines[name]["exporters"] = exporters

    # Avoid HTTPS self-scrape loop (collector listens HTTP on 4318)
    telemetry = config.setdefault("service", {}).setdefault("telemetry", {})
    metrics = telemetry.setdefault("metrics", {})
    metrics["level"] = "basic"
    metrics.pop("readers", None)

    cm["data"]["relay"] = yaml.dump(config, default_flow_style=False, sort_keys=False)
    meta = cm.setdefault("metadata", {})
    meta.pop("managedFields", None)
    meta.pop("resourceVersion", None)

    path = "/tmp/otel-collector-patched.yaml"
    with open(path, "w") as f:
        json.dump(cm, f, indent=2)

    subprocess.run(["oc", "apply", "-f", path], check=True)
    print(
        "Patched otel-collector: clickhouse kept; "
        f"dropped { {k: sorted(v) for k, v in DROP_FROM_PIPELINES.items()} }; "
        "telemetry metrics → basic (no self OTLP readers)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
