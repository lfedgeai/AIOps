"""Assemble Tier 3 collector output for one window: Tier 2's service telemetry
(Prometheus, Jaeger, OpenSearch through the Tier 3 NodePorts) plus Kubernetes state."""
from __future__ import annotations

from tier2.collect import collect as collect_services
from tier2.observation import Telemetry
from tier3.collectors.kube import deployment_rows, event_rows, node_rows, pod_rows
from tier3.observation import KubeState

# Telemetry sources on the Tier 3 cluster that are infrastructure, on top of Tier 2's
# EXCLUDED_SERVICES: the collector's own logs (otelcol-contrib), Jaeger's self-traces
# and the OpAMP server (seen on chart 0.42.4, 2026-10-14).
INFRA_SERVICES = frozenset({"otelcol-contrib", "otel-collector", "jaeger", "opamp-server"})


def without_infra(t: Telemetry) -> Telemetry:
    keep = lambda s: s not in INFRA_SERVICES
    t.services = {k: v for k, v in t.services.items() if keep(k)}
    t.baseline_services = {k: v for k, v in t.baseline_services.items() if keep(k)}
    t.edges = [e for e in t.edges if keep(e.get("from")) and keep(e.get("to"))]
    t.logs = [l for l in t.logs if keep(l.get("service"))]
    for tr in t.traces:
        tr["path"] = [s for s in tr.get("path", []) if keep(s)]
        tr["failed_spans"] = [f for f in tr.get("failed_spans", []) if keep(f.get("service"))]
    t.traces = [tr for tr in t.traces if tr["failed_spans"]]
    return t


def kube_state(kube, start_s: float | None = None, end_s: float | None = None) -> KubeState:
    """Current deployments, pods and nodes; Warning events in [start_s, end_s)."""
    pods = pod_rows(kube.list_pods())
    events = event_rows(kube.list_warning_events(), start_s, end_s) if start_s is not None else []
    return KubeState(deployments=deployment_rows(kube.list_deployments()), pods=pods,
                     nodes=node_rows(kube.list_nodes(), pods), events=events)


def collect(prom, jaeger, logs, kube, *, window_end: float, window_s: int, baseline_end: float,
            baseline_s: int):
    telemetry = collect_services(prom, jaeger, logs, window_end=window_end, window_s=window_s,
                                 baseline_end=baseline_end, baseline_s=baseline_s)
    return without_infra(telemetry), kube_state(kube, window_end - window_s, window_end)
