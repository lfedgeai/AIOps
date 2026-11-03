"""Assemble Telemetry for one window from the three collectors."""
from __future__ import annotations

from datetime import datetime, timezone

from tier2.collectors.jaeger import Jaeger, parse_traces
from tier2.collectors.opensearch import OpenSearch
from tier2.collectors.prometheus import Prometheus
from tier2.observation import Telemetry

TRACE_SERVICES = 6  # query traces for the N services with the highest error rate


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def collect(prom: Prometheus, jaeger: Jaeger, logs: OpenSearch, *, window_end: float,
            window_s: int, baseline_end: float, baseline_s: int) -> Telemetry:
    cur = prom.service_stats(window_end, window_s)
    base = prom.service_stats(baseline_end, baseline_s)
    start = window_end - window_s
    worst = sorted(cur, key=lambda s: (-(cur[s].error_rate or 0), -(cur[s].p95_ms or 0), s))
    raw = []
    for svc in worst[:TRACE_SERVICES]:
        raw.extend(jaeger.traces(svc, start, window_end, errors_only=True))
    seen, uniq = set(), []
    for t in raw:  # the same trace is returned for several services
        if t["traceID"] not in seen:
            seen.add(t["traceID"])
            uniq.append(t)
    failing, _ = parse_traces(uniq)
    # Dependency edges come from a sample of ordinary traces, not only failing ones.
    sample = jaeger.traces("frontend", start, window_end, errors_only=False, limit=30)
    _, edges = parse_traces(sample + uniq)
    return Telemetry(
        window_start=iso(start), window_end=iso(window_end),
        baseline_start=iso(baseline_end - baseline_s), baseline_end=iso(baseline_end),
        services=cur, baseline_services=base, edges=edges,
        logs=logs.error_logs(start, window_end), traces=failing)
