"""Normalized Observation: what the AI is allowed to see.

The builder takes only collector output (Telemetry) and scrub terms. It never
receives a Scenario, FaultSpec or GroundTruth (enforced by a signature test).
Caps and the byte budget are named constants, identical in every run.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# --- fixed shape: the same for every run -----------------------------------
MAX_SERVICES = 12
MAX_LOG_LINES_PER_SERVICE = 3
MAX_LOG_SERVICES = 8
MAX_TRACES = 5
MAX_EDGES = 20
MAX_LOG_CHARS = 240
BYTE_BUDGET = 60_000  # Jev's limit is ~32k tokens; stay well under at ~4 bytes/token

REDACTED = "[redacted]"


@dataclass
class ServiceStats:
    request_rate: float | None = None  # requests per second
    error_rate: float | None = None  # 0..1
    p50_ms: float | None = None
    p95_ms: float | None = None


@dataclass
class Telemetry:
    """Raw collector output. Collectors fill this; nothing here knows the scenario."""
    window_start: str
    window_end: str
    baseline_start: str | None = None
    baseline_end: str | None = None
    services: dict[str, ServiceStats] = field(default_factory=dict)
    baseline_services: dict[str, ServiceStats] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)  # {from, to, calls, errors}
    logs: list[dict] = field(default_factory=list)  # {ts, service, level, message}
    traces: list[dict] = field(default_factory=list)  # {trace_id, duration_ms, path, failed_spans}


# --- scrubbing --------------------------------------------------------------
_FLAG_KEY = re.compile(r"feature[_.\-]?flag", re.I)


class Scrubber:
    """Removes fault metadata from any value that is about to be serialized.

    - dict keys containing 'feature_flag' are dropped with their values (this is
      where flag evaluations and variants appear on spans and logs);
    - flag names and other terms are replaced in text, case-insensitively;
    - strings that still mention 'feature_flag' are dropped.
    Variants are NOT scrubbed as text: 'on'/'off' appear in ordinary words.
    """

    def __init__(self, terms):
        ts = sorted({t for t in terms if t}, key=len, reverse=True)
        # Short terms (experiment IDs like F003) need word boundaries so they do not
        # mangle hex trace IDs; flag names are long and match as substrings.
        parts = [rf"(?<![0-9A-Za-z]){re.escape(t)}(?![0-9A-Za-z])" if len(t) <= 5 else re.escape(t)
                 for t in ts]
        self._pat = re.compile("|".join(parts), re.I) if ts else None

    def text(self, s: str) -> str:
        if _FLAG_KEY.search(s):
            return REDACTED
        return self._pat.sub(REDACTED, s) if self._pat else s

    def value(self, v):
        if isinstance(v, str):
            return self.text(v)
        if isinstance(v, dict):
            return {self.text(k) if isinstance(k, str) else k: self.value(x)
                    for k, x in v.items() if not (isinstance(k, str) and _FLAG_KEY.search(k))}
        if isinstance(v, (list, tuple)):
            return [self.value(x) for x in v]
        return v


# --- building ---------------------------------------------------------------
def _round(x, n=4):
    return None if x is None else round(float(x), n)


def _change(cur: float | None, base: float | None) -> float | None:
    if cur is None or base is None:
        return None
    return round(cur - base, 4)


def _service_row(name: str, cur: ServiceStats, base: ServiceStats | None) -> dict:
    base = base or ServiceStats()
    return {
        "service": name,
        "request_rate_per_s": _round(cur.request_rate),
        "error_rate": _round(cur.error_rate),
        "latency_p50_ms": _round(cur.p50_ms, 1),
        "latency_p95_ms": _round(cur.p95_ms, 1),
        "baseline_request_rate_per_s": _round(base.request_rate),
        "baseline_error_rate": _round(base.error_rate),
        "baseline_latency_p95_ms": _round(base.p95_ms, 1),
        "error_rate_change": _change(cur.error_rate, base.error_rate),
        "latency_p95_change_ms": _change(cur.p95_ms, base.p95_ms),
    }


def _rank_key(row: dict):
    # Biggest deviation from baseline first. With no baseline for a service, fall back
    # to its absolute error rate and latency. Name breaks ties so output is stable.
    err = row["error_rate_change"] if row["error_rate_change"] is not None else (row["error_rate"] or 0.0)
    lat = (row["latency_p95_change_ms"] if row["latency_p95_change_ms"] is not None
           else (row["latency_p95_ms"] or 0.0))
    return (-abs(err), -abs(lat), row["service"])


def _dedupe_logs(logs: list[dict]) -> list[dict]:
    seen: dict[tuple, dict] = {}
    for line in logs:
        msg = re.sub(r"\d+", "#", str(line.get("message", "")))[:MAX_LOG_CHARS]
        key = (line.get("service"), line.get("level"), msg)
        if key in seen:
            seen[key]["count"] += 1
        else:
            seen[key] = {"ts": line.get("ts"), "service": line.get("service"),
                         "level": line.get("level"),
                         "message": str(line.get("message", ""))[:MAX_LOG_CHARS], "count": 1}
    return list(seen.values())


def build_observation(telemetry: Telemetry, scrub_terms=()) -> tuple[dict, dict]:
    """Return (observation, trim_notes). trim_notes is for the result record only
    and must never be sent to the engine."""
    scrub = Scrubber(scrub_terms)
    trims: dict[str, int] = {}

    rows = sorted((_service_row(n, s, telemetry.baseline_services.get(n))
                   for n, s in telemetry.services.items()), key=_rank_key)
    if len(rows) > MAX_SERVICES:
        trims["services"] = len(rows) - MAX_SERVICES
        rows = rows[:MAX_SERVICES]

    edges = sorted((e for e in telemetry.edges), key=lambda e: (-(e.get("errors") or 0),
                                                              str(e.get("from")), str(e.get("to"))))
    if len(edges) > MAX_EDGES:
        trims["edges"] = len(edges) - MAX_EDGES
        edges = edges[:MAX_EDGES]

    # Error/warn lines only, deduplicated, ranked by the services' order above.
    order = {r["service"]: i for i, r in enumerate(rows)}
    lines = [l for l in _dedupe_logs(telemetry.logs)
             if str(l.get("level", "")).upper() in {"ERROR", "WARN", "WARNING", "FATAL"}]
    lines.sort(key=lambda l: (order.get(l["service"], 10_000), -l["count"], str(l["message"])))
    per: dict[str, int] = {}
    logs, dropped = [], 0
    for l in lines:
        svc = l["service"]
        if per.get(svc, 0) >= MAX_LOG_LINES_PER_SERVICE or (
                svc not in per and len(per) >= MAX_LOG_SERVICES):
            dropped += 1
            continue
        per[svc] = per.get(svc, 0) + 1
        logs.append(l)
    if dropped:
        trims["log_lines"] = dropped

    traces = telemetry.traces
    if len(traces) > MAX_TRACES:
        trims["traces"] = len(traces) - MAX_TRACES
        traces = traces[:MAX_TRACES]

    obs = {
        "window": {"start": telemetry.window_start, "end": telemetry.window_end},
        "baseline_window": {"start": telemetry.baseline_start, "end": telemetry.baseline_end},
        "services": rows,
        "dependencies": edges,
        "error_logs": logs,
        "failing_traces": traces,
    }
    obs = scrub.value(obs)

    # Byte budget: drop lowest-ranked items from the tail of each list in turn.
    def size() -> int:
        return len(json.dumps(obs, sort_keys=True).encode())

    for key in ("failing_traces", "error_logs", "dependencies", "services"):
        while size() > BYTE_BUDGET and obs[key]:
            obs[key].pop()
            trims[f"budget_{key}"] = trims.get(f"budget_{key}", 0) + 1
    return obs, trims
