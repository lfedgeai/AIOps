"""Harness-side checks: has the fault become visible, has the system recovered.
Results go to the harness and result file only, never to the engine."""
from __future__ import annotations

from tier2.observation import ServiceStats

ERROR_RATE_DELTA = 0.02  # absolute increase in error ratio
LATENCY_FACTOR = 2.0  # p95 at least this many times baseline ...
LATENCY_DELTA_MS = 200.0  # ... and at least this much slower


def deviates(cur: ServiceStats | None, base: ServiceStats | None) -> bool:
    if cur is None:
        return False
    base = base or ServiceStats()
    if (cur.error_rate or 0.0) - (base.error_rate or 0.0) >= ERROR_RATE_DELTA:
        return True
    if cur.p95_ms is not None and base.p95_ms is not None:
        return cur.p95_ms >= base.p95_ms * LATENCY_FACTOR and cur.p95_ms - base.p95_ms >= LATENCY_DELTA_MS
    return False


def any_deviation(cur: dict[str, ServiceStats], base: dict[str, ServiceStats]) -> list[str]:
    return sorted(s for s in cur if deviates(cur[s], base.get(s)))
