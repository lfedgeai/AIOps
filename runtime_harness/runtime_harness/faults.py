"""Fault registry and honest T0 resolution (MTTD/MTTR baseline).

Purpose
-------
Align the runtime's fault model with the evaluation harness
(``scale_zero`` / ``kill_pod`` / ``config_corruption``) and resolve the
T0 timestamp that MTTD/MTTR are measured from:

- **drill mode**: a fault injected by the operator (or chaos tooling) —
  T0 = ``injected_at`` (exact, eval-compatible).
- **production mode**: fault start is unknown — T0 = *first telemetry
  evidence* for the target (earliest error log / failed span in the
  runtime's own ClickHouse). A lower bound, labelled as such.

How it connects
---------------
- ``orchestrator.run_cycle`` accepts an optional ``FaultSpec``; no fault is
  ever fabricated. Healthy system → detector should return detected=false.
- ``faults.first_fault_evidence`` queries the runtime's ClickHouse
  (``telemetry/clickhouse.py``).
- MTTD = detection_time − T0 · MTTR = remediation_time − T0 (mode-labelled).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

FAULT_TYPES = ("scale_zero", "kill_pod", "config_corruption")
LOOKBACK_MINUTES = 30


@dataclass
class FaultSpec:
    """An injected fault (drill mode). Never fabricated by the harness."""

    type: str                      # one of FAULT_TYPES
    target: str                    # workload name, e.g. "cart"
    injected_at: datetime          # T0 (exact)

    def iso(self) -> str:
        return self.injected_at.isoformat()


@dataclass
class T0:
    """Resolved MTTD/MTTR baseline with provenance."""

    ts: datetime
    source: str  # "injected" | "telemetry-evidence" | "cycle-start"

    @property
    def iso(self) -> str:
        return self.ts.isoformat()


def _parse_ts(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def first_fault_evidence(client, target: str) -> Optional[datetime]:
    """Earliest error signal for ``target`` in the lookback window.

    Queries the runtime's own ClickHouse: earliest ERROR-severity log and
    earliest ERROR-status span for the service. Returns None when the
    system looks quiet (expected for healthy cycles).
    """
    from runtime_harness.telemetry.clickhouse import _esc  # local import, no cycle

    since = (datetime.now(timezone.utc) - timedelta(minutes=LOOKBACK_MINUTES)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    candidates: list[datetime] = []
    queries = [
        # earliest error log for the target
        (
            "SELECT min(Timestamp) FROM {db}.otel_logs "
            "WHERE Timestamp >= '{since}' AND ServiceName = '{svc}' "
            "AND (positionCaseInsensitive(SeverityText, 'ERROR') > 0 "
            "     OR positionCaseInsensitive(Body, 'error') > 0) "
            "FORMAT TabSeparated"
        ),
        # earliest failed span for the target
        (
            "SELECT min(Timestamp) FROM {db}.otel_traces "
            "WHERE Timestamp >= '{since}' AND ServiceName = '{svc}' "
            "AND StatusCode = 'Error' "
            "FORMAT TabSeparated"
        ),
    ]
    for q in queries:
        out = client.query(q.format(db=client.database, since=since, svc=_esc(target)))
        if out.startswith("Error:") or not out.strip():
            continue
        ts = _parse_ts(out.strip().replace(" ", "T", 1))
        if ts is None:
            # ClickHouse may return 'YYYY-MM-DD HH:MM:SS.mmm' variants
            try:
                ts = datetime.strptime(out.strip(), "%Y-%m-%d %H:%M:%S.%f").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue
        candidates.append(ts)
    return min(candidates) if candidates else None


def resolve_t0(
    fault: Optional[FaultSpec], client, target: Optional[str], cycle_start: datetime
) -> T0:
    """Injection time when known; telemetry evidence otherwise; cycle start last."""
    if fault is not None:
        return T0(ts=fault.injected_at, source="injected")
    if target:
        ev = first_fault_evidence(client, target)
        if ev is not None:
            return T0(ts=ev, source="telemetry-evidence")
    return T0(ts=cycle_start, source="cycle-start")


__all__ = ["FAULT_TYPES", "FaultSpec", "T0", "first_fault_evidence", "resolve_t0"]
