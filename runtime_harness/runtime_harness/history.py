"""Cycle history — persisted record for MTTD/MTTR trends.

Purpose
-------
Keep the outcome of every completed cycle (timestamps, T0 source, MTTD/MTTR,
detection/RCA/remediation results) so the UI can render trends. Persists to
``data/cycles.json`` so trends survive restarts.

How it connects
---------------
- ``orchestrator`` appends one record per cycle.
- ``GET /trends`` serves the series; the UI charts it.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "cycles.json"
_MAX = 500


class CycleHistory:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else _DEFAULT_PATH
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._records = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                self._records = []

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._records[-_MAX:], indent=2))

    def append(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._records.append(record)
            self._records = self._records[-_MAX:]
            self._persist()

    def trends(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records[-limit:])

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


_history: Optional[CycleHistory] = None


def get_history() -> CycleHistory:
    global _history
    if _history is None:
        _history = CycleHistory()
    return _history


def new_cycle_record(target: str, t0_source: str) -> dict[str, Any]:
    """Skeleton record the orchestrator fills in as the cycle progresses."""
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "t0_source": t0_source,
        "detected": False,
        "rca_confirmed": False,
        "rca_component": None,
        "fault": None,
        "remediation": None,
        "mttd_s": None,
        "mttr_s": None,
        "mttr_mode": None,
        "health": None,
        "stage": None,
        "error": None,
    }


__all__ = ["CycleHistory", "get_history", "new_cycle_record"]
