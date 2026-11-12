"""In-memory runtime status snapshot for the UI/API.

Purpose
-------
Hold the latest ``RuntimeStatus`` published by the orchestrator during a cycle.

How it connects
---------------
- ``Orchestrator._publish`` writes here; ``GET /status`` and the status page read it.
- Separate from ``prompts`` / ``stage_models`` stores (those are configuration).
"""

from __future__ import annotations

import threading
from typing import Optional

from runtime_harness.models import RuntimeStatus, utc_now


class StatusStore:
    """Thread-safe in-memory status for the UI/API."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = RuntimeStatus(message="Idle — waiting for a cycle")

    def get(self) -> RuntimeStatus:
        with self._lock:
            return self._status.model_copy(deep=True)

    def set(self, status: RuntimeStatus) -> None:
        with self._lock:
            status.updated_at = utc_now()
            self._status = status

    def patch(self, **kwargs) -> RuntimeStatus:
        with self._lock:
            data = self._status.model_dump()
            data.update(kwargs)
            data["updated_at"] = utc_now()
            self._status = RuntimeStatus(**data)
            return self._status.model_copy(deep=True)


STORE = StatusStore()


def get_store() -> StatusStore:
    return STORE
