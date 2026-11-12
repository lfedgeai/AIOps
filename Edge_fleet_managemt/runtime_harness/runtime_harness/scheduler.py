"""Scheduler — continuous cycle execution with UI-settable timing.

Purpose
-------
Run ``Orchestrator.run_cycle`` on an interval so MTTD reflects a continuously
observing runtime, not whoever remembers to call ``POST /cycle``. Timing is
editable live from the UI (``PUT /scheduler``).

How it connects
---------------
- Started as a daemon thread by ``api.py`` on app startup.
- ``GET/PUT /scheduler`` read/update {enabled, interval_s}.
- A threading lock guarantees cycles never overlap.
- ``history.get_history()`` records each cycle for the trends view.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Callable, Optional


class Scheduler:
    def __init__(
        self,
        runner: Callable[[], Any],
        enabled: bool = False,
        interval_s: int = 300,
        gate: Optional[Callable[[], bool]] = None,
    ):
        self._runner = runner
        self._enabled = enabled
        self._interval_s = max(30, int(interval_s))  # floor: 30s
        self._gate = gate  # external condition (e.g. trigger mode == scheduler)
        self._cv = threading.Condition()
        self._cycle_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.last_run_at: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.cycles_run = 0

    # -- state ------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def interval_s(self) -> int:
        return self._interval_s

    def status(self) -> dict[str, Any]:
        with self._cv:
            return {
                "enabled": self._enabled,
                "interval_s": self._interval_s,
                "cycle_running": self._cycle_lock.locked(),
                "cycles_run": self.cycles_run,
                "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
                "last_error": self.last_error,
            }

    def configure(self, enabled: Optional[bool] = None, interval_s: Optional[int] = None) -> None:
        with self._cv:
            if enabled is not None:
                self._enabled = enabled
            if interval_s is not None:
                self._interval_s = max(30, int(interval_s))
            self._cv.notify_all()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="aiops-scheduler", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._enabled or (self._gate and not self._gate()):
                    self._cv.wait()
                interval = self._interval_s
            self._run_once()
            with self._cv:
                self._cv.wait(timeout=interval)

    def try_run(self, fn: Callable[[], Any]) -> bool:
        """Run an externally-triggered cycle under the shared lock.

        Returns False when a cycle is already running (trigger dropped —
        the scheduler heartbeat or the next trigger will cover it).
        """
        if not self._cycle_lock.acquire(blocking=False):
            return False
        try:
            fn()
            return True
        finally:
            self._cycle_lock.release()

    def _run_once(self) -> None:
        if not self._cycle_lock.acquire(blocking=False):
            return  # previous cycle still running — skip this tick
        try:
            self._runner()
            self.cycles_run += 1
            self.last_run_at = datetime.now(timezone.utc)
            self.last_error = None
        except Exception as exc:  # noqa: BLE001 — scheduler must never die
            self.last_error = str(exc)
            self.last_run_at = datetime.now(timezone.utc)
        finally:
            self._cycle_lock.release()


_scheduler: Optional[Scheduler] = None


def get_scheduler() -> Scheduler:
    global _scheduler
    if _scheduler is None:
        raise RuntimeError("scheduler not initialised — call init_scheduler() at startup")
    return _scheduler


def init_scheduler(
    runner: Callable[[], Any],
    enabled: bool,
    interval_s: int,
    gate: Optional[Callable[[], bool]] = None,
) -> Scheduler:
    global _scheduler
    _scheduler = Scheduler(runner, enabled=enabled, interval_s=interval_s, gate=gate)
    _scheduler.start()
    return _scheduler


__all__ = ["Scheduler", "get_scheduler", "init_scheduler"]
