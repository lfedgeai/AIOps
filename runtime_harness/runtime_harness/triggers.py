"""Trigger layer — scheduler | alertmanager (exclusive) + statistical watcher.

Purpose
-------
Decide WHEN cycles run, from multiple sources behind one abstraction:

  scheduler (interval polling — the heartbeat; UI-settable)
  alertmanager (event-driven — POST /alert receives Alertmanager webhooks;
                mutually exclusive with the scheduler as the PRIMARY mode)
  watcher (cheap statistical sensor on the runtime's own ClickHouse: error
           spans per service + service-silence detection; wakes the LLM cycle
           only on suspicion — independent toggle, cheap enough to keep on)
  manual (POST /cycle — always available)

Every trigger carries a target hint and, when known, a T0 (alert ``startsAt``,
watcher onset timestamp) — upgrading production MTTD from "first telemetry
evidence" to a measured onset. Trust-but-verify: triggers are cues; the
tool-loop detector still investigates independently.

Namespace scope: alerts and watcher operate on ``triggers.namespaces``
(default: otel-demo — where the adopted application lives). Platform alerts
(openshift-monitoring infra) are a phase-2 promotion with their own mapping.

How it connects
---------------
- Mode store mirrors ``remediation.py`` (init-once from YAML, live-switchable
  via ``GET|PUT /triggers`` and the UI's exclusive buttons).
- ``api.py`` wires ``POST /alert`` / ``POST /trigger`` and the watcher thread.
- Cycles fire through ``scheduler.try_run`` (shared lock — never overlapping).
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

TRIGGER_MODES = ("scheduler", "alertmanager")
DEFAULT_NAMESPACES = ["otel-demo"]

# ---------------------------------------------------------------------------
# Mode store (init-once; live changes win)
# ---------------------------------------------------------------------------

_MODE_LOCK = threading.Lock()
_TRIGGER_MODE = "scheduler"
_MODE_INITIALIZED = False


def get_trigger_mode() -> str:
    with _MODE_LOCK:
        return _TRIGGER_MODE


def set_trigger_mode(mode: str) -> str:
    global _TRIGGER_MODE
    if mode not in TRIGGER_MODES:
        raise ValueError(f"trigger mode must be one of {TRIGGER_MODES}")
    with _MODE_LOCK:
        _TRIGGER_MODE = mode
    return _TRIGGER_MODE


def init_trigger_mode_from_config(config: dict[str, Any]) -> str:
    global _TRIGGER_MODE, _MODE_INITIALIZED
    with _MODE_LOCK:
        if _MODE_INITIALIZED:
            return _TRIGGER_MODE
        _TRIGGER_MODE = str((config.get("triggers") or {}).get("mode", "scheduler"))
        _MODE_INITIALIZED = True
        return _TRIGGER_MODE


def get_trigger_namespaces(config: Optional[dict[str, Any]] = None) -> list[str]:
    cfg = config or {}
    ns = (cfg.get("triggers") or {}).get("namespaces") or DEFAULT_NAMESPACES
    return list(ns)


# ---------------------------------------------------------------------------
# Trigger history (UI: recent triggers)
# ---------------------------------------------------------------------------

_HIST_LOCK = threading.Lock()
_HISTORY: list[dict[str, Any]] = []
_HISTORY_MAX = 50


def record_trigger(source: str, target: Optional[str], detail: str = "", **extra: Any) -> None:
    with _HIST_LOCK:
        _HISTORY.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "target": target,
                "detail": detail[:200],
                **extra,
            }
        )
        del _HISTORY[:-_HISTORY_MAX]


def trigger_history(limit: int = 20) -> list[dict[str, Any]]:
    with _HIST_LOCK:
        return list(_HISTORY[-limit:])


# ---------------------------------------------------------------------------
# Alertmanager webhook payload parsing
# ---------------------------------------------------------------------------

# label priority for extracting the workload name
_TARGET_LABELS = ("deployment", "service", "exported_service", "service_name", "pod", "job")

_FINGERPRINT_TTL_S = 600  # Alertmanager re-sends; dedup per fingerprint
_FP_LOCK = threading.Lock()
_SEEN_FINGERPRINTS: dict[str, float] = {}


def _dedup(fingerprint: str) -> bool:
    """True if this fingerprint is new (not seen within the TTL)."""
    now = time.time()
    with _FP_LOCK:
        # expire old entries opportunistically
        for fp in [k for k, v in _SEEN_FINGERPRINTS.items() if now - v > _FINGERPRINT_TTL_S]:
            del _SEEN_FINGERPRINTS[fp]
        if fingerprint in _SEEN_FINGERPRINTS:
            return False
        _SEEN_FINGERPRINTS[fingerprint] = now
        return True


def _target_from_labels(labels: dict[str, str]) -> Optional[str]:
    for key in _TARGET_LABELS:
        val = labels.get(key)
        if val:
            # pod names carry a random suffix: cart-7c7758ddfb-tn9km -> cart
            if key == "pod" and val.count("-") >= 2:
                val = val.split("-")[0]
            return val
    return None


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class AlertEvent:
    def __init__(
        self,
        target: Optional[str],
        namespace: Optional[str],
        alertname: str,
        severity: str,
        starts_at: Optional[datetime],
        fingerprint: str,
    ):
        self.target = target
        self.namespace = namespace
        self.alertname = alertname
        self.severity = severity
        self.starts_at = starts_at
        self.fingerprint = fingerprint


def parse_alertmanager_payload(payload: dict[str, Any]) -> list[AlertEvent]:
    """Extract firing alerts as events (deduped, namespace-filtered, valid targets)."""
    namespaces = set(get_trigger_namespaces())
    events: list[AlertEvent] = []
    for alert in payload.get("alerts") or []:
        if alert.get("status") != "firing":
            continue
        labels = alert.get("labels") or {}
        namespace = labels.get("namespace")
        if namespace and namespaces and namespace not in namespaces:
            continue  # out of trigger scope
        target = _target_from_labels(labels)
        if not target:
            continue  # no workload hint — nothing to focus a cycle on
        fingerprint = alert.get("fingerprint") or f"{labels.get('alertname')}:{target}"
        if not _dedup(fingerprint):
            continue
        events.append(
            AlertEvent(
                target=target,
                namespace=namespace,
                alertname=labels.get("alertname", "unknown"),
                severity=labels.get("severity", ""),
                starts_at=_parse_ts(alert.get("startsAt")),
                fingerprint=fingerprint,
            )
        )
    return events


# ---------------------------------------------------------------------------
# Statistical watcher — cheap sensor on the runtime's own ClickHouse
# ---------------------------------------------------------------------------


class StatisticalWatcher:
    """Background sensor: error-span bursts + service silence.

    Every ``interval_s`` it runs two cheap ClickHouse queries and fires a
    cycle (via callback) when a service breaches thresholds — with a
    per-target cooldown so an ongoing incident doesn't refire every tick.
    """

    def __init__(
        self,
        fire: Callable[[str, datetime, str], Any],
        client_factory: Callable[[], Any],
        interval_s: int = 20,
        error_window_m: int = 5,
        error_threshold: int = 25,
        silence_s: int = 90,
        cooldown_s: int = 300,
        namespaces: Optional[list[str]] = None,
    ):
        self.fire = fire  # fire(target, t0, reason)
        self.client_factory = client_factory
        self.interval_s = interval_s
        self.error_window_m = error_window_m
        self.error_threshold = error_threshold
        self.silence_s = silence_s
        self.cooldown_s = cooldown_s
        self.namespaces = namespaces or DEFAULT_NAMESPACES
        self.enabled = False
        self.checks_run = 0
        self.breaches: list[dict[str, Any]] = []
        self._cooldowns: dict[str, float] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # -- queries -----------------------------------------------------------

    def _error_bursts(self, ch) -> list[tuple[str, int, Optional[datetime]]]:
        out = ch.query(
            f"SELECT ServiceName, count(), min(Timestamp) FROM {ch.database}.otel_traces "
            f"WHERE Timestamp >= now() - INTERVAL {self.error_window_m} MINUTE "
            f"AND StatusCode = 'Error' AND ServiceName != '' "
            f"GROUP BY ServiceName HAVING count() >= {self.error_threshold} "
            f"ORDER BY count() DESC LIMIT 5 FORMAT TabSeparated"
        )
        rows = []
        for ln in (out or "").split("\n"):
            if ln.strip():
                parts = ln.split("\t")
                if len(parts) == 3:
                    ts = _parse_ts(parts[2].replace(" ", "T", 1))
                    rows.append((parts[0], int(parts[1]), ts))
        return rows

    def _silent_services(self, ch) -> list[tuple[str, datetime]]:
        out = ch.query(
            f"SELECT service, max(ts) FROM ("
            f"SELECT ServiceName AS service, max(Timestamp) AS ts FROM {ch.database}.otel_logs "
            f"WHERE Timestamp >= now() - INTERVAL 1 HOUR AND ServiceName != '' GROUP BY ServiceName "
            f"UNION ALL "
            f"SELECT ServiceName, max(Timestamp) FROM {ch.database}.otel_traces "
            f"WHERE Timestamp >= now() - INTERVAL 1 HOUR AND ServiceName != '' GROUP BY ServiceName) "
            f"GROUP BY service ORDER BY max(ts) DESC LIMIT 30 FORMAT TabSeparated"
        )
        rows = []
        now = datetime.now(timezone.utc)
        for ln in (out or "").split("\n"):
            if not ln.strip():
                continue
            parts = ln.split("\t")
            if len(parts) != 2:
                continue
            svc, last = parts[0], parts[1]
            ts = _parse_ts(last.replace(" ", "T", 1))
            if ts is None:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age = (now - ts).total_seconds()
            # silent = stale vs fleet AND it was alive earlier in the window
            if self.silence_s < age < 3600:
                rows.append((svc, ts))
        # fleet freshness: ignore if EVERYTHING is stale (backend down, not an incident)
        if rows and len(rows) >= 25:
            return []
        return rows

    # -- loop ---------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="aiops-watcher", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            if not self.enabled:
                continue
            try:
                self._check()
            except Exception as exc:  # noqa: BLE001 — watcher must never die
                self.breaches.append(
                    {"ts": datetime.now(timezone.utc).isoformat(), "error": str(exc)[:200]}
                )
                self.breaches = self.breaches[-20:]

    def _check(self) -> None:
        ch = self.client_factory()
        self.checks_run += 1
        now = time.time()

        def cooled(svc: str) -> bool:
            last = self._cooldowns.get(svc, 0)
            return now - last > self.cooldown_s

        for svc, count, onset in self._error_bursts(ch):
            if cooled(svc):
                self._cooldowns[svc] = now
                t0 = onset or datetime.now(timezone.utc)
                self.breaches.append(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "service": svc,
                        "reason": f"error-burst ({count} failed spans / {self.error_window_m}m)",
                    }
                )
                self.breaches = self.breaches[-20:]
                self.fire(svc, t0, "watcher-errors")

        for svc, last_ts in self._silent_services(ch):
            if cooled(svc):
                self._cooldowns[svc] = now
                self.breaches.append(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "service": svc,
                        "reason": f"silent since {last_ts.isoformat()}",
                    }
                )
                self.breaches = self.breaches[-20:]
                self.fire(svc, last_ts, "watcher-silence")

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "interval_s": self.interval_s,
            "error_threshold": self.error_threshold,
            "error_window_m": self.error_window_m,
            "silence_s": self.silence_s,
            "cooldown_s": self.cooldown_s,
            "checks_run": self.checks_run,
            "breaches": self.breaches[-10:],
        }


# ---------------------------------------------------------------------------
# Alertmanager poller — pull-based trigger (no firewall/inbound needed)
# ---------------------------------------------------------------------------


class AlertmanagerPoller:
    """Poll Alertmanager's v2 API for firing alerts and fire cycles.

    Push mode (POST /alert webhook) requires the runtime to be reachable
    from the cluster; poll mode works everywhere (port-forward / in-cluster
    URL) at the cost of ``interval_s`` detection latency. Same parsing,
    dedup and trust-but-verify semantics as the webhook.
    """

    def __init__(
        self,
        fire: Callable[[str, Any, str, str], Any],
        url: str = "http://127.0.0.1:9093",
        interval_s: int = 15,
    ):
        self.fire = fire  # fire(target, t0, alertname, severity)
        self.url = url.rstrip("/")
        self.interval_s = interval_s
        self.enabled = False
        self.last_poll_at: Optional[str] = None
        self.last_error: Optional[str] = None
        self.polls_run = 0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def _poll_once(self) -> int:
        import httpx

        resp = httpx.get(f"{self.url}/api/v2/alerts", timeout=10.0,
                         params={"active": "true", "silenced": "false"})
        resp.raise_for_status()
        alerts = resp.json() or []
        # v2 API alerts carry no "status" field (webhook payloads do) — with
        # active=true everything returned is firing; tag them for the parser.
        payload = {"alerts": [dict(a, status="firing") for a in alerts]}
        events = parse_alertmanager_payload(payload)
        for ev in events:
            t0 = None
            if ev.starts_at:
                from runtime_harness.faults import T0

                t0 = T0(ts=ev.starts_at, source="alert")
            self.fire(ev.target, t0, ev.alertname, ev.severity)
        return len(events)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            if not self.enabled:
                continue
            try:
                self._poll_once()
                self.polls_run += 1
                self.last_poll_at = datetime.now(timezone.utc).isoformat()
                self.last_error = None
            except Exception as exc:  # noqa: BLE001 — poller must never die
                self.last_error = str(exc)[:200]

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="aiops-am-poller", daemon=True)
        self._thread.start()

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "url": self.url,
            "interval_s": self.interval_s,
            "polls_run": self.polls_run,
            "last_poll_at": self.last_poll_at,
            "last_error": self.last_error,
        }


__all__ = [
    "TRIGGER_MODES",
    "get_trigger_mode",
    "set_trigger_mode",
    "init_trigger_mode_from_config",
    "get_trigger_namespaces",
    "record_trigger",
    "trigger_history",
    "parse_alertmanager_payload",
    "AlertEvent",
    "StatisticalWatcher",
    "AlertmanagerPoller",
]
