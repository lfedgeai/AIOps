"""Observation layouts (change observation-facts, design D1-D5).

A layout turns the observation built by `tier3.observation.build_observation` into the
observation sent to the engine. It is applied after the builder, the scrubber and the
byte budget, on live runs (`tier3/runner.py`) and on replay of a stored `request`
(`tier3/replay.py`).

- `raw` returns the observation unchanged: today's request, byte for byte.
- `facts` is a pure function of the observation dict. It adds no information. It does
  arithmetic (ratios, differences, time distances) and puts facts that already exist
  next to the rows they belong to (a deployment's readiness on its service row, each
  caller's and callee's own state on each service row). It never ranks, flags or names a likely
  cause or component, treats every row the same way and keeps the input's row order.

This module must not import `tier3.scenarios` or `tier3.faults`, and reads nothing but
its argument: no clock, cluster, collectors, scenario or record fields.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from tier3.observation import BYTE_BUDGET

DEFAULT_LAYOUT = "raw"

# Every word `facts` writes, apart from numbers and values copied from the input. The
# tests check the output against this set, and that it holds no word that would point
# at a cause ("suspect", "likely", "root", ...).
VOCABULARY = frozenset("""
    available of baseline desired replicas rollout not measured error rate p95 latency ms
    times request s deployment found now facts callees callers calls errors on service
    ts_s_before_window_end last_seen_s_before_window_end length_s ends_s_before_window_end
    layout_budget_exceeded
""".split())


class LayoutError(ValueError):
    """An unknown layout, or a source that cannot be laid out."""


# --- raw -----------------------------------------------------------------------

def _raw(obs: dict) -> tuple[dict, dict]:
    return obs, {}


# --- facts ---------------------------------------------------------------------

def _num(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _f2(x: float) -> str:
    return f"{x:.2f}"


def _f3(x: float) -> str:
    return f"{x:.3f}"


def _ms(x: float) -> str:
    return f"{x:.1f}"


def _parse_time(s) -> datetime | None:
    """ISO 8601 as the collectors write it ('Z' or '+00:00', up to 9 fractional digits),
    parsed on Python 3.10. None when it does not parse."""
    if not isinstance(s, str):
        return None
    m = re.fullmatch(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)?", s.strip())
    if not m:
        return None
    base, frac, tz = m.groups()
    text = base + (f".{frac[:6].ljust(6, '0')}" if frac else "") + ("+00:00" if tz in (None, "Z") else tz)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _seconds_before(end: datetime | None, s) -> int | None:
    t = _parse_time(s)
    return None if end is None or t is None else int(round((end - t).total_seconds()))


def _count(x) -> str:
    n = _num(x)
    return "not measured" if n is None else str(int(round(n)))


def _deployment_facts(row: dict) -> list[str]:
    """Always the same three statements, whatever the state: available against the
    baseline's available, desired replicas against the baseline's, and the rollout."""
    if row.get("desired") is None:
        return ["deployment not found now"]
    avail, desired = _count(row.get("available")), _count(row.get("desired"))
    b_avail, b_desired = row.get("baseline_available"), row.get("baseline_desired")
    if b_avail is None or b_desired is None:
        base_a = base_d = "(baseline not measured)"
    else:
        base_a = f"(baseline available {_count(b_avail)} of {_count(b_desired)})"
        base_d = f"(baseline {_count(b_desired)})"
    rollout = row.get("rollout")
    return [f"available {avail} of {desired} {base_a}",
            f"desired replicas {desired} {base_d}",
            f"rollout {rollout}" if rollout is not None else "rollout not measured"]


def _pair(label: str, c: float | None, b: float | None, fmt: Callable[[float], str], unit: str = "") -> str:
    if c is None:
        return f"{label} not measured"
    if b is None:
        return f"{label} {fmt(c)}{unit} (baseline not measured)"
    return f"{label} {fmt(c)}{unit} (baseline {fmt(b)}{unit})"


def _rate_fact(cur, base) -> str:
    return _pair("request rate", _num(cur), _num(base), _f2, "/s")


def _error_fact(cur, base) -> str:
    return _pair("error rate", _num(cur), _num(base), _f3)


def _latency_fact(cur, base) -> str:
    """Always current and baseline. The ratio is computed from the shown (rounded) values,
    and only when the shown baseline is above zero."""
    c, b = _num(cur), _num(base)
    if c is None or b is None or round(b, 1) <= 0:
        return _pair("p95 latency", c, b, _ms, " ms")
    return f"p95 latency {_ms(c)} ms, {round(c, 1) / round(b, 1):.1f} times baseline ({_ms(b)} ms)"


def _service_facts(row: dict) -> list[str]:
    return [_error_fact(row.get("error_rate"), row.get("baseline_error_rate")),
            _latency_fact(row.get("latency_p95_ms"), row.get("baseline_latency_p95_ms")),
            _rate_fact(row.get("request_rate_per_s"), row.get("baseline_request_rate_per_s"))]


def _peer(name: str, edge: dict, services: dict, deployments: dict) -> dict:
    """A caller or callee of a service row: the edge's own counts, then that peer's own
    availability and error rate. Callers and callees get the same entry."""
    calls, errors = _num(edge.get("calls")), _num(edge.get("errors"))
    facts = [f"errors on {_count(errors)} of {_count(calls)} calls"] if calls is not None and errors is not None else []
    if name in deployments:
        facts.append(_deployment_facts(deployments[name])[0])
    if name in services:
        s = services[name]
        facts.append(_error_fact(s.get("error_rate"), s.get("baseline_error_rate")))
    return {"service": name, "calls": edge.get("calls"), "errors": edge.get("errors"), "facts": facts}


def _facts(obs: dict) -> tuple[dict, dict]:
    o = copy.deepcopy(obs)
    notes: dict = {}
    w = o.get("window") or {}
    w_start, w_end = _parse_time(w.get("start")), _parse_time(w.get("end"))
    b = o.get("baseline_window") or {}
    b_start, b_end = _parse_time(b.get("start")), _parse_time(b.get("end"))
    # Distances need the window end. Without it, the times stay as the source wrote them.
    if w_end is not None:
        if "window" in o:
            o["window"] = {"length_s": int(round((w_end - w_start).total_seconds())) if w_start else None}
        if "baseline_window" in o:
            o["baseline_window"] = {
                "length_s": int(round((b_end - b_start).total_seconds())) if b_start and b_end else None,
                "ends_s_before_window_end": int(round((w_end - b_end).total_seconds())) if b_end else None}
        for key, field in (("error_logs", "ts"), ("events", "last_seen")):
            for r in o.get(key) or []:
                if field in r:
                    r[f"{field}_s_before_window_end"] = _seconds_before(w_end, r.pop(field))

    deployments = {r["deployment"]: r for r in o.get("deployments") or [] if r.get("deployment")}
    services = {r["service"]: r for r in o.get("services") or [] if r.get("service")}
    edges = [e for e in o.get("dependencies") or [] if e.get("from") and e.get("to")]
    for r in o.get("deployments") or []:
        r["facts"] = _deployment_facts(r)
    for r in o.get("services") or []:
        name = r.get("service")
        own = [_deployment_facts(deployments[name])[0]] if name in deployments else []
        r["facts"] = own + _service_facts(r)
        r["callers"] = [_peer(e["from"], e, services, deployments) for e in edges if e["to"] == name]
        r["callees"] = [_peer(e["to"], e, services, deployments) for e in edges if e["from"] == name]

    def size() -> int:
        return len(json.dumps(o, sort_keys=True).encode())

    # Over the budget: drop one field from every row at once, so no row keeps more than
    # another. Source rows are never dropped here (the builder already trimmed those).
    for key, field in (("services", "callers"), ("services", "callees"), ("services", "facts"),
                       ("deployments", "facts")):
        if size() <= BYTE_BUDGET:
            break
        rows = o.get(key) or []
        for r in rows:
            r.pop(field, None)
        notes[f"layout_budget_{key}_{field}"] = len(rows)
    if size() > BYTE_BUDGET:
        notes["layout_budget_exceeded"] = size()
    return o, notes


# --- registry --------------------------------------------------------------------

@dataclass(frozen=True)
class Layout:
    name: str
    version: int
    fn: Callable[[dict], tuple[dict, dict]]
    helpers: tuple[Callable, ...] = ()
    replayable: bool = True

    @property
    def sha256(self) -> str:
        """Over the layout's own source and its helpers' (design D5). Editing any of
        them without bumping `version` is a compare error."""
        parts = [inspect.getsource(f) for f in (self.fn, *self.helpers)]
        if self.name != DEFAULT_LAYOUT:
            parts += [" ".join(sorted(VOCABULARY)), f"BYTE_BUDGET={BYTE_BUDGET}"]
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()

    @property
    def identity(self) -> dict:
        return {"name": self.name, "version": self.version, "sha256": self.sha256}


LAYOUTS: dict[str, Layout] = {
    "raw": Layout("raw", 1, _raw),
    "facts": Layout("facts", 1, _facts,
                    (_num, _f2, _f3, _ms, _count, _parse_time, _seconds_before, _deployment_facts,
                     _pair, _rate_fact, _error_fact, _latency_fact, _service_facts, _peer)),
}


def list_layouts() -> list[str]:
    return list(LAYOUTS)


def get_layout(name: str) -> Layout:
    if name not in LAYOUTS:
        raise LayoutError(f"no observation layout {name!r} (available: {', '.join(LAYOUTS)})")
    return LAYOUTS[name]


def apply_layout(name: str, obs: dict) -> tuple[dict, dict]:
    """(observation to send, notes for the record's trim_notes). Never changes `obs`."""
    return get_layout(name).fn(obs)


def record_layout(record: dict) -> tuple[dict, bool]:
    """(layout identity, assumed) for a run record. Records written before layouts
    existed have no `observation` field and are read as raw version 1, sha256 unknown."""
    o = record.get("observation")
    if isinstance(o, dict) and o.get("name"):
        return {"name": o["name"], "version": o.get("version"), "sha256": o.get("sha256")}, False
    return {"name": DEFAULT_LAYOUT, "version": 1, "sha256": None}, True
