"""Tier 3 Observation: Tier 2's service telemetry plus Kubernetes state (design D9).

The builder takes only collector output (Telemetry, KubeState) and scrub terms. It
never receives a Scenario, FaultSpec or GroundTruth (enforced by tests). It sends
effects, never the change itself: no pod specs, environment variables, resource
limits, NetworkPolicy objects or taint keys.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from tier2.observation import Scrubber, Telemetry
from tier2.observation import build_observation as build_service_sections

# --- fixed shape: the same for every run -----------------------------------
MAX_DEPLOYMENTS = 25
MAX_PODS = 15
MAX_EVENTS = 20
MAX_EVENT_CHARS = 240
BYTE_BUDGET = 60_000  # as Tier 2: about 15k tokens, under the unverified 32k limit

# Always scrubbed, whatever the caller passes: the prefix of every harness object.
ALWAYS_SCRUB = ("jev-fault",)
EXCLUDED_NAMESPACES = frozenset({"kube-system", "kube-public", "kube-node-lease",
                                 "local-path-storage", "calico-system", "tigera-operator",
                                 "calico-apiserver"})
KEYS = ("window", "baseline_window", "deployments", "pods", "nodes", "events", "services",
        "dependencies", "error_logs", "failing_traces")


@dataclass
class KubeState:
    """Collector output from the Kubernetes API (rows from tier3.collectors.kube)."""
    deployments: dict[str, dict] = field(default_factory=dict)  # name -> desired, ready, ...
    pods: list[dict] = field(default_factory=list)
    nodes: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)  # Warning events in the window


def _deployment_rows(cur: dict[str, dict], base: dict[str, dict]) -> list[dict]:
    rows = []
    for name in sorted(set(cur) | set(base)):
        c, b = cur.get(name), base.get(name)
        rows.append({
            "deployment": name,
            "desired": c["desired"] if c else None,
            "ready": c["ready"] if c else None,
            "available": c["available"] if c else None,
            "updated": c["updated"] if c else None,
            "rollout": c["rollout"] if c else None,
            "baseline_desired": b["desired"] if b else None,
            "baseline_available": b["available"] if b else None,
        })

    def key(r):
        drop = (r["baseline_available"] or 0) - (r["available"] or 0)
        return (-abs(drop), r["rollout"] in ("complete", None), r["deployment"])
    return sorted(rows, key=key)


def _pod_rows(cur: list[dict], base: list[dict]) -> list[dict]:
    before = {p["name"]: p for p in base}
    rows = []
    for p in cur:
        if p.get("namespace") in EXCLUDED_NAMESPACES:
            continue
        b = before.get(p["name"])
        delta = p["restarts"] - (b["restarts"] if b else 0)
        new = bool(before) and b is None
        healthy = p["phase"] == "Running" and p["ready"] and not p.get("deleting")
        if healthy and delta <= 0 and not new:
            continue
        rows.append({"pod": p["name"], "service": p["service"], "node": p["node"],
                     "phase": p["phase"], "ready": p["ready"], "terminating": bool(p.get("deleting")),
                     "new_since_baseline": new, "restarts_in_window": max(delta, 0),
                     "waiting_reason": p.get("waiting_reason"),
                     "last_termination_reason": p.get("last_termination_reason")})
    return sorted(rows, key=lambda r: (r["ready"], -r["restarts_in_window"], r["pod"]))


def _node_rows(nodes: list[dict]) -> list[dict]:
    return [{"node": n["name"], "ready": n["ready"], "pressure": list(n["pressure"]),
             "schedulable": n["schedulable"],
             "taint_effects": sorted(t["effect"] for t in n["taints"] if t.get("effect")),
             "pods": n["pods"]} for n in nodes]


def _event_rows(events: list[dict]) -> list[dict]:
    seen: dict[tuple, dict] = {}
    for e in events:
        if e.get("namespace") in EXCLUDED_NAMESPACES:
            continue
        msg = str(e.get("message", ""))[:MAX_EVENT_CHARS]
        key = (e.get("reason"), e.get("object"), re.sub(r"\d+", "#", msg))
        if key in seen:
            seen[key]["count"] += int(e.get("count") or 1)
            seen[key]["last_seen"] = max(seen[key]["last_seen"] or "", e.get("last_seen") or "")
        else:
            seen[key] = {"reason": e.get("reason"), "object": e.get("object"), "message": msg,
                         "count": int(e.get("count") or 1), "last_seen": e.get("last_seen")}
    return sorted(seen.values(), key=lambda r: (-r["count"], str(r["reason"]), str(r["object"])))


def build_observation(telemetry: Telemetry, kube: KubeState, baseline_kube: KubeState | None = None,
                      scrub_terms=()) -> tuple[dict, dict]:
    """Return (observation, trim_notes). trim_notes is for the result record only
    and must never be sent to the engine."""
    terms = (*ALWAYS_SCRUB, *scrub_terms)
    base = baseline_kube or KubeState()
    svc, trims = build_service_sections(telemetry, terms)
    trims = dict(trims)

    deployments = _deployment_rows(kube.deployments, base.deployments)
    pods = _pod_rows(kube.pods, base.pods)
    events = _event_rows(kube.events)
    for name, rows, cap in (("deployments", deployments, MAX_DEPLOYMENTS), ("pods", pods, MAX_PODS),
                            ("events", events, MAX_EVENTS)):
        if len(rows) > cap:
            trims[name] = len(rows) - cap
            del rows[cap:]

    obs = {
        "window": svc["window"],
        "baseline_window": svc["baseline_window"],
        "deployments": deployments,
        "pods": pods,
        "nodes": _node_rows(kube.nodes),
        "events": events,
        "services": svc["services"],
        "dependencies": svc["dependencies"],
        "error_logs": svc["error_logs"],
        "failing_traces": svc["failing_traces"],
    }
    obs = Scrubber(terms).value(obs)

    def size() -> int:
        return len(json.dumps(obs, sort_keys=True).encode())

    for key in ("failing_traces", "error_logs", "events", "dependencies", "pods", "services",
                "deployments"):
        while size() > BYTE_BUDGET and obs[key]:
            obs[key].pop()
            trims[f"budget_{key}"] = trims.get(f"budget_{key}", 0) + 1
    return obs, trims
