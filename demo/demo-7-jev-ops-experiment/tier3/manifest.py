"""Harness-side checks: has the fault become visible, has the cluster recovered.
Results go to the harness and result file only, never to the engine."""
from __future__ import annotations

from tier2.manifest import any_deviation
from tier2.observation import ServiceStats
from tier3.observation import KubeState


def manifestation(cur_stats: dict[str, ServiceStats], base_stats: dict[str, ServiceStats],
                  cur: KubeState, base: KubeState) -> list[str]:
    """Why the fault counts as visible; empty when it does not yet."""
    why = [f"service {s} deviates" for s in any_deviation(cur_stats, base_stats)]
    for name, b in base.deployments.items():
        c = cur.deployments.get(name)
        if c is None or c["available"] < b["available"]:
            why.append(f"deployment {name} below baseline availability")
        elif c["rollout"] != "complete":
            why.append(f"deployment {name} rollout {c['rollout']}")
    before = {p["name"]: p for p in base.pods}
    for p in cur.pods:
        b = before.get(p["name"])
        if not (p["phase"] == "Running" and p["ready"]) or p.get("deleting"):
            why.append(f"pod {p['name']} not ready")
        elif b and p["restarts"] > b["restarts"]:
            why.append(f"pod {p['name']} restarted")
    if before and set(before) - {p["name"] for p in cur.pods}:
        why.append("pods replaced since baseline")
    bn = {n["name"]: n for n in base.nodes}
    for n in cur.nodes:
        b = bn.get(n["name"])
        if b and (n["ready"] != b["ready"] or n["taints"] != b["taints"]
                  or n["pressure"] != b["pressure"] or n["schedulable"] != b["schedulable"]):
            why.append(f"node {n['name']} changed")
    return why


def recovery_problems(cur: KubeState, base: KubeState, leftovers: list[str],
                      cur_stats: dict[str, ServiceStats], base_stats: dict[str, ServiceStats]) -> list[str]:
    """Empty when the cluster is back at baseline (design D14)."""
    out = list(leftovers)
    for name, b in base.deployments.items():
        c = cur.deployments.get(name)
        if c is None:
            out.append(f"deployment {name} missing")
        elif c["desired"] != b["desired"] or c["available"] < c["desired"] or c["rollout"] != "complete":
            out.append(f"deployment {name} not restored ({c['available']}/{c['desired']} available, "
                       f"rollout {c['rollout']}, baseline desired {b['desired']})")
    for n in cur.nodes:
        if not n["ready"]:
            out.append(f"node {n['name']} not Ready")
        if any(t.get("key") == "jev-fault" for t in n["taints"]):
            out.append(f"node {n['name']} still tainted")
    out += [f"service {s} still deviates" for s in any_deviation(cur_stats, base_stats)]
    return out
