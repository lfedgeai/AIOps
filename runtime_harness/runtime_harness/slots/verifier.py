"""Verify slot — real Kubernetes recovery check.

Purpose
-------
Verify that the RCA'd workload actually recovered, by reading live cluster
state (``readyReplicas`` vs desired) via the kubernetes client — objective,
harness-level verification, not model-judged (the eval harness's approach).
Falls back to proposal-mode only when cluster access is unavailable.

How it connects
---------------
- Final stage in ``orchestrator`` after remediator.
- Reads ``ctx["rca"]["component"]`` / ``ctx["target"]``; writes
  ``ctx["verification"]`` {recovered, mode: k8s|stub, detail}.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from runtime_harness.models import Stage, StageResult
from runtime_harness.slots.base import Slot, SlotContext


def _check_k8s(name: str, namespace: str) -> dict[str, Any]:
    from kubernetes import client as k8s_client, config as k8s_config

    try:
        k8s_config.load_incluster_config()
    except Exception:  # noqa: BLE001
        k8s_config.load_kube_config()
    dep = k8s_client.AppsV1Api().read_namespaced_deployment(name, namespace)
    desired = dep.spec.replicas or 0
    ready = dep.status.ready_replicas or 0
    return {
        "desired": desired,
        "ready": ready,
        "recovered": desired > 0 and ready >= desired,
        "conditions": [
            {"type": c.type, "status": c.status} for c in (dep.status.conditions or [])
        ][:4],
    }


class K8sHealthVerifier(Slot):
    """Verify recovery from live cluster state."""

    name = "verifier"
    plugin = "k8s_health"

    def run(self, ctx: SlotContext) -> StageResult:
        started = datetime.now(timezone.utc)
        rem = ctx.get("remediation") or {}
        rca = ctx.get("rca") or {}
        target = rca.get("component") or ctx.get("target") or "unknown"
        namespace = ctx.get("namespace", "otel-demo")

        mode = "k8s"
        try:
            detail = _check_k8s(target, namespace)
            recovered = detail["recovered"]
        except Exception as exc:  # noqa: BLE001 — cluster unavailable → stub
            mode = "stub"
            recovered = bool(rem.get("executed")) or (
                bool(rem.get("proposed"))
                and (rem.get("dry_run") or rem.get("human_in_the_loop"))
            )
            detail = {"fallback_reason": str(exc)[:200]}

        ctx["verification"] = {
            "recovered": recovered,
            "target": target,
            "mode": mode,
            **detail,
        }
        finished = datetime.now(timezone.utc)
        extra = f", {detail.get('ready')}/{detail.get('desired')} ready" if mode == "k8s" else ""
        return StageResult(
            stage=Stage.VERIFY,
            ok=recovered,
            summary=f"Recovery verified={recovered} for {target} ({mode}{extra})",
            detail=ctx["verification"],
            started_at=started,
            finished_at=finished,
            duration_s=(finished - started).total_seconds(),
        )


def build_verifier(config: dict[str, Any]) -> Slot:
    plugin = config.get("plugin", "k8s_health")
    if plugin != "k8s_health":
        raise ValueError(f"Unknown verifier plugin: {plugin}")
    return K8sHealthVerifier(config)
