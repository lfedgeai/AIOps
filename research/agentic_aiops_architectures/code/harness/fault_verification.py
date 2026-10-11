"""Harness-side verification that an injected fault is resolved on the correct target."""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

from code.harness.evaluation import normalize_component

K8S_FAULT_NAMESPACE = os.environ.get("K8S_FAULT_NAMESPACE", "otel-demo")


def _norm_target(name: str) -> str:
    return normalize_component(name or "")


def _write_targets_expected(entry: dict[str, Any], expected: str) -> bool:
    if entry.get("success") is False:
        return False
    args = entry.get("arguments") or {}
    for key in ("deployment_name", "pod_name", "name"):
        val = args.get(key)
        if not val:
            continue
        norm = _norm_target(str(val))
        if norm == expected or expected in norm or norm in expected:
            return True
    return False


def write_targeted_expected(
    remediation_writes: list[dict[str, Any]] | None,
    expected_variant: str,
) -> bool:
    """True if any write tool targeted the injected service/deployment."""
    expected = _norm_target(expected_variant)
    if not expected or not remediation_writes:
        return False
    return any(_write_targets_expected(entry, expected) for entry in remediation_writes)


def verify_fault_resolved(fault_flag: str, target: str, namespace: str | None = None) -> bool:
    """Check whether the injected fault condition is cleared for target."""
    ns = namespace or K8S_FAULT_NAMESPACE
    try:
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        apps = client.AppsV1Api()
        v1 = client.CoreV1Api()

        if fault_flag == "scale_zero":
            dep = apps.read_namespaced_deployment(target, ns)
            desired = dep.spec.replicas or 0
            ready = dep.status.ready_replicas or 0
            available = dep.status.available_replicas or 0
            return int(desired) >= 1 and (int(ready) >= 1 or int(available) >= 1)

        if fault_flag == "kill_pod":
            pods = v1.list_namespaced_pod(
                ns,
                label_selector=f"app.kubernetes.io/component={target}",
            )
            if not pods.items:
                pods = v1.list_namespaced_pod(ns, label_selector=f"app={target}")
            for p in pods.items:
                if p.status.phase != "Running":
                    return False
                ready = False
                for cond in p.status.conditions or []:
                    if cond.type == "Ready" and cond.status == "True":
                        ready = True
                        break
                if not ready:
                    return False
            return bool(pods.items)

        if fault_flag == "config_corruption":
            dep = apps.read_namespaced_deployment(target, ns)
            for c in dep.spec.template.spec.containers or []:
                for e in c.env or []:
                    if e.name == "HARNESS_INJECTED_FAULT" and e.value == "config_corruption":
                        return False
                    if (
                        e.name == "DATABASE_HOST"
                        and e.value == "invalid-host-that-does-not-exist.local"
                    ):
                        return False
            return (dep.status.ready_replicas or 0) >= 1

        if fault_flag == "memory_limit":
            dep = apps.read_namespaced_deployment(target, ns)
            for c in dep.spec.template.spec.containers or []:
                limits = (c.resources.limits or {}) if c.resources else {}
                if limits.get("memory") == "10Mi":
                    return False
            return (dep.status.ready_replicas or 0) >= 1

        if fault_flag == "network_partition":
            net = client.NetworkingV1Api()
            try:
                net.read_namespaced_network_policy(f"harness-block-{target}", ns)
                return False
            except Exception:
                return True

        if fault_flag == "readiness_probe_fail":
            dep = apps.read_namespaced_deployment(target, ns)
            for c in dep.spec.template.spec.containers or []:
                probe = c.readiness_probe
                cmd = getattr(getattr(probe, "exec", None), "command", None) if probe else None
                if cmd == ["false"]:
                    return False
            return (dep.status.ready_replicas or 0) >= 1

        if fault_flag in ("dependency_removal", "replica_overload"):
            dep = apps.read_namespaced_deployment(target, ns)
            return (dep.status.ready_replicas or 0) >= 1

        return False
    except Exception as e:
        print(f"[harness] verify_fault_resolved({fault_flag}, {target}): {e}", flush=True)
        return False


def poll_recovery_verified(
    fault_flag: str,
    fault_variant: str,
    *,
    remediation_writes: list[dict[str, Any]] | None = None,
    timeout_sec: float = 90.0,
    poll_interval_sec: float = 5.0,
    namespace: str | None = None,
) -> tuple[str | None, bool]:
    """
  Poll until fault is resolved on the injected target AND write tools targeted it.
  Returns (recovery_verified_time ISO8601, verified bool).
  """
    if not write_targeted_expected(remediation_writes, fault_variant):
        print(
            f"[harness] recovery poll skipped: no successful write targeted {fault_variant!r} "
            f"({len(remediation_writes or [])} remediation_writes)",
            flush=True,
        )
        return None, False

    deadline = time.monotonic() + max(1.0, timeout_sec)
    while time.monotonic() < deadline:
        if verify_fault_resolved(fault_flag, fault_variant, namespace):
            ts = datetime.now(timezone.utc).isoformat()
            print(
                f"[harness] recovery verified for {fault_flag}/{fault_variant} at {ts}",
                flush=True,
            )
            return ts, True
        time.sleep(poll_interval_sec)
    return None, False
