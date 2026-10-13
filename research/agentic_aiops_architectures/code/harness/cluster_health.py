"""Cluster baseline checks and harness fault recovery before each experiment."""
from __future__ import annotations

import os
import time
from typing import Any

HARNESS_FAULT_TYPES = (
    "kill_pod",
    "scale_zero",
    "memory_limit",
    "network_partition",
    "readiness_probe_fail",
    "config_corruption",
    "pvc_full",
    "node_taint",
    "replica_overload",
    "dependency_removal",
)

_BAD_DATABASE_HOST = "invalid-host-that-does-not-exist.local"
_HARNESS_FAULT_ENV = "HARNESS_INJECTED_FAULT"


def _load_k8s():
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client


def _deployment_issues(dep: Any, target: str) -> list[str]:
    issues: list[str] = []
    spec_replicas = dep.spec.replicas if dep.spec and dep.spec.replicas is not None else 1
    ready = dep.status.ready_replicas or 0
    available = dep.status.available_replicas or 0

    if spec_replicas < 1:
        issues.append(f"{target}: spec.replicas={spec_replicas} (expected >=1)")
    if ready < max(1, spec_replicas):
        issues.append(f"{target}: ready_replicas={ready}/{spec_replicas}")
    if available < max(1, spec_replicas):
        issues.append(f"{target}: available_replicas={available}/{spec_replicas}")

    containers = (dep.spec.template.spec.containers if dep.spec and dep.spec.template else []) or []
    for c in containers:
        env = {e.name: e.value for e in (c.env or []) if e.name}
        if env.get(_HARNESS_FAULT_ENV):
            issues.append(f"{target}: container {c.name} has {_HARNESS_FAULT_ENV}={env[_HARNESS_FAULT_ENV]!r}")
        if env.get("DATABASE_HOST") == _BAD_DATABASE_HOST:
            issues.append(f"{target}: container {c.name} has corrupted DATABASE_HOST")
        limits = (c.resources.limits if c.resources else {}) or {}
        if limits.get("memory") == "10Mi":
            issues.append(f"{target}: container {c.name} has harness memory_limit=10Mi")
        probe = c.readiness_probe
        if probe and getattr(probe, "exec", None) and probe.exec.command == ["false"]:
            issues.append(f"{target}: container {c.name} has harness readiness_probe_fail")

    return issues


def check_harness_artifacts(namespace: str, target: str) -> list[str]:
    issues: list[str] = []
    client = _load_k8s()
    net = client.NetworkingV1Api()
    policy_name = f"harness-block-{target}"
    try:
        net.read_namespaced_network_policy(policy_name, namespace)
        issues.append(f"network policy {policy_name} still present in {namespace}")
    except Exception:
        pass
    return issues


def check_cluster_baseline(
    target: str | None = None,
    namespace: str | None = None,
) -> list[str]:
    """Return human-readable issues if the target deployment is not in a clean baseline state."""
    target = (target or os.environ.get("HARNESS_FAULT_VARIANT") or "cart").strip()
    namespace = (namespace or os.environ.get("K8S_FAULT_NAMESPACE") or "otel-demo").strip()
    issues: list[str] = []

    try:
        client = _load_k8s()
        apps = client.AppsV1Api()
        dep = apps.read_namespaced_deployment(target, namespace)
        issues.extend(_deployment_issues(dep, target))
        issues.extend(check_harness_artifacts(namespace, target))
    except Exception as e:
        issues.append(f"cannot read deployment {target} in {namespace}: {type(e).__name__}: {e}")

    return issues


def recover_all_harness_faults(target: str, namespace: str) -> None:
    from code.harness.run_harness import recover_k8s_fault

    for fault_type in HARNESS_FAULT_TYPES:
        try:
            recover_k8s_fault(fault_type, target)
        except Exception as e:
            print(
                f"[health-gate] recover {fault_type}/{target} warning: {type(e).__name__}: {e}",
                flush=True,
            )


def wait_deployment_ready(
    target: str,
    namespace: str,
    *,
    timeout_sec: float = 120.0,
    poll_sec: float = 5.0,
) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        issues = check_cluster_baseline(target, namespace)
        blocking = [i for i in issues if "cannot read deployment" not in i]
        if not blocking:
            return True
        time.sleep(poll_sec)
    return False


def check_clickhouse_reachable(ch_http: str | None = None) -> bool:
    import urllib.error
    import urllib.request

    url = (ch_http or os.environ.get("CLICKHOUSE_HTTP") or "http://127.0.0.1:38123").rstrip("/")
    try:
        req = urllib.request.Request(url, data=b"SELECT 1", method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def enforce_cluster_baseline(
    target: str | None = None,
    namespace: str | None = None,
    *,
    auto_recover: bool = True,
    wait_ready: bool = True,
    wait_timeout_sec: float = 120.0,
) -> tuple[bool, list[str]]:
    """
    Ensure harness target is healthy. Optionally recover all known fault types and re-check.
    Returns (ok, remaining_issues).
    """
    target = (target or os.environ.get("HARNESS_FAULT_VARIANT") or "cart").strip()
    namespace = (namespace or os.environ.get("K8S_FAULT_NAMESPACE") or "otel-demo").strip()

    issues = check_cluster_baseline(target, namespace)
    if issues and auto_recover:
        print(
            f"[health-gate] baseline issues on {target}/{namespace}: {issues}; recovering all harness faults",
            flush=True,
        )
        recover_all_harness_faults(target, namespace)
        if wait_ready:
            wait_deployment_ready(target, namespace, timeout_sec=wait_timeout_sec)
        issues = check_cluster_baseline(target, namespace)

    return (len(issues) == 0, issues)
