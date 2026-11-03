"""Tier 3 vocabularies. Separate from Tier 1 and Tier 2 on purpose (design D8)."""
from __future__ import annotations

# The LF Edge harness's ten Kubernetes fault types, spelled as in their
# code/harness/run_harness.py (inject_k8s_fault).
FAULT_TYPES = (
    "scale_zero",
    "kill_pod",
    "memory_limit",
    "network_partition",
    "readiness_probe_fail",
    "config_corruption",
    "dependency_removal",
    "replica_overload",
    "node_taint",
    "pvc_full",
)

DIAGNOSES = (
    "none",
    "service_unavailable",
    "pod_restart",
    "oom_killed",
    "network_blocked",
    "readiness_failure",
    "misconfiguration",
    "node_eviction",
    "disk_full",
    "overload",
    "application_failure",
    "unknown",
)

ACTIONS = (
    "no_action",
    "investigate",
    "restart_service",
    "scale_up",
    "rollback_change",
    "escalate",
)

# kind node names (cluster "jev", tier3/testbed/kind.yaml).
CONTROL_PLANE = "jev-control-plane"
OBSERVABILITY_NODE = "jev-worker"
APP_NODES = ("jev-worker2", "jev-worker3")
NODES = (CONTROL_PLANE, OBSERVABILITY_NODE, *APP_NODES)

# Demo application and data services deployed by chart 0.42.4 with
# tier3/testbed/values/demo.yaml, each a Deployment of the same name (checked with
# `helm template` on 2026-10-14). Not Tier 2's list: Kafka, accounting and
# fraud-detection are not deployed, and the chart's Postgres is astronomy-db.
SERVICES = (
    "ad",
    "astronomy-db",
    "cart",
    "checkout",
    "currency",
    "email",
    "frontend",
    "frontend-proxy",
    "image-provider",
    "payment",
    "product-catalog",
    "quote",
    "recommendation",
    "shipping",
    "valkey-cart",
)

NO_COMPONENT = "none"
# Components the engine may name as affected. The same list in every run.
COMPONENTS = (*SERVICES, *NODES)
