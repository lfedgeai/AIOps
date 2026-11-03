"""Tier 2 vocabularies. Separate from Tier 1 (scenarios.definitions) on purpose."""
from __future__ import annotations

DIAGNOSES = (
    "application_failure",
    "latency_degradation",
    "dependency_unreachable",
    "database_contention",
    "resource_exhaustion",
    "unknown",
    "none",
)

ACTIONS = (
    "investigate",
    "restart_service",
    "rollback_change",
    "scale_up",
    "escalate",
    "no_action",
)

# Services Jev may name as affected. The same list is offered in every run.
# Application services plus the data/messaging stores the demo depends on.
SERVICES = (
    "accounting",
    "ad",
    "cart",
    "checkout",
    "currency",
    "email",
    "fraud-detection",
    "frontend",
    "frontend-proxy",
    "image-provider",
    "kafka",
    "payment",
    "postgresql",
    "product-catalog",
    "quote",
    "recommendation",
    "shipping",
    "valkey-cart",
)

# Answer for "no service is affected" (healthy control).
NO_SERVICE = "none"
