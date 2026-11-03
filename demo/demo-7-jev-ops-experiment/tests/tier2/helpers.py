"""Shared fakes for Tier 2 tests. Nothing here touches the network or Docker."""
from __future__ import annotations

import copy

from tier2.engine import Decision
from tier2.observation import ServiceStats, Telemetry

FLAGS = {"flags": {
    "cartFailure": {"state": "ENABLED", "defaultVariant": "off",
                    "variants": {"off": 0, "100%": 1, "50%": 0.5}},
    "paymentFailure": {"state": "ENABLED", "defaultVariant": "off",
                       "variants": {"off": 0, "100%": 1}},
    "productCatalogFailure": {
        "state": "ENABLED", "defaultVariant": "off", "variants": {"off": False, "on": True},
        "targeting": {"if": [{"==": [{"var": "product_id"}, "OLJCESPC7Z"]}, "off", "off"]}},
}}


class FakeFlagStore:
    def __init__(self, cfg=None):
        self.cfg = copy.deepcopy(cfg or FLAGS)
        self.writes = 0

    def read(self):
        return copy.deepcopy(self.cfg)

    def write(self, config):
        self.cfg = copy.deepcopy(config)
        self.writes += 1


class LyingFlagStore(FakeFlagStore):
    """Accepts writes but does not apply them, to test the read-back check."""
    def write(self, config):
        self.writes += 1


def ok_decision(**kw) -> Decision:
    base = dict(status="ok", incident_detected=True, affected_service="payment",
                diagnosis="application_failure", recommended_action="investigate",
                confidence=0.8, reasoning=None, raw_response={"answers": {}})
    base.update(kw)
    return Decision(**base)


class FakeEngine:
    def __init__(self, decision=None, exc=None):
        self.decision, self.exc, self.calls = decision or ok_decision(), exc, []

    def decide(self, observation):
        self.calls.append(observation)
        if self.exc:
            raise self.exc
        return self.decision


def contaminated_telemetry(flag="paymentFailure", experiment_id="F003") -> Telemetry:
    """Healthy-looking telemetry that deliberately carries fault metadata."""
    return Telemetry(
        window_start="2026-01-01T00:00:00+00:00", window_end="2026-01-01T00:03:00+00:00",
        baseline_start="2025-12-31T23:57:00+00:00", baseline_end="2026-01-01T00:00:00+00:00",
        services={
            "payment": ServiceStats(0.1, 1.0, 40, 90),
            "checkout": ServiceStats(0.3, 0.6, 50, 3000),
            "frontend": ServiceStats(5.0, 0.05, 20, 300),
        },
        baseline_services={"payment": ServiceStats(0.1, 0.0, 40, 80),
                           "checkout": ServiceStats(0.3, 0.0, 50, 200)},
        edges=[{"from": "checkout", "to": "payment", "calls": 5, "errors": 5},
               {"from": "checkout", "to": "flagd", "calls": 50, "errors": 0}],
        logs=[
            {"ts": "t1", "service": "payment", "level": "ERROR",
             "message": f"Payment request failed. Invalid token. app.loyalty.level=gold {flag} {experiment_id}"},
            {"ts": "t2", "service": "checkout", "level": "ERROR",
             "message": "feature_flag.key=paymentFailure variant=100% evaluated"},
        ],
        traces=[{"trace_id": "abcd1234", "path": ["frontend", "checkout", "payment"],
                 "duration_ms": 3000.0,
                 "failed_spans": [{"service": "payment", "operation": f"charge {flag}",
                                   "error": "feature_flag.variant=100%"}]}],
    )
