"""AIOpsEngine interface and the normalized Tier 3 Decision.

The runner, evaluator and collectors depend only on this module. Engine-specific
code (Jev) lives in tier3/jev_adapter.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

STATUS_OK = "ok"
STATUS_INVALID = "invalid"
STATUS_ERROR = "error"


@dataclass
class Decision:
    """Normalized engine answer. raw_response is always kept when one exists."""
    status: str  # ok | invalid | error
    incident_detected: bool | None = None
    affected_component: str | None = None
    diagnosis: str | None = None
    recommended_action: str | None = None
    confidence: float | None = None
    reasoning: str | None = None  # stored for hand scoring, never part of a verdict
    raw_response: object | None = None
    message: str = ""
    telemetry: dict = field(default_factory=dict)  # latency, tokens, cost, attempts...

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class AIOpsEngine(Protocol):
    def decide(self, observation: dict) -> Decision:
        """observation is the serialized Observation (plain JSON-able dict)."""
        ...
