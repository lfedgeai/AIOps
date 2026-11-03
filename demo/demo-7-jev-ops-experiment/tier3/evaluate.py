"""Independent evaluation of a Decision against hidden ground truth.

Pure functions, Tier 2 rules on Tier 3 fields. Runs after the decision is stored.
Never receives the observation builder, the engine or the client. No composite score.
"""
from __future__ import annotations

from datetime import datetime

from tier3.engine import STATUS_OK, Decision
from tier3.scenarios import GroundTruth

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"
DIMENSIONS = ("detection", "localization", "diagnosis", "action")


def _verdict(value, ok: bool) -> str:
    return UNKNOWN if value is None else (PASS if ok else FAIL)


def evaluate(decision: Decision, truth: GroundTruth) -> dict[str, str]:
    """Each dimension is scored on its own. UNKNOWN only if the decision is not ok
    or the field is missing."""
    if decision is None or decision.status != STATUS_OK:
        return {d: UNKNOWN for d in DIMENSIONS}
    return {
        "detection": _verdict(decision.incident_detected,
                              decision.incident_detected == truth.incident),
        "localization": _verdict(decision.affected_component,
                                 decision.affected_component == truth.affected_component),
        "diagnosis": _verdict(decision.diagnosis, decision.diagnosis == truth.diagnosis),
        "action": _verdict(decision.recommended_action,
                           decision.recommended_action in truth.acceptable_actions),
    }


def false_positive_polls(polls: list[dict], truth: GroundTruth) -> dict | None:
    """For a run that expects no incident: how many answered polls reported one.
    None when not applicable. A run is a false-positive run if any poll does."""
    if truth.incident:
        return None
    answered = [p for p in polls if p["decision"]["status"] == STATUS_OK
                and p["decision"]["incident_detected"] is not None]
    fp = sum(1 for p in answered if p["decision"]["incident_detected"])
    return {"false_positive_polls": fp, "answered_polls": len(answered), "polls": len(polls),
            "false_positive_run": (fp > 0) if answered else None}


def seconds_between(a: str | None, b: str | None) -> float | None:
    if a is None or b is None:
        return None
    return round((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds(), 3)
