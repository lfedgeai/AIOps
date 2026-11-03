"""Independent evaluation of a Decision against hidden ground truth.

Pure functions. Runs after the decision is stored. Never receives the
observation builder, the engine or the client. No composite score.
"""
from __future__ import annotations

from datetime import datetime

from tier2.engine import STATUS_OK, Decision
from tier2.scenarios import GroundTruth
from tier2.vocab import NO_SERVICE

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"
DIMENSIONS = ("detection", "localization", "diagnosis", "action")


def _verdict(value, ok: bool) -> str:
    return UNKNOWN if value is None else (PASS if ok else FAIL)


def evaluate(decision: Decision, truth: GroundTruth) -> dict[str, str]:
    """Each dimension is scored on its own. UNKNOWN only if the decision is not ok
    or the field is missing."""
    if decision.status != STATUS_OK:
        return {d: UNKNOWN for d in DIMENSIONS}
    return {
        "detection": _verdict(decision.incident_detected,
                              decision.incident_detected == truth.incident),
        "localization": _verdict(decision.affected_service,
                                 decision.affected_service == truth.affected_service),
        "diagnosis": _verdict(decision.diagnosis, decision.diagnosis == truth.diagnosis),
        "action": _verdict(decision.recommended_action,
                           decision.recommended_action in truth.acceptable_actions),
    }


def is_false_positive(decision: Decision, truth: GroundTruth) -> bool | None:
    """Only defined for the healthy control. None when not applicable or unknown."""
    if truth.incident:
        return None
    if decision.status != STATUS_OK or decision.incident_detected is None:
        return None
    return decision.incident_detected


def _seconds(a: str | None, b: str | None) -> float | None:
    if a is None or b is None:
        return None
    return round((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds(), 3)


def timing(*, started_at, t_inject=None, t_manifest=None, t_observation=None,
           t_decision=None, ended_at=None) -> dict:
    """ISO-8601 timestamps in, seconds out. Anything not measured stays None.

    detection_seconds needs a measured manifestation: it is the time from the
    fault becoming visible to the decision. Otherwise it is None, not a guess.
    """
    return {
        "started_at": started_at,
        "t_inject": t_inject,
        "t_manifest": t_manifest,
        "t_observation": t_observation,
        "t_decision": t_decision,
        "detection_seconds": _seconds(t_manifest, t_decision),
        "diagnosis_seconds": _seconds(t_inject, t_decision) if t_inject else None,
        "total_seconds": _seconds(started_at, ended_at),
    }


def aggregate(records: list[dict]) -> dict:
    """Per-dimension counts overall and per scenario. UNKNOWN is shown but is not
    in the rate denominator. Also the false-positive rate over control runs."""

    def tally(rows: list[dict]) -> dict:
        out = {}
        for dim in DIMENSIONS:
            c = {PASS: 0, FAIL: 0, UNKNOWN: 0}
            for r in rows:
                c[r["evaluation"][dim]] += 1
            decided = c[PASS] + c[FAIL]
            out[dim] = {**c, "pass_rate": (c[PASS] / decided) if decided else None}
        return out

    rows = [r for r in records if r.get("status") == "ok"]
    by_scn: dict[str, list[dict]] = {}
    for r in rows:
        by_scn.setdefault(r["experiment_id"], []).append(r)
    controls = [r for r in rows if not r["ground_truth"]["incident"]]
    fps = [r for r in controls if r.get("false_positive") is True]
    unknown_fp = [r for r in controls if r.get("false_positive") is None]
    return {
        "runs": len(records),
        "runs_ok": len(rows),
        "runs_other": len(records) - len(rows),
        "overall": tally(rows) if rows else {},
        "by_scenario": {k: tally(v) for k, v in sorted(by_scn.items())},
        "false_positives": {"count": len(fps), "control_runs": len(controls),
                            "unknown": len(unknown_fp)},
    }
