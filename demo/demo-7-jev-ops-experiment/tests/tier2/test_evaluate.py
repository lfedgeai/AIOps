from tests.tier2.helpers import ok_decision
from tier2.engine import Decision
from tier2.evaluate import aggregate, evaluate, is_false_positive, timing
from tier2.scenarios import GroundTruth

TRUTH = GroundTruth(True, "payment", "dependency_unreachable", ("investigate", "rollback_change"))
CONTROL = GroundTruth(False, "none", "none", ("no_action",))


def test_all_pass():
    d = ok_decision(diagnosis="dependency_unreachable")
    assert evaluate(d, TRUTH) == {"detection": "PASS", "localization": "PASS",
                                  "diagnosis": "PASS", "action": "PASS"}


def test_mixed_result_is_not_collapsed():
    d = ok_decision(diagnosis="application_failure", recommended_action="scale_up")
    v = evaluate(d, TRUTH)
    assert v == {"detection": "PASS", "localization": "PASS", "diagnosis": "FAIL", "action": "FAIL"}


def test_wrong_service_fails_localization_only():
    v = evaluate(ok_decision(affected_service="checkout", diagnosis="dependency_unreachable"), TRUTH)
    assert v["localization"] == "FAIL" and v["detection"] == "PASS" and v["diagnosis"] == "PASS"


def test_missed_incident_other_dimensions_still_scored():
    d = ok_decision(incident_detected=False, diagnosis="dependency_unreachable")
    v = evaluate(d, TRUTH)
    assert v["detection"] == "FAIL" and v["localization"] == "PASS" and v["diagnosis"] == "PASS"


def test_acceptable_action_list():
    assert evaluate(ok_decision(recommended_action="rollback_change"), TRUTH)["action"] == "PASS"
    assert evaluate(ok_decision(recommended_action="restart_service"), TRUTH)["action"] == "FAIL"


def test_invalid_and_error_are_unknown():
    for status in ("invalid", "error"):
        v = evaluate(Decision(status=status), TRUTH)
        assert set(v.values()) == {"UNKNOWN"}


def test_missing_field_unknown_only_for_that_field():
    v = evaluate(ok_decision(diagnosis=None), TRUTH)
    assert v["diagnosis"] == "UNKNOWN" and v["detection"] == "PASS"


def test_control_pass_and_false_positive():
    quiet = ok_decision(incident_detected=False, affected_service="none", diagnosis="none",
                        recommended_action="no_action")
    assert set(evaluate(quiet, CONTROL).values()) == {"PASS"}
    assert is_false_positive(quiet, CONTROL) is False
    noisy = ok_decision(incident_detected=True)
    assert evaluate(noisy, CONTROL)["detection"] == "FAIL"
    assert is_false_positive(noisy, CONTROL) is True
    assert is_false_positive(noisy, TRUTH) is None
    assert is_false_positive(Decision(status="error"), CONTROL) is None


def test_no_composite_score_field():
    assert "overall" not in evaluate(ok_decision(), TRUTH)


def rec(eid, evaluation, incident=True, fp=None, status="ok"):
    return {"experiment_id": eid, "status": status, "evaluation": evaluation,
            "ground_truth": {"incident": incident}, "false_positive": fp}


def test_aggregate_counts_and_rates_exclude_unknown():
    P, F, U = "PASS", "FAIL", "UNKNOWN"
    ev = lambda d: {"detection": d, "localization": d, "diagnosis": d, "action": d}
    recs = [rec("F1", ev(P)) for _ in range(4)] + [rec("F1", ev(F)), rec("F1", ev(U))]
    a = aggregate(recs)["overall"]["detection"]
    assert (a["PASS"], a["FAIL"], a["UNKNOWN"]) == (4, 1, 1)
    assert a["pass_rate"] == 4 / 5


def test_aggregate_false_positive_rate_and_other_statuses():
    ev = {k: "PASS" for k in ("detection", "localization", "diagnosis", "action")}
    recs = [rec("F000", ev, incident=False, fp=False) for _ in range(4)]
    recs.append(rec("F000", {**ev, "detection": "FAIL"}, incident=False, fp=True))
    recs.append(rec("F001", ev, status="contaminated"))
    agg = aggregate(recs)
    assert agg["false_positives"] == {"count": 1, "control_runs": 5, "unknown": 0}
    assert agg["runs"] == 6 and agg["runs_other"] == 1


def test_timing_nulls_and_values():
    t = timing(started_at="2026-01-01T00:00:00+00:00", t_inject="2026-01-01T00:00:10+00:00",
               t_manifest=None, t_observation="2026-01-01T00:04:10+00:00",
               t_decision="2026-01-01T00:04:15+00:00", ended_at="2026-01-01T00:05:00+00:00")
    assert t["detection_seconds"] is None  # no measured manifestation -> null, not a guess
    assert t["diagnosis_seconds"] == 245.0 and t["total_seconds"] == 300.0
    t2 = timing(started_at="2026-01-01T00:00:00+00:00", t_inject="2026-01-01T00:00:10+00:00",
                t_manifest="2026-01-01T00:00:40+00:00", t_decision="2026-01-01T00:04:15+00:00",
                ended_at="2026-01-01T00:05:00+00:00")
    assert t2["detection_seconds"] == 215.0
    assert timing(started_at="2026-01-01T00:00:00+00:00")["total_seconds"] is None
