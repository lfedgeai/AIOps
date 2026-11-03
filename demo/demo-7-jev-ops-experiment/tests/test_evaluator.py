import pytest
from tests.helpers import ok_decision
from evaluation.evaluator import aggregate, compare, format_summary, make_record
from jev.client import Decision
from scenarios.definitions import Expected

EXP = Expected("high", "investigate", "no", "resource_exhaustion")


def rec(scenario, decision, run=1):
    return make_record(run, scenario, run, {}, EXP, decision)


def test_full_match():
    m = compare(EXP, ok_decision("high", "investigate", "no", probable_cause="resource_exhaustion"))
    assert m == {"severity": True, "action": True, "human_review": True,
                 "probable_cause": True, "overall": True}


def test_wrong_cause_does_not_fail_the_decision():
    m = compare(EXP, ok_decision("high", "investigate", "no", probable_cause="network"))
    assert m["overall"] is True and m["probable_cause"] is False


def test_cause_accuracy_reported_separately():
    good = ok_decision("high", "investigate", "no", probable_cause="resource_exhaustion")
    wrong_cause = ok_decision("high", "investigate", "no", probable_cause="unknown")
    s = aggregate([rec("a", good, 1), rec("a", wrong_cause, 2)])
    assert s["correct"] == 2
    assert s["by_field"]["probable_cause"] == {"correct": 1, "total": 2}
    assert "not in PASS/FAIL" in format_summary(s)


def test_old_records_without_cause_still_compare():
    old = {"severity": "high", "action": "investigate", "human_review": "no"}
    m = compare(old, ok_decision("high", "investigate", "no"))
    assert m["overall"] is True and m["probable_cause"] is False


def test_partial_match():
    m = compare(EXP, ok_decision("high", "investigate", "yes"))
    assert m["severity"] and m["action"] and not m["human_review"] and not m["overall"]


def test_invalid_and_error_are_never_correct():
    for st in ("invalid", "error"):
        assert compare(EXP, Decision(status=st))["overall"] is False


def test_counts_and_accuracy():
    good = ok_decision("high", "investigate", "no")
    bad = ok_decision("normal", "observe", "no")
    s = aggregate([rec("a", good, 1), rec("a", good, 2), rec("b", good, 3), rec("b", bad, 4)])
    assert (s["runs"], s["correct"], s["incorrect"]) == (4, 3, 1)
    assert s["accuracy"] == 0.75
    assert s["by_scenario"]["a"] == {"correct": 2, "total": 2, "errored": 0}
    assert s["by_scenario"]["b"]["correct"] == 1 and s["by_scenario"]["b"]["total"] == 2
    assert s["by_field"]["human_review"] == {"correct": 4, "total": 4}
    assert s["by_field"]["severity"] == {"correct": 3, "total": 4}


def test_confidence_split():
    good = ok_decision("high", "investigate", "no", confidence=0.9)
    bad = ok_decision("normal", "observe", "no", confidence=0.5)
    c = aggregate([rec("a", good), rec("a", bad, 2)])["confidence"]
    assert c["mean_correct"] == 0.9 and c["mean_incorrect"] == 0.5 and abs(c["mean"] - 0.7) < 1e-9


def test_no_confidence_is_none_not_zero():
    d = ok_decision("high", "investigate", "no", confidence=None)
    c = aggregate([rec("a", d)])["confidence"]
    assert c["mean"] is None
    assert "n/a" in format_summary(aggregate([rec("a", d)]))


def test_empty_input():
    s = aggregate([])
    assert s["runs"] == 0 and s["accuracy"] is None
    format_summary(s)


def test_invalid_counts_incorrect_error_excluded():
    good = ok_decision("high", "investigate", "no")
    s = aggregate([rec("a", good, 1), rec("a", Decision(status="invalid"), 2),
                   rec("a", Decision(status="error"), 3)])
    assert s["runs"] == 3 and s["scored"] == 2 and s["invalid"] == 1 and s["errored"] == 1
    assert s["correct"] == 1 and s["incorrect"] == 1 and s["accuracy"] == 0.5
    assert s["by_scenario"]["a"]["errored"] == 1


def test_latency_stats():
    ds = [ok_decision("high", "investigate", "no", latency=x) for x in (100, 200, 300)]
    lat = aggregate([rec("a", d, i) for i, d in enumerate(ds, 1)])["latency_ms"]
    assert lat["mean"] == 200 and lat["p50"] == 200 and lat["max"] == 300


def test_summary_layout():
    out = format_summary(aggregate([rec("a", ok_decision("high", "investigate", "no"))]))
    assert "Overall" in out and "By scenario" in out and "a " in out


def test_telemetry_aggregation():
    d1 = ok_decision("high", "investigate", "no", input_tokens=300, output_tokens=20,
                     cost_usd=0.001, server_elapsed_ms=30, latency=100)
    d2 = ok_decision("high", "investigate", "no", input_tokens=500, output_tokens=30,
                     cost_usd=0.003, server_elapsed_ms=50, latency=200, attempts=2)
    t = aggregate([rec("a", d1, 1), rec("b", d2, 2)])["telemetry"]
    assert t["input_tokens"]["total"] == 800 and t["input_tokens"]["mean"] == 400
    assert t["output_tokens"]["total"] == 50
    assert t["cost_usd"]["total"] == pytest.approx(0.004) and t["cost_usd"]["runs_with_cost"] == 2
    assert t["overhead_ms"]["mean"] == 110  # (70 + 150) / 2
    assert t["attempts"] == 3 and t["runs_retried"] == 1
    assert t["http_status"] == {"200": 2}
    assert t["by_scenario"]["b"]["mean_input_tokens"] == 500


def test_telemetry_without_usage_is_na():
    s = aggregate([rec("a", ok_decision("high", "investigate", "no"))])
    assert s["telemetry"]["input_tokens"]["total"] is None
    assert s["telemetry"]["cost_usd"]["total"] is None
    text = format_summary(s)
    assert "Telemetry" in text and "Cost:          n/a" in text
