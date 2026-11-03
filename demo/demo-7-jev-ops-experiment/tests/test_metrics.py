"""Metrics are checked against small hand-computed cases. Offline, fake decisions only."""
import pytest

import report
import run
from evaluation import metrics
from evaluation.metrics_report import render_metrics
from jev.client import Decision
from scenarios.definitions import get_scenario, scenario_names
from tests.helpers import FakeClient, ok_decision


def rec(exp, got=None, status="ok", latency=100.0, attempts=1, p=None, cost=None):
    """A minimal run record: exp/got are (severity, action, human_review, cause)."""
    keys = ("severity", "action", "human_review", "probable_cause")
    e = dict(zip(keys, exp))
    d = {"status": status, "latency_ms": latency, "attempts": attempts,
         "human_review_probability": p, "cost_usd": cost, "cost_source": "estimated" if cost else None,
         "input_tokens": 1000, "output_tokens": 100}
    if status == "ok":
        d.update(zip(keys, got))
    match = {k: status == "ok" and d.get(k) == e[k] for k in keys}
    match["overall"] = match["severity"] and match["action"] and match["human_review"]
    return {"scenario": "x", "seed": 1, "expected": e, "decision": d, "status": status, "match": match}


HEALTHY = ("normal", "observe", "no", "none")
HIGH = ("high", "investigate", "yes", "application_bug")
CRIT = ("critical", "escalate", "yes", "resource_exhaustion")


def test_wilson_interval():
    lo, hi = metrics.wilson(8, 10)
    assert lo == pytest.approx(0.490, abs=1e-3) and hi == pytest.approx(0.943, abs=1e-3)
    assert metrics.wilson(0, 0) is None
    assert metrics.wilson(0, 5)[0] == 0.0 and metrics.wilson(5, 5)[1] == 1.0


def test_per_class_precision_recall_f1():
    rs = [rec(HIGH, HIGH), rec(HIGH, HIGH), rec(HIGH, ("degraded",) + HIGH[1:]),
          rec(("degraded",) + HIGH[1:], HIGH)]
    c = metrics.compute(rs)["fields"]["severity"]["classes"]
    # high: tp=2, predicted=3, expected=3 -> P=R=F1=2/3
    assert c["high"]["precision"] == pytest.approx(2 / 3) and c["high"]["recall"] == pytest.approx(2 / 3)
    assert c["high"]["f1"] == pytest.approx(2 / 3)
    # degraded: tp=0 -> F1 0
    assert c["degraded"]["f1"] == 0.0 and c["degraded"]["support"] == 1
    assert "normal" not in c  # never expected, never predicted


def test_macro_and_weighted_f1():
    rs = [rec(HIGH, HIGH)] * 3 + [rec(HEALTHY, HIGH)]
    f = metrics.compute(rs)["fields"]["severity"]
    # high: P=3/4 R=1 F1=6/7; normal: P=n/a R=0 F1=0
    assert f["macro_f1"] == pytest.approx((6 / 7 + 0) / 2)
    assert f["weighted_f1"] == pytest.approx((6 / 7 * 3 + 0 * 1) / 4)


def test_invalid_is_wrong_and_errors_are_excluded():
    rs = [rec(HIGH, HIGH), rec(HIGH, status="invalid"), rec(HIGH, status="error")]
    m = metrics.compute(rs)
    assert m["scored"] == 2 and m["accuracy"] == 0.5
    assert m["fields"]["severity"]["confusion"]["high"] == {"high": 1, metrics.INVALID: 1}
    assert m["reliability"]["availability"] == pytest.approx(2 / 3)
    assert m["reliability"]["invalid_rate"] == 0.5


def test_detection_counts():
    rs = [rec(HIGH, HIGH), rec(CRIT, HEALTHY), rec(HEALTHY, HEALTHY), rec(HEALTHY, HIGH),
          rec(HIGH, status="invalid")]
    d = metrics.compute(rs)["detection"]
    assert (d["tp"], d["fp"], d["fn"], d["tn"]) == (1, 1, 2, 1)
    assert d["recall"] == pytest.approx(1 / 3) and d["false_positive_rate"] == 0.5
    assert d["miss_rate"] == pytest.approx(2 / 3)


def test_operational_risk():
    rs = [rec(HIGH, ("critical", "restart", "yes", "none")), rec(HEALTHY, HEALTHY),
          rec(CRIT, ("high", "investigate", "yes", "none"))]
    m = metrics.compute(rs)
    assert m["severity_bias"]["over"] == 1 and m["severity_bias"]["under"] == 1
    assert m["action_risk"]["unwarranted_disruptive"] == 1 and m["action_risk"]["calm_runs"] == 2
    assert m["action_risk"]["missed_escalation"] == 1 and m["action_risk"]["escalation_runs"] == 1


def test_brier_only_with_probabilities():
    rs = [rec(HIGH, HIGH, p=0.9), rec(HEALTHY, HEALTHY, p=0.2), rec(HIGH, HIGH)]
    b = metrics.compute(rs)["calibration"]
    assert b["n"] == 2 and b["brier"] == pytest.approx(((0.9 - 1) ** 2 + 0.2 ** 2) / 2)
    assert metrics.compute([rec(HIGH, HIGH)])["calibration"]["brier"] is None


def test_latency_percentiles():
    rs = [rec(HIGH, HIGH, latency=float(x)) for x in range(1, 101)]
    lat = metrics.compute(rs)["latency"]
    assert lat["p50"] == pytest.approx(50.5) and lat["p95"] == pytest.approx(95.05)
    assert lat["max"] == 100 and lat["n"] == 100


def test_cost_per_correct_needs_every_run_costed():
    m = metrics.compute([rec(HIGH, HIGH, cost=0.01), rec(HIGH, HEALTHY, cost=0.01)])
    assert m["cost"]["total"] == pytest.approx(0.02) and m["cost"]["per_correct"] == pytest.approx(0.02)
    partial = metrics.compute([rec(HIGH, HIGH, cost=0.01), rec(HIGH, HIGH)])
    assert partial["cost"]["per_correct"] is None
    assert metrics.compute([rec(HIGH, HIGH)])["cost"]["per_run"] is None


def test_render_has_sections_glossary_and_not_applicable():
    rs = [rec(HIGH, HIGH), rec(HEALTHY, HIGH)]
    text = render_metrics([("a.jsonl", rs)], ["jev"], [])
    for heading in ("## Headline", "## Incident detection", "## Latency", "## Glossary",
                    "## Not applicable to this benchmark", "## Appendix: confusion matrices"):
        assert heading in text
    assert "MTTD" in text and "Small sample" in text
    assert "| Latency p95 | 100 ms |" in text


def test_confusion_columns_include_expected_only_classes():
    rs = [rec(("degraded", "investigate", "yes", "unknown"), HIGH)]
    text = render_metrics([("a.jsonl", rs)], ["jev"], [])
    assert "| expected \\ answered | application_bug | unknown |" in text


def test_metrics_main_end_to_end(tmp_path, capsys):
    ds = [ok_decision(**get_scenario(s).expected.to_dict()) for s in scenario_names()]
    ds[0] = Decision(status="error", message="boom")
    run.run_experiment(FakeClient(ds), None, 7, 1, tmp_path, out=lambda *_: None)
    path = next(tmp_path.glob("*runs.jsonl"))
    assert report.main(["--metrics", str(path)]) == 0
    out = next(tmp_path.glob("*-metrics-report.md")).read_text()
    assert "Accuracy (operational decision) | 100% [" in out  # the error is excluded, the rest right
    assert "Availability" in out and "85.7%" in out
