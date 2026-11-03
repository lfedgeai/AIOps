import json

from tests.tier3.helpers import QUIET, ok_decision
from tier3.report import (NOT_MEASURED, export_lfedge, export_reasoning, format_scorecard,
                          load_records, scorecard_data)
from tier3.runner import _decision_dict


def poll(n, d, t="2026-01-01T00:00:47+00:00"):
    return {"n": n, "t_decision": t, "request": {"deployments": [], "pods": [], "events": [],
                                                 "services": []},
            "decision": _decision_dict(d)}


def record(eid="K001", detected_at=47.0, incident=True, decision=None, status="ok", fp=None,
           evaluation=None, reasoning=None, run_id="r1"):
    d = decision or ok_decision(reasoning=reasoning)
    polls = [poll(1, ok_decision(**QUIET)), poll(2, d)]
    return {
        "run_id": run_id, "experiment_id": eid, "status": status, "message": "",
        "fault": {"type": "scale_zero", "target": "cart"} if incident else None,
        "ground_truth": {"incident": incident, "component": "cart" if incident else "none",
                         "diagnosis": "service_unavailable" if incident else "none",
                         "acceptable_actions": ["scale_up"]},
        "polls": polls, "scored_poll": 2, "jev": polls[-1]["decision"], "request": polls[-1]["request"],
        "evaluation": evaluation or {"detection": "PASS", "localization": "PASS",
                                     "diagnosis": "FAIL", "action": "PASS"},
        "false_positive": fp,
        "timing": {"t_inject": "2026-01-01T00:00:00+00:00" if incident else None,
                   "t_detect": "2026-01-01T00:00:47+00:00" if detected_at else None,
                   "mttd_seconds": detected_at},
        "usage": {"engine_calls": 2, "answered_calls": 2, "input_tokens": 2000,
                  "output_tokens": 40, "cost_usd": None},
    }


def control(fp_polls):
    d = ok_decision() if fp_polls else ok_decision(**QUIET)
    return record("K000", detected_at=None, incident=False, decision=d,
                  fp={"false_positive_polls": fp_polls, "answered_polls": 8, "polls": 8,
                      "false_positive_run": fp_polls > 0},
                  evaluation={"detection": "FAIL" if fp_polls else "PASS", "localization": "PASS",
                              "diagnosis": "PASS", "action": "PASS"})


def test_export_of_a_detected_run():
    line = export_lfedge([record()])[0]
    assert line["detected"] is True and line["rca_correct"] is True
    assert line["mttd_seconds"] == 47 and line["remediation_executed"] is None
    assert line["mttr_seconds"] is None and line["recovery_verified"] is None
    assert (line["fault_flag"], line["fault_variant"]) == ("scale_zero", "cart")
    assert line["jev_experiment_id"] == "K001" and line["remediation_correct"] is True
    assert line["suggested_remediations"] == ["scale_up"]


def test_export_unknown_verdict_is_null_not_false():
    line = export_lfedge([record(evaluation={"detection": "UNKNOWN", "localization": "UNKNOWN",
                                             "diagnosis": "UNKNOWN", "action": "UNKNOWN"})])[0]
    assert line["rca_correct"] is None and line["remediation_correct"] is None


def test_scorecard_counts_with_denominators():
    recs = [record(), record(detected_at=60.0, run_id="r2"), control(1), control(0)]
    data = scorecard_data(recs)["overall"]
    assert data["detect"] == (2, 2) and data["rca"] == (2, 2) and data["diagnosis"] == (0, 2)
    assert data["mttd"]["median"] == 53.5 and (data["mttd"]["min"], data["mttd"]["max"]) == (47, 60)
    assert data["false_positive_runs"] == (1, 2) and data["false_positive_polls"] == (1, 16)
    assert data["detection_precision"] == (2, 3)
    assert data["detection_recall"] == (2, 2)
    assert data["detection_f1"] == (4, 5)  # 2TP / (2TP + FP + FN) = 4 / (4 + 1 + 0)
    assert data["detection_confusion"] == {"tp": 2, "fn": 0, "fp": 1, "tn": 1}
    assert data["cost_usd"] is None and data["input_tokens"] == 8000


def test_detection_recall_and_f1_count_missed_fault_runs():
    missed = record(run_id="r3", decision=ok_decision(**QUIET), detected_at=None)
    data = scorecard_data([record(), missed, control(0)])["overall"]
    assert data["detection_recall"] == (1, 2)
    assert data["detection_f1"] == (2, 3)  # 2 / (2 + 0 + 1)
    assert data["detection_confusion"] == {"tp": 1, "fn": 1, "fp": 0, "tn": 1}
    text = format_scorecard([record(), missed, control(0)])
    assert "Detection recall (detected fault runs / fault runs): 1/2" in text
    assert "Detection F1 (2TP / (2TP + FP + FN)): 2/3 (0.67)" in text


def test_scorecard_text_has_unmeasured_rows_and_no_percentages():
    text = format_scorecard([record(), control(1)])
    for row in ("Remediation executed", "MTTR", "Recovery verified"):
        assert f"{row}: {NOT_MEASURED}" in text
    assert "%" not in text
    assert "1/1" in text and "Cost: n/a" in text


def test_reasoning_sheet():
    text = export_reasoning([record(reasoning="cart has 0/1 available"), record(run_id="r2")])
    assert "cart has 0/1 available" in text and "no reasoning returned" in text
    assert "| | |" in text  # empty hand-score columns


def test_load_records_skips_summary_and_ledger(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "K001.json").write_text(json.dumps(record()))
    (d / "summary.json").write_text("{}")
    (tmp_path / ".active-fault.json").write_text("{}")
    assert [r["experiment_id"] for r in load_records(tmp_path)] == ["K001"]
