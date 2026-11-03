from tests.tier3.helpers import FakeKube, SeqEngine, ok_decision
from tests.tier4.test_runner import CFG, make
from tier3.scenarios import get_scenario
from tier4.recovery import RECOVERED_BY_ACTION
from tier4.report import export_lfedge, format_scorecard, scorecard_data
from tier4.runner import run_experiment


def records(tmp_path):
    out = []
    for comp in ("cart", "frontend"):
        deps, _, _ = make(tmp_path, SeqEngine(ok_decision(recommended_action="scale_up", affected_component=comp)),
                          kube=FakeKube())
        out.append(run_experiment(get_scenario("K001"), deps, CFG, "r", None))
    return out


def test_scorecard_counts_and_mttr(tmp_path):
    recs = records(tmp_path)
    rem = scorecard_data(recs)["overall"]["remediation"]
    assert rem["executed"] == 2 and rem["by_action"] == 1 and rem["not_recovered"] == 1
    assert rem["mttr"]["n"] == 2 and rem["mttr_verified"]["n"] == 1
    text = format_scorecard(recs)
    assert "## Remediation" in text and "not measured" not in text


def test_export_fills_measured_fields(tmp_path):
    recs = records(tmp_path)
    rows = export_lfedge(recs)
    assert rows[0]["remediation_executed"] is True and rows[0]["recovery_verified"] is True
    assert rows[0]["jev_recovery_outcome"] == RECOVERED_BY_ACTION
    assert rows[1]["recovery_verified"] is False and rows[1]["mttr_verified_seconds"] is None
