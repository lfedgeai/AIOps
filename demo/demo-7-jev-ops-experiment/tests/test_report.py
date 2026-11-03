import json

import report
import run
from jev.client import Decision
from scenarios.definitions import get_scenario, scenario_names
from tests.helpers import FakeClient, ok_decision


def make_results(tmp_path):
    ds = [ok_decision(**get_scenario(s).expected.to_dict()) for s in scenario_names()]
    ds[1] = ok_decision("normal", "observe", "no", confidence=0.4)  # one wrong answer
    ds[2] = Decision(status="error", message="boom")
    run.run_experiment(FakeClient(ds), None, 5, 100, tmp_path, out=lambda *_: None)
    return report.latest(tmp_path)


def test_report_is_computed_from_records(tmp_path):
    path = make_results(tmp_path)
    text = report.render(report.load(path), path.name)
    assert "Overall" in text and "What Jev chose, by scenario" in text
    assert "### degraded (1 runs)" in text
    assert "normal x1" in text          # what Jev actually returned for the miss
    assert "## Misses (2)" in text      # one wrong + one errored
    assert "error" in text


def test_main_writes_markdown_file(tmp_path, capsys):
    path = make_results(tmp_path)
    assert report.main([str(path)]) == 0
    assert path.with_name(path.name.replace(".jsonl", "-report.md")).exists()


def test_no_results_is_a_clear_error(tmp_path):
    import pytest
    with pytest.raises(SystemExit, match="Run run.py first"):
        report.latest(tmp_path)


def test_report_has_telemetry_table(tmp_path):
    path = make_results(tmp_path)
    text = report.render(report.load(path), path.name)
    assert "## Telemetry by scenario" in text and "Tokens in" in text
