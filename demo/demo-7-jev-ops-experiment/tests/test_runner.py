import json

import pytest
from tests.helpers import FakeClient, ok_decision

import run
from generator.server_state import generate
from jev.client import Decision
from scenarios.definitions import get_scenario, scenario_names


def perfect_client(n):
    return FakeClient([ok_decision(**get_scenario(s).expected.to_dict()) for s in scenario_names()] * n)


def test_plan_is_deterministic_and_balanced():
    total = len(scenario_names()) * 20
    a = run.plan_runs(None, total, 1000)
    assert a == run.plan_runs(None, total, 1000)
    counts = {n: sum(1 for s, _ in a if s == n) for n in scenario_names()}
    assert set(counts.values()) == {20}
    assert [seed for _, seed in a] == list(range(1000, 1000 + total))


def test_named_and_random_plans():
    assert {s for s, _ in run.plan_runs("critical", 5, 1)} == {"critical"}
    assert run.plan_runs("random", 20, 5) == run.plan_runs("random", 20, 5)
    assert len(run.plan_runs(None, 1, 5)) == 1


def test_unknown_scenario_exits_nonzero():
    with pytest.raises(SystemExit) as e:
        run.parse_args(["--scenario", "bogus"])
    assert e.value.code != 0


def test_batch_writes_files_and_summary_matches(tmp_path):
    fake = perfect_client(1)
    lines = []
    records, summary = run.run_experiment(fake, None, 10, 7, tmp_path, out=lines.append)
    assert len(records) == 10 and len(fake.calls) == 10
    jsonl = next(tmp_path.glob("*.jsonl"))
    rows = [json.loads(x) for x in jsonl.read_text().splitlines()]
    assert len(rows) == 10
    saved = json.loads(next(tmp_path.glob("*-summary.json")).read_text())["summary"]
    assert saved == summary
    assert summary["runs"] == 10


def test_replay_from_batch_record(tmp_path):
    fake = perfect_client(1)
    records, _ = run.run_experiment(fake, None, 6, 50, tmp_path, out=lambda *_: None)
    r = records[3]
    assert generate(r["scenario"], r["seed"]).to_dict() == r["observations"]


def test_error_does_not_stop_batch(tmp_path):
    fake = FakeClient([Decision(status="error", message="boom"), ok_decision()])
    records, summary = run.run_experiment(fake, "healthy", 3, 1, tmp_path, out=lambda *_: None)
    assert len(records) == 3 and summary["errored"] == 1


def test_single_run_output_pass_and_fail(tmp_path):
    exp = get_scenario("degraded").expected.to_dict()
    lines = []
    run.run_experiment(FakeClient([ok_decision(**exp)]), "degraded", 1, 1234, tmp_path, out=lines.append)
    text = "\n".join(lines)
    assert "Scenario: degraded" in text and "Seed: 1234" in text and "Result: PASS" in text

    lines.clear()
    run.run_experiment(FakeClient([ok_decision("normal", "observe", "no")]), "degraded", 1, 1234, tmp_path, out=lines.append)
    assert "Result: FAIL (severity, action)" in "\n".join(lines)


def test_errored_single_run_is_not_pass(tmp_path):
    lines = []
    run.run_experiment(FakeClient([Decision(status="error", message="down")]), "healthy", 1, 1, tmp_path, out=lines.append)
    text = "\n".join(lines)
    assert "PASS" not in text and "ERROR" in text


def test_main_missing_key_exits_before_any_request(monkeypatch, capsys):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    assert run.main(["--scenario", "healthy"]) == 2
    assert "JEV_API_KEY" in capsys.readouterr().err


def test_recommended_action_is_only_recorded(tmp_path):
    # Runner must never act on a 'restart' recommendation; it only stores it.
    recs, _ = run.run_experiment(FakeClient([ok_decision("critical", "restart", "yes")]), "critical", 1, 1, tmp_path, out=lambda *_: None)
    assert recs[0]["decision"]["action"] == "restart"


def test_no_logs_withholds_logs_but_keeps_metrics(tmp_path):
    fake = FakeClient([ok_decision()])
    recs, _ = run.run_experiment(fake, "hung_worker", 2, 5, tmp_path, out=lambda *_: None,
                                 include_logs=False)
    assert all("logs" not in c for c in fake.calls)
    full = generate("hung_worker", 5).to_dict()
    full.pop("logs")
    assert recs[0]["observations"] == full and recs[0]["include_logs"] is False
    assert list(tmp_path.glob("*-nologs.jsonl"))


def test_logs_sent_by_default_and_shown(tmp_path):
    fake = FakeClient([ok_decision()])
    lines = []
    run.run_experiment(fake, "hung_worker", 1, 5, tmp_path, out=lines.append)
    assert fake.calls[0]["logs"]
    assert "Logs (last 15 min):" in "\n".join(lines) and "deadlock" in "\n".join(lines) \
        or "watchdog" in "\n".join(lines)


def test_no_logs_flag_parses():
    assert run.parse_args(["--no-logs"]).no_logs is True


def test_meta_and_record_telemetry(tmp_path):
    fake = FakeClient([ok_decision(input_tokens=321, output_tokens=21)])
    recs, _ = run.run_experiment(fake, "critical", 2, 3, tmp_path, out=lambda *_: None)
    assert recs[0]["log_lines_sent"] == len(generate("critical", 3).logs)
    assert recs[0]["started_at"]
    meta = json.loads(next(tmp_path.glob("*-summary.json")).read_text())["meta"]
    for k in ("started_at", "finished_at", "wall_time_s", "runs_per_min", "python", "git_commit"):
        assert k in meta
    assert "api_key" not in json.dumps(meta).lower()


def test_no_logs_records_zero_log_lines(tmp_path):
    recs, _ = run.run_experiment(FakeClient([ok_decision()]), "critical", 1, 3, tmp_path,
                                 out=lambda *_: None, include_logs=False)
    assert recs[0]["log_lines_sent"] == 0


def test_single_run_shows_telemetry(tmp_path):
    lines = []
    run.run_experiment(FakeClient([ok_decision(input_tokens=321, output_tokens=21)]), "healthy", 1, 1,
                       tmp_path, out=lines.append)
    text = "\n".join(lines)
    assert "Telemetry" in text and "321 in / 21 out" in text and "Wall time:" in text
