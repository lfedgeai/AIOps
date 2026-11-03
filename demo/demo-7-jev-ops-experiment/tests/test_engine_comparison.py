"""Engine selection in run.py and the comparison report. Offline, fake engines only."""
import json

import pytest

import report
import run
from scenarios.definitions import get_scenario, scenario_names
from tests.helpers import FakeClient, ok_decision


def perfect(n=1):
    return [ok_decision(**get_scenario(s).expected.to_dict()) for s in scenario_names()] * n


class FakeLLM(FakeClient):
    """A baseline-style engine: has .info and takes the run seed."""

    accepts_seed = True

    def __init__(self, decisions, name="ollama:qwen3:8b", transport="ollama", **extra):
        super().__init__(decisions)
        self.info = {"name": name, "transport": transport, "model_requested": "x", **extra}
        self.seeds = []

    def decide(self, observations, seed=None):
        self.seeds.append(seed)
        return super().decide(observations)


def quiet(*_):
    pass


def test_default_engine_keeps_file_names_and_marks_jev(tmp_path):
    records, _ = run.run_experiment(FakeClient(perfect()), None, 7, 10, tmp_path, out=quiet)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert any(n.endswith("-7runs.jsonl") for n in names)
    assert all(r["engine"]["name"] == "jev" for r in records)
    meta = json.loads(next(tmp_path.glob("*-summary.json")).read_text())["meta"]
    assert meta["engine"]["name"] == "jev"


def test_baseline_engine_gets_seed_and_suffixed_files(tmp_path):
    eng = FakeLLM(perfect())
    records, _ = run.run_experiment(eng, None, 7, 500, tmp_path, out=quiet)
    assert eng.seeds == list(range(500, 507))
    assert next(tmp_path.glob("*.jsonl")).name.endswith("-7runs-ollama-qwen3-8b.jsonl")
    assert records[0]["engine"] == {"name": "ollama:qwen3:8b", "transport": "ollama",
                                    "model_requested": "x", "model_version": "fake-1"}


def test_engine_without_seed_support_is_called_without_seed(tmp_path):
    eng = FakeLLM(perfect(), name="claude-sonnet", transport="claude-cli")
    eng.accepts_seed = False
    run.run_experiment(eng, None, 1, 1, tmp_path, out=quiet)
    assert eng.seeds == [None]


def test_fallback_notice_reaches_meta(tmp_path):
    eng = FakeLLM(perfect(), name="claude-opus", transport="claude-cli", notice="using the claude CLI")
    run.run_experiment(eng, None, 1, 1, tmp_path, out=quiet)
    meta = json.loads(next(tmp_path.glob("*-summary.json")).read_text())["meta"]
    assert meta["engine"]["notice"] == "using the claude CLI"


def test_single_run_output_names_engine_without_probability(tmp_path):
    lines = []
    eng = FakeLLM([ok_decision()], name="claude-sonnet", transport="anthropic-api")
    eng.decisions[0].human_review_probability = None
    run.run_experiment(eng, "healthy", 1, 1, tmp_path, out=lines.append)
    text = "\n".join(lines)
    assert "claude-sonnet (anthropic-api)" in text and "P(yes)" not in text


@pytest.mark.parametrize("value", ["gpt-5", "ollama:"])
def test_unknown_engine_rejected(value):
    with pytest.raises(SystemExit) as e:
        run.parse_args(["--engine", value])
    assert e.value.code != 0


def test_engine_flag_parses():
    assert run.parse_args(["--engine", "claude-opus"]).engine == "claude-opus"
    assert run.parse_args([]).engine == "jev"


# --- comparison report -------------------------------------------------------

def results(tmp_path, engine, decisions, seed=100, runs=7, **kw):
    d = tmp_path / engine.replace(":", "-")
    run.run_experiment(engine == "jev" and FakeClient(decisions) or FakeLLM(decisions, name=engine, **kw),
                       None, runs, seed, d, out=quiet)
    path = next(d.glob("*.jsonl"))
    return path.name, report.load(path)


def test_compare_side_by_side(tmp_path):
    wrong = perfect()
    wrong[0] = ok_decision("critical", "escalate", "yes")
    a = results(tmp_path, "jev", perfect())
    b = results(tmp_path, "claude-sonnet", wrong, transport="anthropic-api")
    text = report.render_compare([a, b])
    assert "| | jev / baseline | claude-sonnet / baseline |" in text
    assert "| Accuracy | 7/7 (100%) | 6/7 (86%) |" in text
    assert "| healthy | 1/1 (100%) | 0/1 (0%) |" in text
    assert "Warning" not in text


def test_compare_warns_on_different_seeds_and_logs(tmp_path):
    a = results(tmp_path, "jev", perfect(), seed=100)
    b = results(tmp_path, "ollama:qwen3:8b", perfect(), seed=999)
    for r in b[1]:
        r["include_logs"] = False
    text = report.render_compare([a, b])
    assert "same scenario and seed pairs" in text and "with logs and some without" in text


def test_compare_flags_cli_transport(tmp_path):
    a = results(tmp_path, "jev", perfect())
    b = results(tmp_path, "claude-opus", perfect(), transport="claude-cli")
    assert "ran through the `claude -p` CLI" in report.render_compare([a, b])


def test_old_records_without_engine_read_as_jev(tmp_path):
    name, recs = results(tmp_path, "jev", perfect())
    for r in recs:
        r.pop("engine")
    assert report.engine_of(recs)["name"] == "jev"
    assert "What Jev chose" in report.render(recs, name)


def test_compare_main_writes_file(tmp_path, capsys):
    results(tmp_path, "jev", perfect())
    results(tmp_path, "claude-sonnet", perfect(), transport="anthropic-api")
    files = [str(p) for p in sorted(tmp_path.glob("*/*runs*.jsonl"))]
    assert report.main(["--compare", *files]) == 0
    assert list(tmp_path.glob("*/*-compare-report.md"))


def test_compare_needs_two_files(tmp_path):
    with pytest.raises(SystemExit):
        report.main(["--compare", "only-one.jsonl"])
