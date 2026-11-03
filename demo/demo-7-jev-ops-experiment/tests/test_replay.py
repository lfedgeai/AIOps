"""Replay of stored Tier 2/3/4 observations (jev-prompt-variants task 4). Offline: the
source records come from the offline fake runners (leak-bait telemetry included); the
replaying engine is a fake that records what it is sent."""
import hashlib
import json
from pathlib import Path

import pytest

from tests.tier3.helpers import QUIET, SeqEngine, ok_decision
from tier3.replay import load_sources, plan, replay_batch, replay_record, source_truncated
from tier3.report import load_records
from tier3.vocab import FAULT_TYPES

TIER3_SECRETS = ("jev-fault", "feature_flag", "flagd", "ground_truth", "acceptable_actions",
                 "expected", "fault_type", "remediation", "recovered_by_action", *FAULT_TYPES)


class Capture:
    """Answers from a script and records exactly what it was sent."""

    def __init__(self, *items, prompt=None):
        self.inner = SeqEngine(*items)
        self.sent = self.inner.calls
        self.info = {"name": "capture", "transport": "fake"}
        if prompt is not None:
            self.prompt = prompt

    def decide(self, obs):
        return self.inner.decide(obs)


def _tier3_source(tmp_path, eid, engine=None):
    from tests.tier3.test_runner import CFG, make
    from tier3.runner import run_experiment
    from tier3.scenarios import get_scenario
    deps, *_ = make(tmp_path, engine or SeqEngine(ok_decision(**QUIET)))
    return run_experiment(get_scenario(eid), deps, CFG, "src", tmp_path)


def _variants(tier_module, tier):
    from prompts.loader import list_variants
    return [tier_module.load_prompt(n) for n in list_variants(tier)]


# --- the engine sees only the stored request (4.5) -----------------------------

def _all_tier3_ids():
    from tier3.scenarios import load_all
    return sorted(load_all())


@pytest.mark.parametrize("layout", __import__("tier3.layouts", fromlist=["x"]).list_layouts())
@pytest.mark.parametrize("eid", _all_tier3_ids())
def test_tier3_replay_sends_only_the_stored_requests(tmp_path, eid, layout):
    from jev.client import JevClient, JevConfig
    from tier3 import questions
    from tier3.jev_adapter import JevAdapter
    from tier3.layouts import apply_layout
    src = _tier3_source(tmp_path, eid)
    before = json.dumps(src, sort_keys=True, default=str)
    for v in _variants(questions, "tier3"):
        eng = Capture(ok_decision(**QUIET), prompt=v)
        rec = replay_record(src, eng, run_id="rp", layout=layout)
        assert eng.sent == [apply_layout(layout, p["request"])[0] for p in src["polls"]]
        assert rec["observation"]["name"] == layout
        adapter = JevAdapter(JevClient(JevConfig(api_key="k"), post=lambda *a: None), prompt=v)
        for obs in eng.sent:
            req = adapter.build_request(obs)
            assert req["questions"] == v.questions and req["state"] == {"observations": obs}
            text = json.dumps(req)
            for s in (*TIER3_SECRETS, eid):
                assert s not in text, f"{s!r} reached the engine on replay of {eid} under {v.name}"
        assert rec["prompt"] == v.identity and rec["mode"] == "replay"
    assert json.dumps(src, sort_keys=True, default=str) == before, "replay changed the source record"


@pytest.mark.parametrize("eid", ["K000", "K001", "K003", "K009"])
def test_tier4_records_replay_without_remediation(tmp_path, eid):
    from tests.tier4.test_runner import CFG, make
    from tier3.scenarios import get_scenario
    from tier4.runner import run_experiment
    deps, *_ = make(tmp_path, SeqEngine(ok_decision(**QUIET), ok_decision()))
    src = run_experiment(get_scenario(eid), deps, CFG, "src", tmp_path)
    assert "remediation" in src
    eng = Capture(ok_decision(**QUIET))
    rec = replay_record(src, eng, run_id="rp")
    assert eng.sent == [p["request"] for p in src["polls"]]
    assert "remediation" not in rec and "remediation" not in json.dumps(eng.sent)
    assert rec["timing"]["mttd_seconds"] is None


def _tier2_ids():
    from tier2.scenarios import load_all
    return sorted(load_all())


@pytest.mark.parametrize("eid", _tier2_ids())
def test_tier2_replay_sends_only_the_stored_request(tmp_path, eid):
    from jev.client import JevClient, JevConfig
    from tests.tier2.helpers import FakeEngine, ok_decision as ok2
    from tests.tier2.test_runner import CFG, make
    from tier2 import jev_adapter
    from tier2.replay import replay_record as replay2
    from tier2.runner import run_experiment
    from tier2.scenarios import get_scenario
    deps, *_ = make()
    src = run_experiment(get_scenario(eid), deps, CFG, "src", tmp_path)
    for v in _variants(jev_adapter, "tier2"):
        eng = FakeEngine(ok2())
        eng.prompt = v
        rec = replay2(src, eng, run_id="rp")
        assert eng.calls == [src["request"]]
        req = jev_adapter.JevAdapter(JevClient(JevConfig(api_key="k"), post=lambda *a: None),
                                     prompt=v).build_request(eng.calls[0])
        text = json.dumps(req)
        for s in ("cartFailure", "paymentFailure", "productCatalogFailure", "flagd", "feature_flag",
                  "ground_truth", "acceptable_actions", "expected", eid):
            assert s not in text, f"{s!r} reached the engine on replay of {eid}"
        assert rec["prompt"] == v.identity and rec["timing"]["diagnosis_seconds"] is None
        assert rec["evaluation"]["detection"] in ("PASS", "FAIL")


# --- scoring, censoring, timings (4.1) ------------------------------------------

def _poll(n, detected, secs):
    from tier3.runner import _decision_dict
    d = ok_decision() if detected else ok_decision(**QUIET)
    return {"n": n, "seconds_since_inject": secs, "request": {"poll": n}, "decision": _decision_dict(d)}


def _src(incident, detections, run_id="live"):
    gt = ({"incident": True, "component": "cart", "diagnosis": "service_unavailable",
           "acceptable_actions": ["scale_up"]} if incident else
          {"incident": False, "component": "none", "diagnosis": "none", "acceptable_actions": ["no_action"]})
    return {"run_id": run_id, "experiment_id": "K001" if incident else "K000", "mode": "poll",
            "status": "ok", "ground_truth": gt, "fault": None,
            "polls": [_poll(i + 1, det, 15.0 * (i + 1)) for i, det in enumerate(detections)]}


def test_fault_run_detected_on_a_stored_poll():
    src = _src(True, [False, True])
    rec = replay_record(src, Capture(ok_decision(**QUIET), ok_decision()), run_id="rp")
    assert rec["scored_poll"] == 2 and rec["detect_poll"] == 2 and not rec["censored"]
    assert rec["replay_detect_seconds_since_inject"] == 30.0 and rec["timing"]["mttd_seconds"] is None
    assert rec["evaluation"] == {"detection": "PASS", "localization": "PASS", "diagnosis": "PASS",
                                 "action": "PASS"}


def test_stricter_variant_on_a_one_poll_fault_run_is_censored():
    src = _src(True, [True])
    assert source_truncated(src)
    rec = replay_record(src, Capture(ok_decision(**QUIET)), run_id="rp")
    assert rec["censored"] is True and rec["detect_poll"] is None
    assert rec["evaluation"]["detection"] == "FAIL" and rec["scored_poll"] == 1


def test_full_window_miss_is_not_censored():
    src = _src(True, [False, False, False])  # live never detected: the whole window was stored
    assert not source_truncated(src)
    assert replay_record(src, Capture(ok_decision(**QUIET)), run_id="rp")["censored"] is False


def test_control_counts_false_positive_polls_over_every_stored_poll():
    src = _src(False, [False] * 15)
    script = [ok_decision() if n in (9, 12) else ok_decision(**QUIET) for n in range(1, 16)]
    rec = replay_record(src, Capture(*script), run_id="rp")
    assert len(rec["polls"]) == 15 and rec["scored_poll"] == 9
    assert rec["false_positive"] == {"false_positive_polls": 2, "answered_polls": 15, "polls": 15,
                                     "false_positive_run": True}
    assert rec["censored"] is False


def test_truncated_control_is_censored_when_the_variant_stays_quiet():
    """Tier 4 stops no-incident runs at the first detection too."""
    src = _src(False, [False, False, True])
    rec = replay_record(src, Capture(ok_decision(**QUIET)), run_id="rp")
    assert rec["censored"] is True and rec["false_positive"]["false_positive_run"] is False


def test_engine_errors_are_recorded_and_all_errors_make_an_error_record():
    src = _src(True, [False, True])
    rec = replay_record(src, Capture(RuntimeError("boom"), ok_decision()), run_id="rp")
    assert rec["status"] == "ok" and rec["polls"][0]["decision"]["status"] == "error"
    rec = replay_record(src, Capture(RuntimeError("boom")), run_id="rp")
    assert rec["status"] == "error" and "boom" in rec["message"]
    assert set(rec["evaluation"].values()) == {"UNKNOWN"}  # an error is never scored as a miss


def test_all_error_record_scores_unknown():
    from tier3.engine import Decision
    src = _src(True, [True])
    rec = replay_record(src, Capture(Decision(status="error", message="x")), run_id="rp")
    assert rec["status"] == "error"
    assert set(rec["evaluation"].values()) == {"UNKNOWN"}


def test_source_and_prompt_recorded():
    src = _src(True, [True])
    rec = replay_record(src, Capture(ok_decision()), run_id="rp", source_path="results/x/K001.json")
    assert rec["source"]["path"] == "results/x/K001.json" and rec["source"]["run_id"] == "live"
    assert rec["source"]["prompt"]["name"] == "baseline"  # no prompt field: read as baseline
    assert rec["prompt"] is None  # the fake engine carries no variant


# --- batch: files, skipping, sources untouched (4.3) ------------------------------

def _digest(root: Path) -> dict:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def test_batch_writes_replay_records_and_never_touches_sources(tmp_path):
    srcdir = tmp_path / "live" / "run1"
    srcdir.mkdir(parents=True)
    a, b, bad = _src(True, [True]), _src(False, [False, False]), _src(True, [True])
    bad["status"] = "contaminated"
    for name, r in (("K001.json", a), ("K000.json", b), ("K001-2.json", bad)):
        (srcdir / name).write_text(json.dumps(r))
    (srcdir / "summary.json").write_text("{}")
    before = _digest(tmp_path / "live")

    out = tmp_path / "results"
    s = replay_batch(tmp_path / "live", Capture(ok_decision(**QUIET)), out, run_id="rp", out=lambda *_: None)
    assert _digest(tmp_path / "live") == before
    assert s["replayed"] == 2 and [x["reason"] for x in s["skipped"]] == ["status not ok"]
    assert s["censored"] == 1
    files = sorted(p.name for p in (out / "replay" / "rp").iterdir())
    assert files == ["K000.json", "K001.json", "summary.json"]
    assert plan(load_sources(tmp_path / "live"))["engine_calls"] == 3


def test_replay_records_are_not_read_as_sources_or_by_live_scorecards(tmp_path):
    out = tmp_path / "results"
    srcdir = tmp_path / "results" / "run1"
    srcdir.mkdir(parents=True)
    (srcdir / "K001.json").write_text(json.dumps(_src(True, [True])))
    replay_batch(srcdir, Capture(ok_decision()), out, run_id="rp", out=lambda *_: None)
    assert len(load_sources(out)) == 1  # the replay record is not a source
    assert len(load_records(out)) == 1 and len(load_records(out, include_replay=True)) == 2


# --- command line (4.1, 4.2) -------------------------------------------------------

def _forbid_engine(monkeypatch, target):
    def boom(*a, **k):
        raise AssertionError("built an engine on a dry run")
    monkeypatch.setattr(target, boom)


def test_tier3_cli_dry_run_calls_nothing(tmp_path, monkeypatch, capsys):
    from tier3 import cli
    (tmp_path / "K001.json").write_text(json.dumps(_src(True, [False, True])))
    _forbid_engine(monkeypatch, "tier3.cli.build_engine")
    assert cli.main(["replay", str(tmp_path), "--dry-run"]) == 0
    assert "replay 1 record(s), 2 engine call(s)" in capsys.readouterr().out


def test_tier3_cli_replay_uses_the_engine_and_variant(tmp_path, monkeypatch, capsys):
    from tier3 import cli
    src = tmp_path / "live"
    src.mkdir()
    (src / "K001.json").write_text(json.dumps(_src(True, [True])))
    seen = {}

    def fake_build(name, prompt):
        seen.update(name=name, prompt=prompt)
        return Capture(ok_decision(), prompt=prompt)
    monkeypatch.setattr(cli, "build_engine", fake_build)
    assert cli.main(["replay", str(src), "--results-dir", str(tmp_path / "out")]) == 0
    assert seen["name"] == "jev" and seen["prompt"].name == "baseline"
    assert len(list((tmp_path / "out" / "replay").rglob("K001.json"))) == 1
    assert "censored runs: 0/1" in capsys.readouterr().out


def test_tier3_scorecard_notes_left_out_replay_records(tmp_path, monkeypatch, capsys):
    from tier3 import cli
    replay_batch_dir = tmp_path / "r"
    (tmp_path / "live").mkdir()
    (tmp_path / "live" / "K001.json").write_text(json.dumps(_src(True, [True])))
    replay_batch(tmp_path / "live", Capture(ok_decision()), replay_batch_dir, run_id="rp", out=lambda *_: None)
    assert cli.main(["scorecard", str(replay_batch_dir)]) == 0
    captured = capsys.readouterr()
    assert "no run records found" in captured.out and "--include-replay" in captured.err
    assert cli.main(["scorecard", str(replay_batch_dir), "--include-replay"]) == 0
    assert "| K001 |" in capsys.readouterr().out


def test_tier2_cli_dry_run_calls_nothing(tmp_path, monkeypatch, capsys):
    from tests.tier2.test_runner import CFG, make
    from tier2 import cli
    from tier2.runner import run_experiment
    from tier2.scenarios import get_scenario
    deps, *_ = make()
    run_experiment(get_scenario("F003"), deps, CFG, "src", tmp_path)
    _forbid_engine(monkeypatch, "tier2.cli.build_engine")
    assert cli.main(["replay", str(tmp_path), "--dry-run"]) == 0
    assert "replay 1 record(s), 1 engine call(s)" in capsys.readouterr().out
