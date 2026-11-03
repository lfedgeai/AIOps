"""Prompt-variant comparison reports (jev-prompt-variants task 5). Offline: hand-built
records, fake engines carrying made-up variants."""
import json
import os
from pathlib import Path

import pytest

from prompts.compare import HashConflict, cell, f1_cell, render
from prompts.loader import Variant
from tests.test_replay import Capture, _src
from tests.tier3.helpers import QUIET, ok_decision
from tier3.compare import ROWS, format_compare, load_items
from tier3.replay import replay_batch
from tier3.report import engine_name


def variant(name, version=1, sha=None, dev=(), style="wording"):
    return Variant(name=name, version=version, tier="tier3", rationale="test", dev_records=tuple(dev),
                   questions={}, sha256=sha or f"{name}-{version}".ljust(64, "0"), path="", style=style)


def as_live(src: dict) -> dict:
    """A complete live record for a hand-built source: scored with the live rules from its
    own stored answers (replay with an engine that echoes them), then marked live."""
    from tier3.engine import Decision
    from tier3.replay import replay_record
    answers = iter(p["decision"] for p in src["polls"])

    class Echo:
        def decide(self, obs):
            d = next(answers)
            return Decision(**{k: d[k] for k in ("status", "incident_detected", "affected_component",
                                                  "diagnosis", "recommended_action", "confidence")})
    rec = replay_record(src, Echo(), run_id=src["run_id"])
    for k in ("source", "censored", "detect_poll", "replay_detect_seconds_since_inject"):
        rec.pop(k)
    rec.update(mode="poll", polls=src["polls"], prompt=None, engine=None)
    return rec


def write_live(root: Path, records: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, rec in records.items():
        if rec.get("ground_truth") and "polls" in rec and "evaluation" not in rec:
            rec = as_live(rec)
        (root / name).write_text(json.dumps(rec))
    return root


def live_set(root):
    """Two fault runs (one poll each, both detected live) and two no-incident runs over
    three polls; the second no-incident run raised a false alarm on poll 2 live."""
    return write_live(root, {"K001.json": _src(True, [True]), "K001-2.json": _src(True, [True]),
                             "K000.json": _src(False, [False, False, False]),
                             "K000-2.json": _src(False, [False, True, False])})


def replay(src_root, out, v, *answers, run_id):
    replay_batch(src_root, Capture(*answers, prompt=v), out, run_id=run_id, out=lambda *_: None)
    return out / "replay" / run_id


# --- cells ---------------------------------------------------------------------

def test_cells():
    assert cell((27, 30)) == "27/30 (90%)" and cell((0, 0)) == "n/a" and cell((1, 3)) == "1/3 (33%)"
    assert f1_cell((36, 40)) == "36/40 (0.90)" and f1_cell((0, 0)) == "n/a"


# --- Tier 3 ----------------------------------------------------------------------

def test_three_variants_side_by_side(tmp_path):
    src = live_set(tmp_path / "live")
    out = tmp_path / "out"
    base = replay(src, out, variant("baseline"), ok_decision(), run_id="a")
    # strict: quiet everywhere -> fault runs censored, no false alarms
    strict = replay(src, out, variant("strict-incident"), ok_decision(**QUIET), run_id="b")
    text = format_compare([str(src), str(base), str(strict)])
    head = [l for l in text.splitlines() if l.startswith("| |")][0]
    assert head == ("| | jev / baseline v1 (live) | capture / baseline v1 (replay) | "
                    "capture / strict-incident v1 (replay) |")
    rows = {l.split(" | ")[0][2:]: l for l in text.splitlines() if l.startswith("| ") and " | " in l}
    assert rows["detect (fault runs)"].endswith("| 2/2 (100%) | 2/2 (100%) | 0/2 (0%) |")
    assert rows["FP runs (no-incident runs)"].endswith("| 1/2 (50%) | 2/2 (100%) | 0/2 (0%) |")
    assert rows["censored runs (replay)"].endswith("| n/a | 0/4 (0%) | 2/4 (50%) |")
    assert rows["detection F1"].endswith("| 4/5 (0.80) | 4/6 (0.67) | 0/2 (0.00) |")
    assert "Both the original live baseline and a replayed baseline are shown" in text
    assert "4 record(s) had no prompt field" in text  # the live sources


def test_policy_stated_variants_are_labelled(tmp_path):
    src = live_set(tmp_path / "live")
    out = tmp_path / "out"
    replay(src, out, variant("strict-incident", style="policy"), ok_decision(), run_id="a")
    head = [l for l in format_compare([str(out)]).splitlines() if l.startswith("| |")][0]
    assert head == "| | capture / strict-incident v1 (replay, policy stated) |"


def test_compared_over_shared_sources_with_a_warning(tmp_path):
    src = live_set(tmp_path / "live")
    out = tmp_path / "out"
    a = replay(src, out, variant("baseline"), ok_decision(), run_id="a")
    b = replay(src, out, variant("strict-incident"), ok_decision(), run_id="b")
    os.remove(b / "K000-2.json")  # one replay stopped early
    text = format_compare([str(a), str(b)])
    assert "capture / baseline v1 (replay) has 1 source record(s) the other columns lack (K000-2.json)" in text  # replay files carry their source file names
    assert "compared over the 3 they share" in text
    assert "| runs ok | 3/3 (100%) | 3/3 (100%) |" in text


def test_unpaired_uses_every_record_and_warns(tmp_path):
    a = live_set(tmp_path / "a")
    b = write_live(tmp_path / "b", {"K001.json": _src(True, [True], run_id="other")})
    for f in b.iterdir():
        rec = json.loads(f.read_text())
        rec["engine"] = {"name": "claude-opus"}
        f.write_text(json.dumps(rec))
    text = format_compare([str(a), str(b)], unpaired=True)
    assert "--unpaired" in text and "| runs ok | 1/1 (100%) | 4/4 (100%) |" in text
    paired = format_compare([str(a), str(b)])
    assert "| runs ok | n/a | n/a |" in paired


def test_hash_conflict_stops_the_report(tmp_path):
    src = live_set(tmp_path / "live")
    out = tmp_path / "out"
    replay(src, out, variant("strict-incident", sha="a" * 64), ok_decision(), run_id="a")
    replay(src, out, variant("strict-incident", sha="b" * 64), ok_decision(), run_id="b")
    with pytest.raises(HashConflict, match="strict-incident.*two different hashes"):
        format_compare([str(out)])


def test_cli_compare_exits_2_on_hash_conflict_and_needs_inputs(tmp_path, capsys):
    from tier3 import cli
    src = live_set(tmp_path / "live")
    out = tmp_path / "out"
    replay(src, out, variant("x", sha="a" * 64), ok_decision(), run_id="a")
    replay(src, out, variant("x", sha="b" * 64), ok_decision(), run_id="b")
    assert cli.main(["compare", str(out)]) == 2
    assert "two different hashes" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["compare"])


def test_dev_and_confirm_confirm_first_and_flag_written_on(tmp_path):
    dev, conf = live_set(tmp_path / "dev"), live_set(tmp_path / "confirm")
    out = tmp_path / "out"
    v = variant("strict-incident", dev=(str(conf),))  # written on the confirm set: flagged
    replay(dev, out, v, ok_decision(**QUIET), run_id="d")
    replay(conf, out, v, ok_decision(**QUIET), run_id="c")
    items = load_items([str(out), str(dev), str(conf)])
    text = render("t", items, ROWS, engine_name=engine_name, dev=[str(dev)], confirm=[str(conf)],
                  dev_records_of=lambda name, ver: v.dev_records if name == v.name else None)
    assert text.index("## Confirm set (not used to write variants)") < text.index("## Dev set")
    confirm_part = text[text.index("## Confirm"):text.index("## Dev set")]
    dev_part = text[text.index("## Dev set"):]
    assert "**Flag:** capture / strict-incident v1 (replay) was written on this set" in confirm_part
    assert "Flag" not in dev_part
    assert "| runs ok | 4/4 (100%) | 4/4 (100%) |" in confirm_part  # live baseline + replay


def test_sets_are_matched_through_symlinked_combined_dirs(tmp_path):
    real = live_set(tmp_path / "run1")
    combined = tmp_path / "combined"
    combined.mkdir()
    for f in real.iterdir():
        (combined / f.name).symlink_to(f)
    out = tmp_path / "out"
    replay(combined, out, variant("baseline"), ok_decision(), run_id="a")
    text = format_compare([str(out)], confirm=[str(real)])
    assert "| runs ok | 4/4 (100%) | 4/4 (100%) |" in text  # live (via real) and replay (via links)


def test_live_and_replay_never_share_a_column(tmp_path):
    src = live_set(tmp_path / "live")
    base = replay(src, tmp_path / "out", variant("baseline"), ok_decision(), run_id="a")
    for f in base.glob("K*.json"):  # same engine name and prompt as the live records
        rec = json.loads(f.read_text())
        rec["engine"], rec["prompt"] = {"name": "jev"}, None
        f.write_text(json.dumps(rec))
    head = [l for l in format_compare([str(src), str(base)]).splitlines() if l.startswith("| |")][0]
    assert head == "| | jev / baseline v1 (live) | jev / baseline v1 (replay) |"


# --- Tier 2 ----------------------------------------------------------------------

def _t2(incident, detected, run="live"):
    gt = ({"incident": True, "service": "payment", "diagnosis": "application_failure",
           "acceptable_actions": ["investigate"]} if incident else
          {"incident": False, "service": "none", "diagnosis": "none", "acceptable_actions": ["no_action"]})
    passed = "PASS" if detected == incident else "FAIL"
    return {"run_id": run, "experiment_id": "F003" if incident else "F000", "status": "ok",
            "ground_truth": gt, "request": {"r": 1},
            "jev": {"status": "ok", "incident_detected": detected},
            "evaluation": {"detection": passed, "localization": "PASS", "diagnosis": "FAIL", "action": "PASS"}}


def test_tier2_rows(tmp_path):
    from tier2.compare import format_compare as t2
    root = write_live(tmp_path / "live", {"F003.json": _t2(True, True), "F003-2.json": _t2(True, False),
                                          "F000.json": _t2(False, True), "F000-2.json": _t2(False, False)})
    text = t2([str(root)])
    assert "| detection | 2/4 (50%) |" in text
    assert "| FP runs (healthy control) | 1/2 (50%) |" in text
    assert "| detection precision | 1/2 (50%) |" in text and "| detection recall | 1/2 (50%) |" in text
    assert "| detection F1 | 2/4 (0.50) |" in text and "| diagnosis | 0/4 (0%) |" in text


# --- Tier 1 (report.py) ------------------------------------------------------------

def _t1_file(tmp_path, name, prompt, sha, correct=True):
    from tests.helpers import FakeClient, ok_decision as ok1
    import run
    from tests.test_prompt_wiring import Prompted
    v = Variant(name=prompt, version=1, tier="tier1", rationale="t", dev_records=(), questions={},
                sha256=sha, path="")
    d = tmp_path / name
    run.run_experiment(Prompted(FakeClient([ok1()] if correct else [ok1("critical", "escalate", "yes")]), v),
                       "healthy", 2, 7, d, out=lambda *_: None)
    return str(next(d.glob("*.jsonl")))


def test_report_labels_and_by_prompt(tmp_path, capsys):
    import report
    a = _t1_file(tmp_path, "a", "baseline", "a" * 64)
    b = _t1_file(tmp_path, "b", "strict-incident", "b" * 64, correct=False)
    c = _t1_file(tmp_path, "c", "strict-incident", "b" * 64, correct=False)
    assert report.main(["--compare", a, b, c, "--by-prompt"]) == 0
    out = capsys.readouterr().out
    assert "| | jev / baseline | jev / strict-incident |" in out
    assert "| Accuracy | 2/2 (100%) | 0/4 (0%) |" in out
    assert report.main(["--compare", a, b]) == 0
    assert "| | jev / baseline | jev / strict-incident |" in capsys.readouterr().out


def test_report_hash_conflict_and_flag_checks(tmp_path, capsys):
    import report
    a = _t1_file(tmp_path, "a", "strict-incident", "a" * 64)
    b = _t1_file(tmp_path, "b", "strict-incident", "b" * 64)
    assert report.main(["--compare", a, b]) == 2
    assert "two different hashes" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        report.main(["--by-prompt"])
