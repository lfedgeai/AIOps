"""Observation layouts (change observation-facts). Offline: observations come from the
fake Tier 3 runner (every scenario, leak-bait telemetry) and from hand-built dicts."""
import ast
import inspect
import json
import re

import pytest

from tests.tier3.helpers import QUIET, SeqEngine, ok_decision
from tier3 import layouts
from tier3.layouts import (LAYOUTS, VOCABULARY, LayoutError, apply_layout, get_layout, record_layout)
from tier3.scenarios import get_scenario, load_all
from tier3.vocab import FAULT_TYPES

SUSPECT_WORDS = ("suspect", "likely", "root", "cause", "origin", "culprit", "probable", "blame",
                 "faulty", "broken", "anomal")


def _observations(tmp_path, eid):
    from tests.tier3.test_runner import CFG, make
    from tier3.runner import run_experiment
    deps, *_ = make(tmp_path, SeqEngine(ok_decision(**QUIET)))
    rec = run_experiment(get_scenario(eid), deps, CFG, "r", tmp_path)
    return [p["request"] for p in rec["polls"]]


def _tokens(value) -> set[str]:
    """Every word in every key and string value."""
    out: set[str] = set()

    def walk(v):
        if isinstance(v, dict):
            for k, x in v.items():
                out.update(re.findall(r"[A-Za-z][A-Za-z0-9_.\-]*", str(k)))
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str):
            out.update(re.findall(r"[A-Za-z][A-Za-z0-9_.\-]*", v))
    walk(value)
    return {t.rstrip(".-") for t in out}


def obs_example():
    return {
        "window": {"start": "2026-11-01T22:23:35+00:00", "end": "2026-11-01T22:26:35+00:00"},
        "baseline_window": {"start": "2026-11-01T22:23:20+00:00", "end": "2026-11-01T22:26:20+00:00"},
        "deployments": [
            {"deployment": "cart", "desired": 1, "ready": 0, "available": 0, "updated": 1,
             "rollout": "progressing", "baseline_desired": 1, "baseline_available": 1},
            {"deployment": "frontend", "desired": 1, "ready": 1, "available": 1, "updated": 1,
             "rollout": "complete", "baseline_desired": 1, "baseline_available": 1},
            {"deployment": "valkey-cart", "desired": 0, "ready": 0, "available": 0, "updated": 0,
             "rollout": "complete", "baseline_desired": 1, "baseline_available": 1}],
        "pods": [], "nodes": [],
        "events": [{"reason": "BackOff", "object": "pod/cart-1", "message": "m", "count": 2,
                    "last_seen": "2026-11-01T22:24:35Z"}],
        "services": [
            {"service": "frontend", "request_rate_per_s": 16.0, "error_rate": 0.4, "latency_p50_ms": 3.0,
             "latency_p95_ms": 90.0, "baseline_request_rate_per_s": 16.0, "baseline_error_rate": 0.0,
             "baseline_latency_p95_ms": 30.0, "error_rate_change": 0.4, "latency_p95_change_ms": 60.0},
            {"service": "cart", "request_rate_per_s": 2.0, "error_rate": 0.0, "latency_p50_ms": 1.0,
             "latency_p95_ms": 2.0, "baseline_request_rate_per_s": 2.0, "baseline_error_rate": 0.0,
             "baseline_latency_p95_ms": 0.0, "error_rate_change": 0.0, "latency_p95_change_ms": 2.0}],
        "dependencies": [{"from": "frontend", "to": "cart", "calls": 10, "errors": 7}],
        "error_logs": [{"ts": "2026-11-01T22:26:23.5754788Z", "service": "frontend", "level": "ERROR",
                        "message": "failed", "count": 3}],
        "failing_traces": [],
    }


# --- registry and raw ------------------------------------------------------------

def test_raw_is_byte_identical_on_every_scenario(tmp_path):
    for eid in sorted(load_all()):
        for obs in _observations(tmp_path, eid):
            out, notes = apply_layout("raw", obs)
            assert json.dumps(out) == json.dumps(obs) and notes == {}


def test_unknown_layout():
    with pytest.raises(LayoutError, match="available: raw, facts"):
        get_layout("summary")


def test_record_layout_defaults_to_raw_v1():
    assert record_layout({}) == ({"name": "raw", "version": 1, "sha256": None}, True)
    ident = get_layout("facts").identity
    assert record_layout({"observation": ident}) == (ident, False)


def test_hashes_are_stable_and_distinct():
    assert get_layout("facts").sha256 == get_layout("facts").sha256
    assert len({lay.sha256 for lay in LAYOUTS.values()}) == len(LAYOUTS)


def test_layouts_module_imports_no_ground_truth():
    tree = ast.parse(inspect.getsource(layouts))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.startswith(("tier3.scenarios", "tier3.faults", "tier2.scenarios", "tier3.runner",
                                 "tier3.replay")) for m in mods), mods


# --- facts -----------------------------------------------------------------------

def test_facts_never_changes_its_input_and_is_deterministic():
    o = obs_example()
    before = json.dumps(o, sort_keys=True)
    a, _ = apply_layout("facts", o)
    b, _ = apply_layout("facts", o)
    assert json.dumps(o, sort_keys=True) == before
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_facts_adds_no_new_tokens_on_every_scenario(tmp_path):
    for eid in sorted(load_all()):
        for obs in _observations(tmp_path, eid):
            out, _ = apply_layout("facts", obs)
            new = _tokens(out) - _tokens(obs) - VOCABULARY
            assert not new, f"{eid}: facts wrote tokens not in its input or vocabulary: {sorted(new)}"
            text = json.dumps(out)
            for s in ("jev-fault", "ground_truth", "acceptable_actions", eid, *FAULT_TYPES):
                assert s not in text, (eid, s)


def test_no_suspect_words_in_vocabulary_or_output(tmp_path):
    for w in VOCABULARY:
        assert not any(s in w.lower() for s in SUSPECT_WORDS), w
    out, _ = apply_layout("facts", obs_example())
    added = _tokens(out) - _tokens(obs_example())
    assert not any(s in t.lower() for t in added for s in SUSPECT_WORDS), added


def test_peer_state_is_placed_on_both_ends_of_an_edge():
    out, _ = apply_layout("facts", obs_example())
    front = next(s for s in out["services"] if s["service"] == "frontend")
    cart = next(s for s in out["services"] if s["service"] == "cart")
    (callee,) = front["callees"]
    assert callee["service"] == "cart" and callee["calls"] == 10 and callee["errors"] == 7
    assert callee["facts"] == ["errors on 7 of 10 calls", "available 0 of 1 (baseline available 1 of 1)",
                               "error rate 0.000 (baseline 0.000)"]
    (caller,) = cart["callers"]
    assert caller["service"] == "frontend"
    assert caller["facts"] == ["errors on 7 of 10 calls", "available 1 of 1 (baseline available 1 of 1)",
                               "error rate 0.400 (baseline 0.000)"]
    assert front["callers"] == [] and cart["callees"] == []


def test_every_row_gets_the_same_statements_and_order_is_kept():
    o = obs_example()
    out, _ = apply_layout("facts", o)
    assert [s["service"] for s in out["services"]] == [s["service"] for s in o["services"]]
    assert [d["deployment"] for d in out["deployments"]] == [d["deployment"] for d in o["deployments"]]
    for s in out["services"]:
        assert set(s) == set(o["services"][0]) | {"facts", "callers", "callees"}
        assert [f.split(" ")[0] for f in s["facts"]] == ["available", "error", "p95", "request"]
    for d in out["deployments"]:
        assert [f.split(" ")[0] for f in d["facts"]] == ["available", "desired", "rollout"]


def test_statements_always_show_both_values():
    out, _ = apply_layout("facts", obs_example())
    front = next(s for s in out["services"] if s["service"] == "frontend")
    assert "error rate 0.400 (baseline 0.000)" in front["facts"]
    assert "p95 latency 90.0 ms, 3.0 times baseline (30.0 ms)" in front["facts"]
    assert "request rate 16.00/s (baseline 16.00/s)" in front["facts"]
    cart = next(s for s in out["services"] if s["service"] == "cart")
    assert "p95 latency 2.0 ms (baseline 0.0 ms)" in cart["facts"]  # no ratio, no infinity
    valkey = next(d for d in out["deployments"] if d["deployment"] == "valkey-cart")
    assert valkey["facts"] == ["available 0 of 0 (baseline available 1 of 1)",
                               "desired replicas 0 (baseline 1)", "rollout complete"]
    assert not any("same" in f for r in out["services"] + out["deployments"] for f in r["facts"])


@pytest.mark.parametrize("cur,base,want", [
    (0.014, 0.006, "error rate 0.014 (baseline 0.006)"),
    (0.004, 0.0, "error rate 0.004 (baseline 0.000)"),
    (None, 0.1, "error rate not measured"),
    (0.1, None, "error rate 0.100 (baseline not measured)"),
])
def test_error_rate_boundaries(cur, base, want):
    assert layouts._error_fact(cur, base) == want


@pytest.mark.parametrize("cur,base,want", [
    (10, 0.04, "p95 latency 10.0 ms (baseline 0.0 ms)"),
    (10, 0.05, "p95 latency 10.0 ms, 100.0 times baseline (0.1 ms)"),  # ratio of the shown values
    (1.9, 1.9, "p95 latency 1.9 ms, 1.0 times baseline (1.9 ms)"),
    (True, 1.0, "p95 latency not measured"),
])
def test_latency_boundaries(cur, base, want):
    assert layouts._latency_fact(cur, base) == want


def test_new_deployment_without_baseline_writes_no_python_words():
    row = {"deployment": "x", "desired": 1, "ready": None, "available": 1, "updated": 1,
           "rollout": None, "baseline_desired": None, "baseline_available": None}
    facts = layouts._deployment_facts(row)
    assert facts == ["available 1 of 1 (baseline not measured)", "desired replicas 1 (baseline not measured)",
                     "rollout not measured"]
    assert not any("None" in f for f in facts)
    assert layouts._deployment_facts({**row, "desired": None}) == ["deployment not found now"]


def test_times_become_distances_and_keep_their_meaning():
    out, _ = apply_layout("facts", obs_example())
    assert out["window"] == {"length_s": 180}
    assert out["baseline_window"] == {"length_s": 180, "ends_s_before_window_end": 15}
    assert out["error_logs"][0]["ts_s_before_window_end"] == 11 and "ts" not in out["error_logs"][0]
    assert out["events"][0]["last_seen_s_before_window_end"] == 120 and "last_seen" not in out["events"][0]


def test_unparseable_time_is_null_not_invented():
    o = obs_example()
    o["error_logs"][0]["ts"] = "yesterday"
    out, _ = apply_layout("facts", o)
    assert out["error_logs"][0]["ts_s_before_window_end"] is None


def test_missing_window_end_keeps_the_source_times():
    o = obs_example()
    o["window"]["end"] = None
    out, _ = apply_layout("facts", o)
    assert out["window"] == o["window"] and out["error_logs"][0]["ts"] == o["error_logs"][0]["ts"]


def _size(o):
    return len(json.dumps(o, sort_keys=True).encode())


def test_budget_drops_one_field_from_every_row_in_order(monkeypatch):
    o = obs_example()
    full, _ = apply_layout("facts", o)
    monkeypatch.setattr(layouts, "BYTE_BUDGET", _size(full) - 10)
    out, notes = apply_layout("facts", o)
    assert notes == {"layout_budget_services_callers": 2}
    assert all("callers" not in s and "callees" in s and "facts" in s for s in out["services"])
    assert len(out["services"]) == len(o["services"]) and len(out["deployments"]) == len(o["deployments"])


def test_budget_still_over_is_noted(monkeypatch):
    monkeypatch.setattr(layouts, "BYTE_BUDGET", 10)
    out, notes = apply_layout("facts", obs_example())
    assert "layout_budget_exceeded" in notes
    assert all("facts" not in d for d in out["deployments"])


def test_helpers_list_is_complete():
    """Every module function _facts reaches must be hashed, or an edit would not
    change the layout's sha256."""
    names = {n for n, v in vars(layouts).items() if inspect.isfunction(v) and v.__module__ == layouts.__name__}
    listed = {f.__name__ for f in get_layout("facts").helpers} | {"_facts"}
    seen, todo = set(), ["_facts"]
    while todo:
        fn = todo.pop()
        if fn in seen:
            continue
        seen.add(fn)
        tree = ast.parse(inspect.getsource(getattr(layouts, fn)).lstrip())
        todo += [n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in names]
    assert seen <= listed, seen - listed


def test_hash_changes_with_helper_and_budget(monkeypatch):
    before = get_layout("facts").sha256
    raw_before = get_layout("raw").sha256
    monkeypatch.setattr(layouts, "BYTE_BUDGET", 1)
    assert get_layout("facts").sha256 != before
    assert get_layout("raw").sha256 == raw_before
    monkeypatch.undo()
    monkeypatch.setattr(inspect, "getsource",
                        lambda f, _g=inspect.getsource: _g(f) + ("#x" if f is layouts._f3 else ""))
    assert get_layout("facts").sha256 != before


# --- replay and compare ------------------------------------------------------------

def test_replay_refuses_a_laid_out_source(tmp_path):
    from tests.test_replay import Capture, _tier3_source
    from tier3.replay import replay_record
    src = _tier3_source(tmp_path, "K001")
    src["observation"] = get_layout("facts").identity
    with pytest.raises(LayoutError, match="replay needs raw source records"):
        replay_record(src, Capture(ok_decision(**QUIET)), run_id="rp", layout="facts")


def test_replay_cli_refuses_before_any_call(tmp_path, capsys):
    from tests.test_replay import _tier3_source
    from tier3.cli import main
    src = _tier3_source(tmp_path / "s", "K001")
    src["observation"] = get_layout("facts").identity
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "K001.json").write_text(json.dumps(src))
    assert main(["replay", str(tmp_path / "src"), "--observation", "facts", "--dry-run"]) == 2
    assert "replay needs raw source records" in capsys.readouterr().err


def test_cli_rejects_unknown_layout(capsys):
    from tier3.cli import main
    with pytest.raises(SystemExit):
        main(["replay", "x", "--observation", "summary"])
    assert "invalid choice" in capsys.readouterr().err


def _rec(mode, layout_ident, eid="K001", path="/x/K001.json"):
    return {"experiment_id": eid, "mode": mode, "polls": [], "engine": {"name": "jev"},
            "source": {"path": path}, "observation": layout_ident}


def test_compare_groups_by_layout_and_rejects_two_hashes():
    from pathlib import Path

    from prompts.compare import HashConflict, group_records
    facts = get_layout("facts").identity
    items = [(Path("a.json"), _rec("replay", get_layout("raw").identity)),
             (Path("b.json"), _rec("replay", facts))]
    groups = group_records(items, lambda r: "jev", lambda r: record_layout(r)[0])
    assert [g.label for g in groups] == ["jev / baseline v1 (replay)", "jev / baseline v1 + facts v1 (replay)"]
    bad = {**facts, "sha256": "0" * 64}
    old = {"experiment_id": "K001", "mode": "replay", "polls": [], "engine": {"name": "jev"},
           "source": {"path": "/x/K001.json"}}  # written before layouts existed
    mixed = group_records(items + [(Path("o.json"), old)], lambda r: "jev", lambda r: record_layout(r)[0])
    assert [len(g.records) for g in mixed] == [2, 1]
    with pytest.raises(HashConflict, match="layout 'facts' v1 has two different hashes"):
        group_records(items + [(Path("c.json"), _rec("replay", bad))], lambda r: "jev",
                      lambda r: record_layout(r)[0])
