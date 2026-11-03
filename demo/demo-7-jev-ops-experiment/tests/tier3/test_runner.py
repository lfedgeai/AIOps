import json

import pytest

from tests.tier3.helpers import (QUIET, FakeDeadman, FakeKube, SeqEngine, contaminated_events,
                                 ok_decision, telemetry)
from tier2.observation import ServiceStats
from tier2.workload import Workload
from tier3.collect import kube_state
from tier3.engine import Decision
from tier3.faults import FaultInjector
from tier3.runner import Config, Deps, run_batch, run_experiment
from tier3.scenarios import get_scenario, load_all
from tier3.vocab import FAULT_TYPES

BASE = {"frontend": ServiceStats(5.0, 0.0, 20, 250), "checkout": ServiceStats(0.3, 0.0, 50, 200)}
BAD = {"frontend": ServiceStats(5.0, 0.3, 20, 300), "checkout": ServiceStats(0.3, 0.5, 50, 900)}
CFG = Config(telemetry_lag_s=15, recover_timeout_s=30, recover_poll_s=10, manifest_timeout_s=30,
             manifest_poll_s=10, scrub_terms=("flagd", *FAULT_TYPES))


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += max(s, 0)


def make(tmp_path, engine=None, healthy=True, sticky=False, deadman=None, kube=None, reset_exc=None):
    clock, calls = Clock(), []
    kube = kube or FakeKube()
    kube.events = contaminated_events()
    if sticky:
        kube.sticky.add("cart")
    inj = FaultInjector(kube, ledger=tmp_path / ".active-fault.json")
    orig_inject, orig_reset = inj.inject, inj.reset

    def inject(f):
        calls.append("inject")
        return orig_inject(f)

    def reset():
        calls.append("reset")
        if reset_exc:
            raise reset_exc
        return orig_reset()
    inj.inject, inj.reset = inject, reset
    eng = engine or SeqEngine()

    def health():
        calls.append("health")
        if not healthy:
            raise RuntimeError("prometheus not ready")

    def faulted():
        return inj.ledger.exists() or bool(kube.unavailable)

    def stats(at, window):
        return dict(BAD if faulted() else BASE)

    def collect(**kw):
        calls.append("collect")
        return telemetry(faulted()), kube_state(kube, 0, 4e9)

    wl = Workload(get=lambda u: {"state": "running", "user_count": 10}, post=lambda u, d: {"success": True})
    deps = Deps(engine=eng, injector=inj, workload=wl, health=health, stats=stats,
                kube_state=lambda start_s=None, end_s=None: kube_state(kube, start_s, end_s),
                collect=collect, deadman=deadman, clock=clock, sleep=clock.sleep)
    return deps, kube, calls, eng


def test_detected_on_third_poll(tmp_path):
    eng = SeqEngine(ok_decision(**QUIET), ok_decision(**QUIET), ok_decision())
    deps, kube, calls, _ = make(tmp_path, eng)
    rec = run_experiment(get_scenario("K001"), deps, CFG, "r1", tmp_path)
    assert rec["status"] == "ok" and len(rec["polls"]) == 3 and rec["scored_poll"] == 3
    t = rec["timing"]
    assert t["t_detect"] == rec["polls"][2]["t_decision"]
    assert t["mttd_seconds"] == pytest.approx(15 + 2 * 15)  # lag, then two 15 s intervals
    assert set(rec["evaluation"].values()) == {"PASS"}
    assert t["t_manifest"] is not None
    assert calls.index("inject") < calls.index("collect") and calls[-1] == "reset"
    assert kube.deps["cart"]["spec"]["replicas"] == 1
    saved = json.loads((tmp_path / "r1" / "K001.json").read_text())
    for k in ("fault", "saved_state", "ground_truth", "placement", "polls", "jev", "evaluation",
              "timing", "usage", "versions", "git_commit", "restored"):
        assert k in saved, k


def test_never_detected_scores_last_poll(tmp_path):
    deps, _, _, eng = make(tmp_path, SeqEngine(ok_decision(**QUIET)))
    scn = get_scenario("K001")
    rec = run_experiment(scn, deps, CFG, "r", tmp_path)
    expected_polls = 1 + int((scn.detection_window_s - CFG.telemetry_lag_s) // scn.poll_interval_s)
    assert len(rec["polls"]) == expected_polls == rec["scored_poll"]
    assert rec["timing"]["t_detect"] is None and rec["timing"]["mttd_seconds"] is None
    assert rec["evaluation"]["detection"] == "FAIL"


def test_missed_incident_still_scores_other_fields(tmp_path):
    missed = ok_decision(incident_detected=False)  # names cart and service_unavailable
    deps, _, _, _ = make(tmp_path, SeqEngine(missed))
    rec = run_experiment(get_scenario("K001"), deps, CFG, "r", tmp_path)
    assert rec["evaluation"] == {"detection": "FAIL", "localization": "PASS", "diagnosis": "PASS",
                                 "action": "PASS"}


def test_control_counts_false_positive_polls(tmp_path):
    scn = get_scenario("K000")
    n = 1 + int((scn.detection_window_s - CFG.telemetry_lag_s) // scn.poll_interval_s)
    seq = [ok_decision(**QUIET)] * n
    seq[3] = ok_decision()  # one false alarm
    deps, _, calls, _ = make(tmp_path, SeqEngine(*seq))
    rec = run_experiment(scn, deps, CFG, "r", tmp_path)
    assert "inject" not in calls and "reset" not in calls
    assert len(rec["polls"]) == n  # polling runs the whole window
    fp = rec["false_positive"]
    assert fp == {"false_positive_polls": 1, "answered_polls": n, "polls": n, "false_positive_run": True}
    assert rec["scored_poll"] == 4 and rec["evaluation"]["detection"] == "FAIL"
    assert rec["timing"]["mttd_seconds"] is None and rec["timing"]["t_inject"] is None


def test_quiet_control_passes(tmp_path):
    deps, _, _, _ = make(tmp_path, SeqEngine(ok_decision(**QUIET)))
    rec = run_experiment(get_scenario("K000"), deps, CFG, "r", tmp_path)
    assert set(rec["evaluation"].values()) == {"PASS"}
    assert rec["false_positive"]["false_positive_run"] is False


def test_engine_error_mid_poll_continues_and_resets(tmp_path):
    eng = SeqEngine(ok_decision(**QUIET), RuntimeError("engine blew up"), ok_decision())
    deps, kube, calls, _ = make(tmp_path, eng)
    rec = run_experiment(get_scenario("K001"), deps, CFG, "r", tmp_path)
    assert rec["status"] == "ok" and rec["polls"][1]["decision"]["status"] == "error"
    assert "engine blew up" in rec["polls"][1]["decision"]["message"]
    assert rec["scored_poll"] == 3 and "reset" in calls
    assert rec["usage"]["engine_calls"] == 3 and rec["usage"]["answered_calls"] == 2


def test_all_polls_failing_is_an_error_run(tmp_path):
    deps, _, calls, _ = make(tmp_path, SeqEngine(Decision(status="error", message="HTTP 500")))
    rec = run_experiment(get_scenario("K001"), deps, CFG, "r", tmp_path)
    assert rec["status"] == "error" and "HTTP 500" in rec["message"]
    assert set(rec["evaluation"].values()) == {"UNKNOWN"} and "reset" in calls


def test_unhealthy_start_injects_nothing(tmp_path):
    deps, _, calls, eng = make(tmp_path, healthy=False)
    rec = run_experiment(get_scenario("K001"), deps, CFG, "r", tmp_path)
    assert rec["status"] == "error" and "inject" not in calls and eng.calls == []
    assert set(rec["evaluation"].values()) == {"UNKNOWN"}


def test_leftovers_stop_the_run(tmp_path):
    kube = FakeKube()
    kube.policies["jev-fault-np"] = {"metadata": {"name": "jev-fault-np"}}
    deps, _, calls, _ = make(tmp_path, kube=kube)
    rec = run_experiment(get_scenario("K001"), deps, CFG, "r", tmp_path)
    assert rec["status"] == "error" and "reset-faults" in rec["message"] and "inject" not in calls


def test_read_back_mismatch_is_error_and_reset_runs(tmp_path):
    kube = FakeKube()
    deps, _, calls, eng = make(tmp_path, kube=kube)
    kube.lying = True
    rec = run_experiment(get_scenario("K001"), deps, CFG, "r", tmp_path)
    assert rec["status"] in ("error", "contaminated") and "read-back" in json.dumps(rec)
    assert "reset" in calls and eng.calls == []
    assert not (tmp_path / ".active-fault.json").exists()


def test_stuck_rollout_is_contaminated_and_stops_batch(tmp_path):
    deps, _, _, _ = make(tmp_path, sticky=True)
    summary = run_batch([get_scenario("K001"), get_scenario("K002")], deps, CFG, tmp_path,
                        repeats=2, run_id="b")
    assert summary["stopped_early"] and len(summary["runs"]) == 1
    rec = json.loads((tmp_path / "b" / "K001.json").read_text())
    assert rec["status"] == "contaminated" and "cart" in rec["message"]
    assert (tmp_path / "b" / "summary.json").exists()


def test_reset_failure_is_contaminated_and_leaves_timer_armed(tmp_path):
    dm = FakeDeadman()
    deps, _, _, _ = make(tmp_path, deadman=dm, reset_exc=RuntimeError("api down"))
    rec = run_experiment(get_scenario("K001"), deps, CFG, "r", tmp_path)
    assert rec["status"] == "contaminated" and "reset failed" in rec["message"]
    assert dm.armed and dm.disarmed == []


def test_deadman_armed_before_inject_and_disarmed_after_reset(tmp_path):
    dm = FakeDeadman()
    deps, _, _, _ = make(tmp_path, deadman=dm)
    scn = get_scenario("K001")
    rec = run_experiment(scn, deps, CFG, "r", tmp_path)
    assert dm.armed == [int(max(scn.duration_s, scn.detection_window_s) + 15 + 120)]
    assert dm.disarmed == ["jev-fault-reset-test"] and rec["deadman_unit"] == "jev-fault-reset-test"


def test_single_mode_makes_one_call(tmp_path):
    deps, _, _, eng = make(tmp_path)
    rec = run_experiment(get_scenario("K001"), deps, Config(**{**CFG.__dict__, "mode": "single"}),
                         "r", tmp_path)
    assert len(eng.calls) == 1 and len(rec["polls"]) == 1 and rec["timing"]["t_manifest"]


def test_placement_recorded(tmp_path):
    deps, kube, _, _ = make(tmp_path)
    rec = run_experiment(get_scenario("K009"), deps, CFG, "r", tmp_path)
    assert rec["placement"] and all(n.startswith("jev-worker") for n in rec["placement"].values())
    assert kube.nodes["jev-worker2"]["spec"].get("taints") == []


@pytest.mark.parametrize("layout", __import__("tier3.layouts", fromlist=["x"]).list_layouts())
@pytest.mark.parametrize("eid", sorted(load_all()))
def test_nothing_leaks_into_any_poll(tmp_path, eid, layout):
    """Whole-path leak test: every scenario, every poll, every layout, contaminated fixtures."""
    from dataclasses import replace

    from jev.client import JevClient, JevConfig
    from tier3.jev_adapter import JevAdapter

    scn = get_scenario(eid)
    deps, _, _, eng = make(tmp_path, SeqEngine(ok_decision(**QUIET)))
    rec = run_experiment(scn, deps, replace(CFG, layout=layout), "r", tmp_path)
    assert rec["observation"]["name"] == layout
    assert len(eng.calls) >= 2, "expected several polls"
    adapter = JevAdapter(JevClient(JevConfig(api_key="k"), post=lambda *a: None))
    question_sets = set()
    for sent_obs, poll in zip(eng.calls, rec["polls"]):
        sent = json.dumps(sent_obs)
        assert sent == json.dumps(poll["request"])
        secrets = {"jev-fault", "feature_flag", "flagd", scn.experiment_id, "ground_truth",
                   "acceptable_actions", "expected", "fault_type", *FAULT_TYPES}
        for s in secrets:
            assert s not in sent, f"{s!r} leaked into poll {poll['n']} of {eid}"
        for label in (scn.expected.diagnosis, *scn.expected.acceptable_actions):
            if label not in ("none", "unknown", "no_action"):
                assert label not in sent, f"expected label {label!r} leaked for {eid}"
        question_sets.add(json.dumps(adapter.build_request(sent_obs)["questions"], sort_keys=True))
    assert len(question_sets) == 1


def test_questions_identical_across_scenarios():
    from jev.client import JevClient, JevConfig
    from tier3.jev_adapter import JevAdapter

    a = JevAdapter(JevClient(JevConfig(api_key="k"), post=lambda *x: None))
    reqs = [a.build_request({"scenario_specific": eid}) for eid in load_all()]
    assert len({json.dumps(r["questions"], sort_keys=True) for r in reqs}) == 1


def test_key_never_in_result(tmp_path):
    from jev.client import JevClient, JevConfig
    from tier3.jev_adapter import JevAdapter

    class R:
        status_code, text = 200, "{}"

        def json(self):
            return {"answers": {"incident_detected": {"noul": 0.9},
                                "affected_component": {"choice": "cart"},
                                "diagnosis": {"choice": "service_unavailable"},
                                "recommended_action": {"choice": "scale_up"}}}

    key = "sk-super-secret-123"
    eng = JevAdapter(JevClient(JevConfig(api_key=key), post=lambda *a: R()))
    deps, _, _, _ = make(tmp_path, engine=eng)
    run_experiment(get_scenario("K001"), deps, CFG, "k", tmp_path)
    assert key not in (tmp_path / "k" / "K001.json").read_text()


def test_batch_calls_between_runs_after_each_run(tmp_path):
    deps, _, calls, _ = make(tmp_path)
    deps.between_runs = lambda: calls.append("between")
    summary = run_batch([get_scenario("K000"), get_scenario("K001")], deps, CFG, tmp_path,
                        repeats=1, run_id="b")
    assert len(summary["runs"]) == 2 and calls.count("between") == 2
    assert calls.index("between") > calls.index("health")


def test_run_that_never_called_the_engine_uses_zero_tokens(tmp_path):
    deps, _, _, eng = make(tmp_path, healthy=False)
    rec = run_experiment(get_scenario("K001"), deps, CFG, "r", tmp_path)
    assert rec["status"] == "error" and eng.calls == []
    assert rec["usage"]["input_tokens"] == 0 and rec["usage"]["cost_usd"] == 0
