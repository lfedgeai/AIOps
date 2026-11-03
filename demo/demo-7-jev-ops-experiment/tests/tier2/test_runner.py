import copy
import json

import pytest

from tests.tier2.helpers import (FLAGS, FakeEngine, FakeFlagStore, contaminated_telemetry,
                                 ok_decision)
from tier2.engine import Decision
from tier2.faults import FaultInjector, effective_variant
from tier2.observation import ServiceStats
from tier2.runner import Config, Deps, run_batch, run_experiment
from tier2.scenarios import get_scenario, load_all
from tier2.workload import Workload

BASE = {"payment": ServiceStats(0.1, 0.0, 40, 80), "checkout": ServiceStats(0.3, 0.0, 50, 200)}
BAD = {"payment": ServiceStats(0.1, 1.0, 40, 90), "checkout": ServiceStats(0.3, 0.6, 50, 3000)}


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0
        self.slept = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s
        self.slept += s


def make(engine=None, healthy=True, sticky=False, collect_exc=None, running=True):
    clock, calls = Clock(), []
    cfg = copy.deepcopy(FLAGS)
    for sc in load_all().values():  # the fake store knows every flag the catalogue uses
        if sc.fault and sc.fault.flag not in cfg["flags"]:
            cfg["flags"][sc.fault.flag] = {"state": "ENABLED", "defaultVariant": "off",
                                           "variants": {"off": 0, sc.fault.variant: 1}}
        elif sc.fault:
            cfg["flags"][sc.fault.flag]["variants"].setdefault(sc.fault.variant, 1)
    store = FakeFlagStore(cfg)
    inj = FaultInjector(store)
    orig_inject, orig_reset = inj.inject, inj.reset
    inj.inject = lambda f: (calls.append("inject"), orig_inject(f))[1]
    inj.reset = lambda: (calls.append("reset"), orig_reset())[1]
    eng = engine or FakeEngine()
    orig_decide = eng.decide
    eng.decide = lambda o: (calls.append("decide"), orig_decide(o))[1]

    def health():
        calls.append("health")
        if not healthy:
            raise RuntimeError("prometheus not ready")

    def stats(at, window):
        calls.append("stats")
        faulted = bool(inj.active()) or (sticky and "inject" in calls)
        return dict(BAD if faulted else BASE)

    def collect(**kw):
        calls.append("collect")
        if collect_exc:
            raise collect_exc
        return contaminated_telemetry()

    wl = Workload(get=lambda u: {"state": "running" if running else "stopped", "user_count": 10},
                  post=lambda u, d: {"success": True})
    deps = Deps(engine=eng, injector=inj, workload=wl, health=health, stats=stats, collect=collect,
                clock=clock, sleep=clock.sleep)
    return deps, store, calls, eng


CFG = Config(manifest_poll_s=5, manifest_timeout_s=30, recover_timeout_s=30, recover_poll_s=5,
             scrub_terms=("cartFailure", "paymentFailure", "productCatalogFailure", "flagd", "F003"))


def test_step_order_and_result_contents(tmp_path):
    deps, store, calls, _ = make()
    rec = run_experiment(get_scenario("F003"), deps, CFG, "run1", tmp_path)
    order = [c for c in calls if c in ("health", "inject", "collect", "decide", "reset")]
    assert order[:5] == ["health", "inject", "collect", "decide", "reset"]
    assert rec["status"] == "ok"
    for k in ("experiment_id", "fault", "ground_truth", "jev", "evaluation", "timing", "versions",
              "git_commit", "trim_notes", "request"):
        assert k in rec
    assert set(rec["evaluation"]) == {"detection", "localization", "diagnosis", "action"}
    assert rec["jev"]["raw_response"] == {"answers": {}}
    assert rec["timing"]["t_manifest"] is not None and rec["timing"]["detection_seconds"] is not None
    saved = json.loads((tmp_path / "run1" / "F003.json").read_text())
    assert saved["experiment_id"] == "F003"
    assert all(effective_variant(f) == "off" for f in store.read()["flags"].values())


def test_unhealthy_start_injects_nothing(tmp_path):
    deps, store, calls, eng = make(healthy=False)
    rec = run_experiment(get_scenario("F003"), deps, CFG, "r", tmp_path)
    assert rec["status"] == "error" and "prometheus not ready" in rec["message"]
    assert "inject" not in calls and eng.calls == []
    assert set(rec["evaluation"].values()) == {"UNKNOWN"}


def test_workload_not_running_stops_before_injection(tmp_path):
    deps, _, calls, _ = make(running=False)
    rec = run_experiment(get_scenario("F003"), deps, CFG, "r", tmp_path)
    assert rec["status"] == "error" and "inject" not in calls


def test_leftover_fault_stops_run(tmp_path):
    deps, store, calls, _ = make()
    FaultInjector(store).inject(get_scenario("F003").fault)
    rec = run_experiment(get_scenario("F001"), deps, CFG, "r", tmp_path)
    assert rec["status"] == "error" and "already active" in rec["message"]
    assert "inject" not in calls


def test_reset_runs_when_engine_raises(tmp_path):
    deps, store, calls, _ = make(engine=FakeEngine(exc=RuntimeError("engine blew up")))
    rec = run_experiment(get_scenario("F003"), deps, CFG, "r", tmp_path)
    assert rec["status"] == "error" and "engine blew up" in rec["message"]
    assert "reset" in calls
    assert all(effective_variant(f) == "off" for f in store.read()["flags"].values())


def test_reset_runs_when_collector_fails(tmp_path):
    deps, store, calls, eng = make(collect_exc=ConnectionError("opensearch down"))
    rec = run_experiment(get_scenario("F003"), deps, CFG, "r", tmp_path)
    assert rec["status"] == "error" and "opensearch down" in rec["message"]
    assert "reset" in calls and eng.calls == []


def test_not_recovered_is_contaminated(tmp_path):
    deps, _, _, _ = make(sticky=True)
    rec = run_experiment(get_scenario("F003"), deps, CFG, "r", tmp_path)
    assert rec["status"] == "contaminated"


def test_batch_stops_on_contamination(tmp_path):
    deps, _, _, eng = make(sticky=True)
    scns = [get_scenario("F003"), get_scenario("F001")]
    summary = run_batch(scns, deps, CFG, tmp_path, repeats=2, run_id="b")
    assert summary["stopped_early"] and len(summary["runs"]) == 1
    assert (tmp_path / "b" / "summary.json").exists()


def test_control_runs_same_path_without_injection(tmp_path):
    quiet = ok_decision(incident_detected=False, affected_service="none", diagnosis="none",
                        recommended_action="no_action")
    deps, _, calls, eng = make(engine=FakeEngine(quiet))
    rec = run_experiment(get_scenario("F000"), deps, CFG, "r", tmp_path)
    assert "inject" not in calls and "reset" not in calls
    assert calls.count("collect") == 1 and calls.count("decide") == 1
    assert rec["timing"]["t_inject"] is None and rec["timing"]["t_manifest"] is None
    assert set(rec["evaluation"].values()) == {"PASS"} and rec["false_positive"] is False


def test_control_false_positive_recorded(tmp_path):
    deps, _, _, _ = make(engine=FakeEngine(ok_decision(incident_detected=True)))
    rec = run_experiment(get_scenario("F000"), deps, CFG, "r", tmp_path)
    assert rec["false_positive"] is True and rec["evaluation"]["detection"] == "FAIL"


def test_invalid_reply_keeps_raw_and_unknown(tmp_path):
    bad = Decision(status="invalid", raw_response={"answers": {"diagnosis": {"choice": "gremlins"}}},
                   message="out-of-set")
    deps, _, _, _ = make(engine=FakeEngine(bad))
    rec = run_experiment(get_scenario("F003"), deps, CFG, "r", tmp_path)
    assert rec["jev"]["status"] == "invalid" and "gremlins" in json.dumps(rec["jev"]["raw_response"])
    assert set(rec["evaluation"].values()) == {"UNKNOWN"}


def test_repeats_write_separate_files(tmp_path):
    deps, _, _, _ = make()
    run_batch([get_scenario("F003")], deps, CFG, tmp_path, repeats=3, run_id="rep")
    files = sorted(p.name for p in (tmp_path / "rep").glob("F003*.json"))
    assert files == ["F003-2.json", "F003-3.json", "F003.json"]


@pytest.mark.parametrize("eid", sorted(load_all()))
def test_ground_truth_and_fault_metadata_never_in_request(tmp_path, eid):
    """Leak test over the whole catalogue: what the engine receives has no answer."""
    scn = get_scenario(eid)
    deps, _, _, eng = make()
    rec = run_experiment(scn, deps, CFG, "r", tmp_path)
    assert len(eng.calls) == 1
    sent = json.dumps(eng.calls[0])
    assert sent == json.dumps(rec["request"])
    secrets = {"feature_flag", "flagd", scn.experiment_id, "ground_truth", "acceptable_actions",
               "expected", "fault_type"}
    if scn.fault:
        secrets.add(scn.fault.flag)
    for s in secrets:
        assert s not in sent, f"{s!r} leaked into the engine input for {eid}"
    # eng.calls holds the state only; the question vocabulary is sent separately and is
    # the same for every scenario, so expected labels must not appear in the state.
    for label in (scn.expected.diagnosis, *scn.expected.acceptable_actions):
        if label not in ("none", "unknown", "no_action"):
            assert label not in sent, f"expected label {label!r} leaked for {eid}"


def test_questions_identical_across_scenarios():
    from jev.client import JevClient, JevConfig
    from tier2.jev_adapter import JevAdapter

    a = JevAdapter(JevClient(JevConfig(api_key="k"), post=lambda *x: None))
    reqs = [a.build_request({"scenario_specific": eid}) for eid in load_all()]
    assert len({json.dumps(r["questions"], sort_keys=True) for r in reqs}) == 1


def test_key_never_in_result(tmp_path):
    from jev.client import JevClient, JevConfig
    from tier2.jev_adapter import JevAdapter

    class R:
        status_code, text = 200, "{}"
        def json(self):
            return {"answers": {"incident_detected": {"noul": 0.9},
                                "affected_service": {"choice": "payment"},
                                "diagnosis": {"choice": "application_failure"},
                                "recommended_action": {"choice": "investigate"}}}

    key = "sk-super-secret-123"
    eng = JevAdapter(JevClient(JevConfig(api_key=key), post=lambda *a: R()))
    deps, _, _, _ = make(engine=eng)
    run_experiment(get_scenario("F003"), deps, CFG, "k", tmp_path)
    assert key not in (tmp_path / "k" / "F003.json").read_text()
