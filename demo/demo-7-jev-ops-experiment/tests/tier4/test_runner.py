import json

import pytest

from tests.tier3.helpers import QUIET, FakeDeadman, FakeKube, SeqEngine, ok_decision, telemetry
from tier2.observation import ServiceStats
from tier2.workload import Workload
from tier3.collect import kube_state
from tier3.faults import MEMORY_LIMIT, FaultInjector
from tier3.runner import Config, Deps
from tier3.scenarios import get_scenario, load_all
from tier3.vocab import FAULT_TYPES
from tier4.executor import Executor
from tier4.recovery import NOT_RECOVERED, RECOVERED_BY_ACTION, RECOVERED_WITHOUT_ACTION, RecoveryConfig
from tier4.runner import Tier4Deps, run_batch, run_experiment

GOOD = {"frontend": ServiceStats(5.0, 0.0, 20, 250), "checkout": ServiceStats(0.3, 0.0, 50, 200)}
BAD = {"frontend": ServiceStats(5.0, 0.3, 20, 300), "checkout": ServiceStats(0.3, 0.5, 50, 900)}
CFG = Config(telemetry_lag_s=15, recover_timeout_s=30, recover_poll_s=10, scrub_terms=("flagd", *FAULT_TYPES))


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += max(s, 0)


def broken(kube) -> bool:
    """The simulated service is down while an app Deployment is at zero or cart has the 10Mi limit."""
    if any(d["spec"]["replicas"] == 0 for n, d in kube.deps.items() if n != "load-generator"):
        return True
    c = kube.deps["cart"]["spec"]["template"]["spec"]["containers"][0]
    return c.get("resources", {}).get("limits", {}).get("memory") == MEMORY_LIMIT


def make(tmp_path, engine, kube=None, deadman=None, undo=None):
    clock, kube = Clock(), kube or FakeKube()
    led = tmp_path / ".active-fault.json"
    inj = FaultInjector(kube, ledger=led)
    wl = Workload(get=lambda u: {"state": "running", "user_count": 10}, post=lambda u, d: {"success": True})
    base = Deps(engine=engine, injector=inj, workload=wl, health=lambda: None,
                stats=lambda at, w: dict(BAD if broken(kube) else GOOD),
                kube_state=lambda start_s=None, end_s=None: kube_state(kube, start_s, end_s),
                collect=lambda **kw: (telemetry(broken(kube)), kube_state(kube, 0, 4e9)),
                deadman=deadman, clock=clock, sleep=clock.sleep)
    deps = Tier4Deps(base=base, recovery=RecoveryConfig(),
                     executor=lambda: Executor(kube, ledger=led, undo=undo, clock=clock, sleep=clock.sleep))
    return deps, kube, led


def test_right_action_recovers(tmp_path):
    deps, kube, led = make(tmp_path, SeqEngine(ok_decision(recommended_action="scale_up", affected_component="cart")))
    rec = run_experiment(get_scenario("K001"), deps, CFG, "r", tmp_path)
    rem = rec["remediation"]
    assert rec["status"] == "ok" and rem["outcome"] == RECOVERED_BY_ACTION and rem["recovery_verified"]
    t = rem["timing"]
    assert t["mttr_seconds"] is not None and t["mttr_verified_seconds"] >= t["mttr_seconds"]
    assert t["time_to_remediate_seconds"] is not None and not led.exists()
    assert len(rec["polls"]) == 1  # no Jev call after acting


def test_wrong_component_does_not_recover_and_is_restored(tmp_path):
    deps, kube, led = make(tmp_path, SeqEngine(ok_decision(recommended_action="scale_up",
                                                           affected_component="frontend")))
    rec = run_experiment(get_scenario("K001"), deps, CFG, "r", tmp_path)
    rem = rec["remediation"]
    assert rem["outcome"] == NOT_RECOVERED and not rem["recovery_verified"]
    assert rem["timing"]["mttr_seconds"] is not None and rem["timing"]["mttr_verified_seconds"] is None
    assert kube.deployment("frontend")["spec"]["replicas"] == 1
    assert kube.deployment("cart")["spec"]["replicas"] == 1 and not led.exists()


def test_rollback_fixes_template_fault(tmp_path):
    kube = FakeKube()
    before = {}

    def undo(name):  # stands in for kubectl: the previous revision's template
        kube.replace_template(name, before["tpl"])
        return 0, f"deployment.apps/{name} rolled back"
    deps, kube, led = make(tmp_path, SeqEngine(ok_decision(recommended_action="rollback_change",
                                                           diagnosis="oom_killed")), kube=kube, undo=undo)
    before["tpl"] = kube.deployment("cart")["spec"]["template"]
    rec = run_experiment(get_scenario("K003"), deps, CFG, "r", tmp_path)
    assert rec["remediation"]["outcome"] == RECOVERED_BY_ACTION and not led.exists()
    assert kube.deployment("cart")["spec"]["template"] == before["tpl"]


def test_investigate_is_not_auto_remediated(tmp_path):
    deps, kube, _ = make(tmp_path, SeqEngine(ok_decision(recommended_action="investigate")))
    rec = run_experiment(get_scenario("K001"), deps, CFG, "r", tmp_path)
    rem = rec["remediation"]
    assert rem["action"]["not_executable"] == "not auto-remediated" and rem["outcome"] == NOT_RECOVERED
    assert rem["timing"]["mttr_seconds"] is None


def test_never_detected_runs_no_action(tmp_path):
    deps, kube, _ = make(tmp_path, SeqEngine(ok_decision(**QUIET)))
    rec = run_experiment(get_scenario("K001"), deps, CFG, "r", tmp_path)
    assert rec["remediation"]["action"]["not_executable"] == "no incident reported"
    assert rec["remediation"]["outcome"] == NOT_RECOVERED and rec["status"] == "ok"


def test_false_alarm_action_on_control_is_recorded_and_restored(tmp_path):
    dm = FakeDeadman()
    deps, kube, led = make(tmp_path, SeqEngine(ok_decision(recommended_action="scale_up",
                                                           affected_component="frontend")), deadman=dm)
    rec = run_experiment(get_scenario("K000"), deps, CFG, "r", tmp_path)
    rem = rec["remediation"]
    assert rem["action_on_false_alarm"] and rem["remediation_executed"]
    assert rem["outcome"] == RECOVERED_WITHOUT_ACTION  # healthy before and after
    assert dm.armed and dm.disarmed == ["jev-fault-reset-test"]
    assert kube.deployment("frontend")["spec"]["replicas"] == 1 and not led.exists()
    assert len(rec["polls"]) == 1  # stopped at the first detection


def test_action_failure_is_error_and_still_reset(tmp_path):
    kube = FakeKube()
    deps, kube, led = make(tmp_path, SeqEngine(ok_decision(recommended_action="rollback_change")),
                           kube=kube, undo=lambda n: (1, "error: the server is down"))
    rec = run_experiment(get_scenario("K001"), deps, CFG, "r", tmp_path)
    assert rec["status"] == "error" and "rollout undo" in rec["message"]
    assert kube.deployment("cart")["spec"]["replicas"] == 1 and not led.exists()


def test_batch_gets_a_fresh_executor_each_run(tmp_path):
    deps, kube, _ = make(tmp_path, SeqEngine(ok_decision(recommended_action="scale_up", affected_component="cart")))
    s = run_batch([get_scenario("K001")], deps, CFG, tmp_path, repeats=2, run_id="b")
    assert [r["outcome"] for r in s["runs"]] == [RECOVERED_BY_ACTION] * 2
    assert json.loads((tmp_path / "b" / "summary.json").read_text())["scorecard"]["overall"]


@pytest.mark.parametrize("layout", __import__("tier3.layouts", fromlist=["x"]).list_layouts())
@pytest.mark.parametrize("eid", sorted(load_all()))
def test_whole_path_leaks_nothing(tmp_path, eid, layout):
    from dataclasses import replace
    deps, kube, _ = make(tmp_path, SeqEngine(ok_decision(**QUIET), ok_decision()))
    rec = run_experiment(get_scenario(eid), deps, replace(CFG, layout=layout), "r", tmp_path)
    assert rec["observation"]["name"] == layout
    assert "remediation" in rec
    for p in rec["polls"]:
        text = json.dumps(p["request"]).lower()
        for term in (*FAULT_TYPES, "jev-fault", eid.lower(), "recovered_by_action", "not_recovered",
                     "remediation", "scale cart", "ground_truth"):
            assert term not in text, (eid, term)
