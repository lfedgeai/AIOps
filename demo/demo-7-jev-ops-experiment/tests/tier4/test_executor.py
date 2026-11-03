import json

import pytest

from tests.tier3.helpers import FakeKube
from tier3.faults import FaultInjector, reset_faults
from tier3.scenarios import get_scenario
from tier4.executor import RESTART_ANNOTATION, Executor, ExecutorError

BASE = {"cart": 1, "frontend": 1}


def ex(tmp_path, kube=None, undo=None):
    kube = kube or FakeKube()
    return Executor(kube, ledger=tmp_path / ".active-fault.json", undo=undo,
                    clock=lambda: 1_800_000_000.0, sleep=lambda s: None), kube


def test_scale_up_to_baseline(tmp_path):
    e, kube = ex(tmp_path, FakeKube({"cart": 0}))
    out = e.execute("scale_up", "cart", BASE)
    assert out["executed"] and out["operation"] == "scale cart 0 -> 1"
    assert kube.deployment("cart")["spec"]["replicas"] == 1 and out["t_action_done"]


def test_scale_up_healthy_adds_one(tmp_path):
    e, kube = ex(tmp_path)
    assert e.execute("scale_up", "frontend", BASE)["executed"]
    assert kube.deployment("frontend")["spec"]["replicas"] == 2


def test_scale_up_cap(tmp_path):
    e, kube = ex(tmp_path, FakeKube({"frontend": 3}))
    out = e.execute("scale_up", "frontend", BASE)
    assert not out["executed"] and out["not_executable"] == "replica cap"
    assert kube.calls == [] and not (tmp_path / ".active-fault.json").exists()


def test_restart_sets_annotation(tmp_path):
    e, kube = ex(tmp_path)
    out = e.execute("restart_service", "cart", BASE)
    ann = kube.deployment("cart")["spec"]["template"]["metadata"]["annotations"]
    assert out["executed"] and out["rollout_complete"] and RESTART_ANNOTATION in ann


@pytest.mark.parametrize("action", ["investigate", "escalate", "no_action"])
def test_not_acted(tmp_path, action):
    e, kube = ex(tmp_path)
    out = e.execute(action, "cart", BASE)
    assert not out["executed"] and out["not_executable"] == "not auto-remediated" and kube.calls == []


@pytest.mark.parametrize("component, reason", [
    ("jev-worker2", "not a Deployment (a node)"), ("load-generator", "infrastructure"),
    ("none", "no component named"), (None, "no component named"),
    ("nosuch", "not a Deployment in the namespace")])
def test_refusals(tmp_path, component, reason):
    e, kube = ex(tmp_path)
    out = e.execute("restart_service", component, BASE)
    assert out["not_executable"] == reason and kube.calls == []


def test_second_call_refused(tmp_path):
    e, _ = ex(tmp_path)
    e.execute("no_action", "none", BASE)
    with pytest.raises(ExecutorError):
        e.execute("scale_up", "cart", BASE)


def test_ledger_written_before_acting(tmp_path):
    seen = {}
    kube = FakeKube()
    orig = kube.scale

    def scale(name, n):
        seen["ledger"] = json.loads((tmp_path / ".active-fault.json").read_text())
        orig(name, n)
    kube.scale = scale
    e, _ = ex(tmp_path, kube)
    e.execute("scale_up", "frontend", BASE)
    assert seen["ledger"]["action"]["saved"]["replicas"] == {"frontend": 1}


def test_read_back_mismatch_raises(tmp_path):
    kube = FakeKube()
    kube.lying = True
    e, _ = ex(tmp_path, kube)
    with pytest.raises(ExecutorError, match="read-back"):
        e.execute("scale_up", "frontend", BASE)
    kube.lying = False
    assert e.restore() and kube.deployment("frontend")["spec"]["replicas"] == 1


def test_rollback_uses_undo_only(tmp_path):
    calls = []
    e, _ = ex(tmp_path, undo=lambda n: (calls.append(n), (0, "deployment.apps/cart rolled back"))[1])
    out = e.execute("rollback_change", "cart", BASE)
    assert out["executed"] and calls == ["cart"] and "rolled back" in out["undo_output"]


def test_rollback_without_history(tmp_path):
    e, _ = ex(tmp_path, undo=lambda n: (1, 'error: no rollout history found for deployment "cart"'))
    out = e.execute("rollback_change", "cart", BASE)
    assert not out["executed"] and out["not_executable"] == "no previous revision"
    assert not (tmp_path / ".active-fault.json").exists()


def test_restore_action_then_fault_same_deployment(tmp_path):
    kube = FakeKube()
    original = kube.deployment("cart")["spec"]["template"]
    led = tmp_path / ".active-fault.json"
    inj = FaultInjector(kube, ledger=led)
    inj.inject(get_scenario("K003").fault)  # memory_limit on cart
    e = Executor(kube, ledger=led, clock=lambda: 1.0, sleep=lambda s: None)
    e.execute("restart_service", "cart", BASE)
    assert e.restore() and "action" not in json.loads(led.read_text())
    inj.reset()
    assert kube.deployment("cart")["spec"]["template"] == original and not led.exists()


def test_crash_recovery_from_ledger_alone(tmp_path):
    kube = FakeKube()
    led = tmp_path / ".active-fault.json"
    FaultInjector(kube, ledger=led).inject(get_scenario("K001").fault)  # cart -> 0
    Executor(kube, ledger=led, clock=lambda: 1.0, sleep=lambda s: None).execute("scale_up", "frontend", BASE)
    changed = reset_faults(kube, ledger=led)  # a fresh process: no in-memory state
    assert kube.deployment("frontend")["spec"]["replicas"] == 1
    assert kube.deployment("cart")["spec"]["replicas"] == 1 and not led.exists()
    assert any("frontend replicas -> 1" in c for c in changed)


def test_action_only_ledger_is_removed_on_restore(tmp_path):
    e, kube = ex(tmp_path)
    e.execute("scale_up", "frontend", BASE)
    e.restore()
    assert not (tmp_path / ".active-fault.json").exists()
    assert kube.deployment("frontend")["spec"]["replicas"] == 1
