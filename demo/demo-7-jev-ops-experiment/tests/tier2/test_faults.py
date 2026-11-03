import pytest

from tests.tier2.helpers import FakeFlagStore, LyingFlagStore
from tier2.faults import (FaultError, FaultInjector, effective_variant, reset_faults)
from tier2.scenarios import FaultSpec
from tier2.workload import Workload, WorkloadError


def test_inject_then_reset_restores_exactly():
    store = FakeFlagStore()
    before = store.read()
    inj = FaultInjector(store)
    inj.inject(FaultSpec("cartFailure", "100%"))
    assert effective_variant(store.read()["flags"]["cartFailure"]) == "100%"
    assert inj.active() == {"cartFailure": "100%"}
    assert inj.reset() == ["cartFailure"]
    assert store.read() == before


def test_targeting_flag_sets_branch_not_default():
    store = FakeFlagStore()
    inj = FaultInjector(store)
    inj.inject(FaultSpec("productCatalogFailure", "on"))
    flag = store.read()["flags"]["productCatalogFailure"]
    assert flag["targeting"]["if"][1] == "on"
    assert flag["defaultVariant"] == "off"  # targeting overrides the default
    assert inj.active() == {"productCatalogFailure": "on"}
    inj.reset()
    assert store.read()["flags"]["productCatalogFailure"]["targeting"]["if"][1] == "off"


def test_unknown_flag_and_variant():
    inj = FaultInjector(FakeFlagStore())
    with pytest.raises(FaultError, match="nope"):
        inj.inject(FaultSpec("nope", "on"))
    with pytest.raises(FaultError, match="200%"):
        inj.inject(FaultSpec("cartFailure", "200%"))


def test_read_back_mismatch_raises():
    inj = FaultInjector(LyingFlagStore())
    with pytest.raises(FaultError, match="read-back mismatch"):
        inj.inject(FaultSpec("cartFailure", "100%"))


def test_reset_without_inject_is_noop():
    store = FakeFlagStore()
    assert FaultInjector(store).reset() == []
    assert store.writes == 0


def test_reset_faults_idempotent_and_global():
    store = FakeFlagStore()
    assert reset_faults(store) == []
    assert store.writes == 0  # clean system: nothing written
    FaultInjector(store).inject(FaultSpec("paymentFailure", "100%"))
    assert reset_faults(store) == ["paymentFailure"]
    assert reset_faults(store) == []


def test_workload_require_running():
    ok = Workload(get=lambda u: {"state": "running", "user_count": 5})
    assert ok.require_running() == 5
    for bad in ({"state": "stopped", "user_count": 0}, {"state": "running", "user_count": 0}):
        with pytest.raises(WorkloadError, match="not running"):
            Workload(get=lambda u, b=bad: b).require_running()


def test_workload_set_users():
    calls = []
    w = Workload(post=lambda u, d: calls.append((u, d)) or {"success": True})
    w.set_users(30)
    assert calls[0][0].endswith("/loadgen/swarm") and calls[0][1]["user_count"] == 30
    with pytest.raises(WorkloadError):
        Workload(post=lambda u, d: {"success": False}).set_users(1)
