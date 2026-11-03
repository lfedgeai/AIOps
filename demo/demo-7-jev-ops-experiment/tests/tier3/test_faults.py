import json

import pytest

from tests.tier3.helpers import FakeKube
from tier3.deadman import Deadman
from tier3.faults import (BAD_DB_HOST, PREFIX, FaultError, FaultInjector, reset_faults)
from tier3.scenarios import FaultSpec, load_all
from tier3.vocab import FAULT_TYPES

CATALOGUE = {s.fault.type: s.fault for s in load_all().values() if s.fault}


def injector(tmp_path, kube=None):
    kube = kube or FakeKube()
    return kube, FaultInjector(kube, ledger=tmp_path / ".active-fault.json")


@pytest.mark.parametrize("ftype", FAULT_TYPES)
def test_round_trip_restores_exactly(tmp_path, ftype):
    kube, inj = injector(tmp_path)
    before = kube.snapshot()
    saved = inj.inject(CATALOGUE[ftype])
    assert inj.ledger.exists() and saved
    if ftype != "kill_pod":
        assert kube.snapshot() != before, f"{ftype} changed nothing"
    inj.reset()
    assert kube.snapshot() == before
    assert not inj.ledger.exists()
    assert inj.leftovers() == []


def test_two_replicas_restored_to_two(tmp_path):
    kube, inj = injector(tmp_path, FakeKube(replicas={"cart": 2}))
    inj.inject(FaultSpec("scale_zero", "cart"))
    assert kube.deps["cart"]["spec"]["replicas"] == 0
    inj.reset()
    assert kube.deps["cart"]["spec"]["replicas"] == 2


def test_replica_overload_restores_both_counts(tmp_path):
    kube, inj = injector(tmp_path)
    inj.inject(FaultSpec("replica_overload", "cart"))
    assert kube.deps["load-generator"]["spec"]["replicas"] == 3
    assert kube.deps["cart"]["spec"]["replicas"] == 0
    inj.reset()
    assert kube.deps["load-generator"]["spec"]["replicas"] == 1
    assert kube.deps["cart"]["spec"]["replicas"] == 1


def test_other_taints_are_kept(tmp_path):
    kube, inj = injector(tmp_path)
    other = {"key": "dedicated", "value": "gpu", "effect": "NoSchedule"}
    kube.nodes["jev-worker2"]["spec"]["taints"] = [other]
    inj.inject(FaultSpec("node_taint", "jev-worker2"))
    keys = [t["key"] for t in kube.nodes["jev-worker2"]["spec"]["taints"]]
    assert keys == ["dedicated", PREFIX]
    inj.reset()
    assert kube.nodes["jev-worker2"]["spec"]["taints"] == [other]


def test_config_corruption_only_adds_database_host(tmp_path):
    kube, inj = injector(tmp_path)
    before = json.loads(json.dumps(kube.deps["cart"]["spec"]["template"]))
    inj.inject(FaultSpec("config_corruption", "cart"))
    after = kube.deps["cart"]["spec"]["template"]
    env = after["spec"]["containers"][0]["env"]
    assert env[-1] == {"name": "DATABASE_HOST", "value": BAD_DB_HOST}
    after["spec"]["containers"][0]["env"] = env[:-1]
    assert after == before


@pytest.mark.parametrize("ftype", FAULT_TYPES)
def test_no_self_describing_marker(tmp_path, ftype):
    kube, inj = injector(tmp_path)
    inj.inject(CATALOGUE[ftype])
    cluster = json.dumps({"deps": kube.list_deployments(), "pods": kube.list_pods(),
                          "nodes": kube.list_nodes(), "policies": kube.list_network_policies()})
    for marker in ("HARNESS", "harness-", *FAULT_TYPES):
        assert marker not in cluster, f"{ftype} left {marker!r} in the cluster"


def test_read_back_mismatch_raises_and_reset_still_works(tmp_path):
    kube, inj = injector(tmp_path)
    kube.lying = True
    with pytest.raises(FaultError, match="read-back mismatch"):
        inj.inject(FaultSpec("scale_zero", "cart"))
    kube.lying = False
    assert inj.ledger.exists()  # saved before the change, so reset has what it needs
    inj.reset()
    assert not inj.ledger.exists()


def test_inject_refuses_while_a_ledger_exists(tmp_path):
    kube, inj = injector(tmp_path)
    inj.inject(FaultSpec("scale_zero", "cart"))
    other = FaultInjector(kube, ledger=inj.ledger)
    with pytest.raises(FaultError, match="already active"):
        other.inject(FaultSpec("kill_pod", "cart"))


def test_pvc_full_write_failure_raises(tmp_path):
    kube, inj = injector(tmp_path)
    kube.exec_rc = 127
    with pytest.raises(FaultError, match="write failed"):
        inj.inject(FaultSpec("pvc_full", "cart"))


def test_pvc_full_write_killed_by_oom_counts_as_applied(tmp_path, monkeypatch):
    kube, inj = injector(tmp_path)

    def killed(pod, command):
        kube.pods[pod]["status"]["containerStatuses"][0]["restartCount"] += 1
        return 137, "command terminated with exit code 137"
    monkeypatch.setattr(kube, "exec", killed)
    saved = inj.inject(FaultSpec("pvc_full", "cart"))
    assert inj.ledger.exists() and saved["fill"]["restarts"] == 0
    assert "restarted" in inj.reset()[0]


def test_pvc_full_exit_137_without_restart_still_fails(tmp_path, monkeypatch):
    kube, inj = injector(tmp_path)
    monkeypatch.setattr("tier3.faults._sleep", lambda s: None)
    monkeypatch.setattr(kube, "exec", lambda pod, command: (137, "killed"))
    with pytest.raises(FaultError, match="write failed"):
        inj.inject(FaultSpec("pvc_full", "cart"))


def test_pvc_full_waits_for_the_restart_to_be_reported(tmp_path, monkeypatch):
    kube, inj = injector(tmp_path)
    pods = {}

    def killed(pod, command):
        pods["name"] = pod
        return 137, "command terminated with exit code 137"

    def late_report(s):  # the kubelet reports the restart after the exec returns
        kube.pods[pods["name"]]["status"]["containerStatuses"][0]["restartCount"] = 1
    monkeypatch.setattr(kube, "exec", killed)
    monkeypatch.setattr("tier3.faults._sleep", late_report)
    inj.inject(FaultSpec("pvc_full", "cart"))
    assert "restarted" in inj.reset()[0]


def test_pvc_full_restore_skips_rm_after_container_restart(tmp_path):
    kube, inj = injector(tmp_path)
    saved = inj.inject(FaultSpec("pvc_full", "cart"))
    pod = kube.pods[saved["fill"]["pod"]]
    pod["status"]["containerStatuses"][0]["restartCount"] += 1  # OOMKilled by the write
    kube.exec_rc = 1  # exec would fail while the container restarts
    assert "restarted" in inj.reset()[0]
    assert not inj.ledger.exists()


def test_kill_pod_replaces_every_pod(tmp_path):
    kube, inj = injector(tmp_path, FakeKube(replicas={"cart": 2}))
    old = {p["metadata"]["name"] for p in kube.list_pods({"opentelemetry.io/name": "cart"})}
    saved = inj.inject(FaultSpec("kill_pod", "cart"))
    new = {p["metadata"]["name"] for p in kube.list_pods({"opentelemetry.io/name": "cart"})}
    assert set(saved["killed_pods"]) == old and not (old & new) and len(new) == 2


def test_reset_faults_restores_from_ledger_after_harness_death(tmp_path):
    kube, inj = injector(tmp_path, FakeKube(replicas={"cart": 2}))
    before = kube.snapshot()
    inj.inject(FaultSpec("memory_limit", "cart"))
    del inj  # the harness died; only the ledger is left
    changed = reset_faults(kube, ledger=tmp_path / ".active-fault.json")
    assert any("pod template of cart" in c for c in changed)
    assert kube.snapshot() == before and not (tmp_path / ".active-fault.json").exists()


def test_reset_faults_removes_stray_objects_and_is_idempotent(tmp_path):
    kube = FakeKube()
    kube.policies["jev-fault-np"] = {"metadata": {"name": "jev-fault-np"}}
    kube.policies["keep-me"] = {"metadata": {"name": "keep-me"}}
    kube.nodes["jev-worker3"]["spec"]["taints"] = [{"key": PREFIX, "effect": "NoExecute"}]
    changed = reset_faults(kube, ledger=tmp_path / "none.json")
    assert len(changed) == 2 and sorted(kube.policies) == ["keep-me"]
    assert kube.nodes["jev-worker3"]["spec"]["taints"] == []
    assert reset_faults(kube, ledger=tmp_path / "none.json") == []


def test_clean_cluster_reports_no_changes(tmp_path):
    assert reset_faults(FakeKube(), ledger=tmp_path / "none.json") == []


def test_leftovers_name_each_object(tmp_path):
    kube, inj = injector(tmp_path)
    inj.inject(FaultSpec("network_partition", "cart"))
    left = inj.leftovers()
    assert any("ledger" in x for x in left) and any("NetworkPolicy jev-fault-np" in x for x in left)


def test_network_policy_selects_target_pods_and_denies_ingress(tmp_path):
    kube, inj = injector(tmp_path)
    inj.inject(FaultSpec("network_partition", "cart"))
    pol = kube.policies["jev-fault-np"]["spec"]
    assert pol == {"podSelector": {"matchLabels": {"opentelemetry.io/name": "cart"}},
                   "policyTypes": ["Ingress"], "ingress": []}


def test_deadman_arm_and_disarm_commands():
    calls = []

    class R:
        returncode, stderr = 0, ""
    d = Deadman(run=lambda cmd: (calls.append(cmd), R())[1], python="/py", cwd="/repo")
    unit = d.arm(480)
    assert calls[0][:2] == ["systemd-run", "--user"] and "--on-active=480s" in calls[0]
    assert calls[0][-3:] == ["-m", "tier3", "reset-faults"] and f"--unit={unit}" in calls[0]
    d.disarm(unit)
    assert calls[1] == ["systemctl", "--user", "stop", f"{unit}.timer"]
