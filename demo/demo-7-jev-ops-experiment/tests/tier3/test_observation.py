import ast
import inspect
import json

from tests.tier3.helpers import FakeKube, contaminated_events, telemetry
from tier2.observation import ServiceStats, Telemetry
from tier3 import observation
from tier3.collect import kube_state
from tier3.faults import FaultInjector
from tier3.observation import KEYS, KubeState, build_observation
from tier3.scenarios import FaultSpec, GroundTruth, Scenario
from tier3.vocab import FAULT_TYPES

WINDOW = (0, 4e9)
TERMS = ("flagd", *FAULT_TYPES, "K009")


def state(kube):
    return kube_state(kube, *WINDOW)


def test_fixed_keys_healthy_and_faulty():
    kube = FakeKube()
    a, _ = build_observation(telemetry(False), state(kube), state(kube), TERMS)
    kube.events = contaminated_events()
    b, _ = build_observation(telemetry(True), state(kube), state(kube), TERMS)
    assert tuple(a) == tuple(b) == KEYS


def test_healthy_pods_omitted():
    kube = FakeKube()
    obs, _ = build_observation(telemetry(False), state(kube), state(kube), TERMS)
    assert obs["pods"] == [] and obs["events"] == []


def test_scale_zero_visible_as_replica_counts_only(tmp_path):
    kube = FakeKube()
    base = state(kube)
    FaultInjector(kube, ledger=tmp_path / "l.json").inject(FaultSpec("scale_zero", "cart"))
    obs, _ = build_observation(telemetry(True), state(kube), base, TERMS)
    row = next(r for r in obs["deployments"] if r["deployment"] == "cart")
    assert (row["desired"], row["available"], row["baseline_desired"], row["baseline_available"]) == (0, 0, 1, 1)
    assert obs["deployments"][0]["deployment"] == "cart"  # biggest drop ranked first
    text = json.dumps(obs)
    for spec_word in ("template", "containers", "env", "limits", "selector", "replicas\""):
        assert spec_word not in text


def test_memory_limit_never_shows_the_limit(tmp_path):
    kube = FakeKube()
    base = state(kube)
    FaultInjector(kube, ledger=tmp_path / "l.json").inject(FaultSpec("memory_limit", "cart"))
    obs, _ = build_observation(telemetry(True), state(kube), base, TERMS)
    assert "10Mi" not in json.dumps(obs) and "160Mi" not in json.dumps(obs)


def test_taint_key_removed_but_event_and_effect_kept(tmp_path):
    kube = FakeKube()
    base = state(kube)
    FaultInjector(kube, ledger=tmp_path / "l.json").inject(FaultSpec("node_taint", "jev-worker2"))
    kube.events = contaminated_events()
    obs, _ = build_observation(telemetry(True), state(kube), base, TERMS)
    text = json.dumps(obs)
    assert "jev-fault" not in text
    sched = next(e for e in obs["events"] if e["reason"] == "FailedScheduling")
    assert "untolerated taint {[redacted]: true}" in sched["message"]
    node = next(n for n in obs["nodes"] if n["node"] == "jev-worker2")
    assert node["taint_effects"] == ["NoExecute"] and "taints" not in node


def test_scrubber_removes_terms_but_keeps_evidence():
    kube = FakeKube()
    kube.events = contaminated_events("K009")
    obs, _ = build_observation(telemetry(True, "K009"), state(kube), state(kube), TERMS)
    text = json.dumps(obs)
    for t in ("jev-fault", "K009", *FAULT_TYPES):
        assert t not in text, t
    reasons = {e["reason"] for e in obs["events"]}
    assert {"FailedScheduling", "BackOff", "Unhealthy", "OOMKilling"} <= reasons
    assert "jev-worker2" in text  # node names are evidence, not scrubbed


def test_excluded_namespaces_and_infra_dropped():
    kube = FakeKube()
    kube.events = contaminated_events()
    obs, _ = build_observation(telemetry(True), state(kube), state(kube), TERMS)
    text = json.dumps(obs)
    assert "coredns" not in text and "kube-system" not in text
    assert "load-generator" not in text and "flagd" not in text and "prometheus" not in text


def test_restarted_and_new_pods_listed(tmp_path):
    kube = FakeKube()
    base = state(kube)
    pod = next(p for p in kube.pods.values() if p["metadata"]["name"].startswith("cart"))
    pod["status"]["containerStatuses"][0].update(
        restartCount=3, state={"waiting": {"reason": "CrashLoopBackOff"}},
        lastState={"terminated": {"reason": "OOMKilled"}})
    pod["status"]["conditions"][0]["status"] = "False"
    obs, _ = build_observation(telemetry(True), state(kube), base, TERMS)
    row = obs["pods"][0]
    assert (row["service"], row["ready"], row["restarts_in_window"]) == ("cart", False, 3)
    assert (row["waiting_reason"], row["last_termination_reason"]) == ("CrashLoopBackOff", "OOMKilled")


def test_events_deduplicated_with_counts():
    kube = FakeKube()
    kube.events = [contaminated_events()[3]] * 3
    obs, _ = build_observation(telemetry(True), state(kube), state(kube), TERMS)
    assert len(obs["events"]) == 1 and obs["events"][0]["count"] == 12


def test_byte_budget_trims_tail_and_notes_outside_request(monkeypatch):
    monkeypatch.setattr(observation, "BYTE_BUDGET", 2500)
    kube = FakeKube()
    kube.events = [dict(contaminated_events()[1], message=f"Back-off {'x' * 50} {chr(97 + i)}")
                   for i in range(20)]
    obs, trims = build_observation(telemetry(True), state(kube), state(kube), TERMS)
    assert len(json.dumps(obs, sort_keys=True).encode()) <= 2500
    assert any(k.startswith("budget_") for k in trims)
    assert "trim" not in json.dumps(obs).lower()


def test_unmeasured_values_are_null():
    obs, _ = build_observation(Telemetry("a", "b", services={"cart": ServiceStats(1, 0, None, None)}),
                               KubeState(deployments={"cart": {"desired": 1, "ready": 1, "available": 1,
                                                               "updated": 1, "rollout": "complete"}}))
    assert obs["deployments"][0]["baseline_available"] is None
    assert obs["services"][0]["latency_p95_ms"] is None


def test_builder_signature_has_no_scenario_types():
    params = inspect.signature(build_observation).parameters
    for p in params.values():
        assert p.annotation not in {Scenario, FaultSpec, GroundTruth}
        assert p.name not in {"scenario", "fault", "ground_truth", "truth", "expected"}


def test_observation_module_does_not_import_scenarios_or_faults():
    for mod in (observation, __import__("tier3.collect", fromlist=["x"]),
                __import__("tier3.collectors.kube", fromlist=["x"]),
                __import__("tier3.layouts", fromlist=["x"])):
        tree = ast.parse(inspect.getsource(mod))
        imported = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module:
                imported.add(n.module)
            elif isinstance(n, ast.Import):
                imported.update(a.name for a in n.names)
        assert not any(m.startswith(("tier3.scenarios", "tier3.faults", "tier2.scenarios"))
                       for m in imported), mod.__name__


def test_collector_and_jaeger_self_telemetry_dropped():
    from tier3.collect import without_infra
    t = telemetry(True)
    t.services["otelcol-contrib"] = ServiceStats(1, 0.9, 1, 1)
    t.logs.append({"ts": "t", "service": "otelcol-contrib", "level": "WARN", "message": "unable to open connection to broker"})
    t.edges.append({"from": "jaeger", "to": "frontend", "calls": 1, "errors": 0})
    t.traces.append({"trace_id": "x", "path": ["jaeger"], "duration_ms": 1,
                     "failed_spans": [{"service": "jaeger", "operation": "q", "error": "e"}]})
    text = json.dumps(build_observation(without_infra(t), KubeState())[0])
    assert "otelcol-contrib" not in text and "broker" not in text and "jaeger" not in text
    assert "checkout" in text  # application evidence kept


def test_real_cluster_warnings_reach_the_observation():
    kube = FakeKube()
    kube.events = contaminated_events()[:2]  # captured on kind in task 1.10
    obs, _ = build_observation(telemetry(True), state(kube), state(kube), TERMS)
    got = {(e["reason"], e["message"]) for e in obs["events"]}
    assert ("Unhealthy", "Readiness probe failed: ") in got
    assert any(r == "BackOff" and "cart-5dd4c7fd4f-6tvxh" in m for r, m in got)
