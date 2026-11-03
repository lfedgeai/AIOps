import base64
import os

import yaml

from tests.tier3.helpers import FakeKube
from tier3 import testbed
from tier3.collectors.kube import (JSON_PATCH, MERGE_PATCH, Kube, KubeConfig, deployment_rows,
                                   event_rows, is_infra, node_rows, pod_rows)
from tier3.faults import FaultInjector


class Recorder:
    def __init__(self, responses=None):
        self.calls, self.responses = [], responses or {}

    def __call__(self, method, path, *, params=None, body=None, content_type="application/json"):
        self.calls.append((method, path, params, body, content_type))
        return self.responses.get((method, path), {"items": []})


def test_kubeconfig_data_written_to_private_files(tmp_path):
    b64 = lambda s: base64.b64encode(s.encode()).decode()
    cfg = {"current-context": "kind-jev",
           "contexts": [{"name": "kind-jev", "context": {"cluster": "c", "user": "u"}}],
           "clusters": [{"name": "c", "cluster": {"server": "https://127.0.0.1:6443/",
                                                  "certificate-authority-data": b64("CA")}}],
           "users": [{"name": "u", "user": {"client-certificate-data": b64("CERT"),
                                            "client-key-data": b64("KEY")}}]}
    p = tmp_path / "kubeconfig"
    p.write_text(yaml.safe_dump(cfg))
    k = KubeConfig.load(p)
    assert k.server == "https://127.0.0.1:6443"
    assert open(k.key_file).read() == "KEY" and oct(os.stat(k.key_file).st_mode)[-3:] == "600"


def test_request_paths_and_patch_types():
    rec = Recorder()
    k = Kube(rec, "otel-demo")
    k.scale("cart", 0)
    k.replace_template("cart", {"spec": {}})
    k.set_taints("jev-worker2", [])
    k.list_pods({"opentelemetry.io/name": "cart"})
    k.list_warning_events()
    scale, tpl, taint, pods, ev1, ev2 = rec.calls
    assert scale[:2] == ("PATCH", "/apis/apps/v1/namespaces/otel-demo/deployments/cart/scale")
    assert scale[3] == {"spec": {"replicas": 0}} and scale[4] == MERGE_PATCH
    assert tpl[3] == [{"op": "replace", "path": "/spec/template", "value": {"spec": {}}}]
    assert tpl[4] == JSON_PATCH
    assert taint[1] == "/api/v1/nodes/jev-worker2" and taint[3] == {"spec": {"taints": None}}
    assert pods[2] == {"labelSelector": "opentelemetry.io/name=cart"}
    assert {ev1[1], ev2[1]} == {"/api/v1/namespaces/otel-demo/events", "/api/v1/namespaces/default/events"}
    assert ev1[2] == {"fieldSelector": "type=Warning"}


def test_missing_policy_delete_returns_false():
    rec = Recorder({("DELETE", "/apis/networking.k8s.io/v1/namespaces/otel-demo/networkpolicies/x"): None})
    assert Kube(rec).delete_network_policy("x") is False


def test_infra_prefixes():
    assert is_infra("load-generator") and is_infra("prometheus-server") and is_infra("flagd")
    assert not is_infra("cart") and not is_infra("frontend-proxy") and not is_infra("jaegerish")


def _dep(name, desired, avail, updated=None, gen=1, observed=1, stalled=False):
    st = {"observedGeneration": observed, "replicas": desired, "availableReplicas": avail,
          "readyReplicas": avail, "updatedReplicas": desired if updated is None else updated}
    if stalled:
        st["conditions"] = [{"type": "Progressing", "status": "False",
                             "reason": "ProgressDeadlineExceeded"}]
    return {"metadata": {"name": name, "generation": gen}, "spec": {"replicas": desired}, "status": st}


def test_deployment_rollout_states():
    rows = deployment_rows([_dep("cart", 1, 1), _dep("checkout", 1, 1, updated=0),
                            _dep("payment", 1, 0, stalled=True), _dep("ad", 1, 1, gen=2),
                            _dep("load-generator", 3, 3)])
    assert rows["cart"]["rollout"] == "complete"
    assert rows["checkout"]["rollout"] == "progressing" and rows["ad"]["rollout"] == "progressing"
    assert rows["payment"]["rollout"] == "stalled"
    assert "load-generator" not in rows


def test_pod_node_and_event_rows():
    kube = FakeKube()
    pod = next(p for p in kube.pods.values() if p["metadata"]["name"].startswith("cart"))
    pod["status"]["containerStatuses"][0].update(
        restartCount=2, state={"waiting": {"reason": "CrashLoopBackOff"}},
        lastState={"terminated": {"reason": "OOMKilled"}})
    rows = pod_rows(kube.list_pods())
    cart = next(r for r in rows if r["service"] == "cart")
    assert (cart["restarts"], cart["waiting_reason"], cart["last_termination_reason"]) == (
        2, "CrashLoopBackOff", "OOMKilled")
    assert not any(r["service"] == "load-generator" for r in rows)
    kube.nodes["jev-worker2"]["status"]["conditions"].append({"type": "DiskPressure", "status": "True"})
    nodes = {n["name"]: n for n in node_rows(kube.list_nodes(), rows)}
    assert nodes["jev-worker2"]["pressure"] == ["DiskPressure"]
    assert nodes["jev-control-plane"]["taints"][0]["effect"] == "NoSchedule"
    events = [{"metadata": {"namespace": "otel-demo"}, "reason": "BackOff", "message": "m",
               "involvedObject": {"kind": "Pod", "name": "cart-1"}, "lastTimestamp": ts}
              for ts in ("2026-01-01T00:00:10Z", "2026-01-01T01:00:00Z")]
    start = 1767225600.0  # 2026-01-01T00:00:00Z
    assert len(event_rows(events, start, start + 60)) == 1


def test_health_check_names_leftover_and_suggests_reset(tmp_path):
    kube = FakeKube()
    kube.policies["jev-fault-np"] = {"metadata": {"name": "jev-fault-np"}}

    class Prom:
        def span_metrics_present(self, now):
            return True

    class Jaeger:
        def services(self):
            return ["cart"]

    class Workload:
        def require_running(self):
            return 10

    get = lambda url: (200, b'{"status": "green"}')
    checks = testbed.run_checks(kube, Prom(), Jaeger(), Workload(),
                                FaultInjector(kube, ledger=tmp_path / "l.json"), os_url="http://x",
                                get=get)
    bad = [c for c in checks if not c.ok]
    assert [c.name for c in bad] == ["leftovers"]
    assert "jev-fault-np" in bad[0].detail and "reset-faults" in bad[0].detail


def test_health_check_reports_missing_deployments(tmp_path):
    kube = FakeKube()
    del kube.deps["valkey-cart"]
    checks = testbed.run_checks(kube, None, None, None, FaultInjector(kube, ledger=tmp_path / "l"),
                                os_url="http://x", get=lambda u: (200, b"{}"))
    dep = next(c for c in checks if c.name == "deployments")
    assert not dep.ok and "valkey-cart" in dep.detail


def test_subchart_pods_map_to_infra_not_to_their_component_label():
    prom = {"metadata": {"name": "prometheus-6f7d-abcde",
                         "labels": {"app.kubernetes.io/name": "prometheus",
                                    "app.kubernetes.io/component": "server"}},
            "spec": {"nodeName": "jev-worker"}, "status": {"phase": "Running"}}
    cart = {"metadata": {"name": "cart-1", "labels": {"opentelemetry.io/name": "cart",
                                                      "app.kubernetes.io/component": "cart"}},
            "spec": {}, "status": {"phase": "Running"}}
    assert [r["service"] for r in pod_rows([prom, cart])] == ["cart"]
    assert is_infra("opamp-server") and is_infra("otel-collector-agent")


def test_health_check_flags_an_empty_app_node(tmp_path):
    kube = FakeKube()
    for p in kube.pods.values():
        if p["spec"]["nodeName"] == "jev-worker2":
            p["spec"]["nodeName"] = "jev-worker3"
    checks = testbed.run_checks(kube, None, None, None, FaultInjector(kube, ledger=tmp_path / "l"),
                                os_url="http://x", get=lambda u: (200, b"{}"))
    c = next(c for c in checks if c.name == "placement")
    assert not c.ok and "jev-worker2 0 pods" in c.detail and "rebalance.sh" in c.detail


def test_placement_ok_matches_the_health_check():
    kube = FakeKube()
    assert testbed.placement_ok(kube)
    for p in kube.pods.values():
        if p["spec"]["nodeName"] == "jev-worker2":
            p["spec"]["nodeName"] = "jev-worker3"
    assert not testbed.placement_ok(kube)
