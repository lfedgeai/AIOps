"""Shared fakes for Tier 3 tests. Nothing here touches the network, Docker or a cluster."""
from __future__ import annotations

import copy
import itertools

from tier2.observation import ServiceStats, Telemetry
from tier3.engine import Decision
from tier3.vocab import FAULT_TYPES

NS = "otel-demo"
APP = ("cart", "valkey-cart", "checkout", "frontend", "payment")
INFRA = ("load-generator", "flagd", "prometheus-server")
NODES = ("jev-control-plane", "jev-worker", "jev-worker2", "jev-worker3")


def _deployment(name: str, replicas: int = 1) -> dict:
    labels = {"opentelemetry.io/name": name}
    return {
        "metadata": {"name": name, "namespace": NS, "generation": 1, "labels": dict(labels)},
        "spec": {"replicas": replicas, "selector": {"matchLabels": dict(labels)},
                 "template": {"metadata": {"labels": dict(labels)},
                              "spec": {"containers": [{
                                  "name": name, "image": f"demo/{name}:3.1.0",
                                  "env": [{"name": "OTEL_SERVICE_NAME", "value": name}],
                                  "resources": {"limits": {"memory": "160Mi"}}}]}}},
    }


class FakeKube:
    """In-memory stand-in for tier3.collectors.kube.Kube. Applies changes instantly:
    pods follow replica counts, so a restore is visible at once."""

    def __init__(self, replicas: dict | None = None):
        self.ns = NS
        self.deps = {n: _deployment(n, (replicas or {}).get(n, 1)) for n in (*APP, *INFRA)}
        self.nodes = {n: {"metadata": {"name": n, "labels": {}},
                          "spec": ({"taints": [{"key": "node-role.kubernetes.io/control-plane",
                                                "effect": "NoSchedule"}]}
                                   if n == "jev-control-plane" else {}),
                          "status": {"conditions": [{"type": "Ready", "status": "True"}]}}
                      for n in NODES}
        self.policies: dict[str, dict] = {}
        self.events: list[dict] = []
        self.files: set[tuple[str, str]] = set()
        self.exec_rc = 0
        self.lying = False  # accept writes without applying them
        self.sticky: set[str] = set()  # deployments that stay unavailable once broken
        self.unavailable: set[str] = set()
        self.calls: list[tuple] = []
        self._ids = itertools.count(1)
        self.pods: dict[str, dict] = {}
        self._sync_pods()

    # --- simulation -----------------------------------------------------------
    def _new_pod(self, dep: str, i: int) -> dict:
        n = next(self._ids)
        node = "jev-worker" if dep in INFRA else ("jev-worker2", "jev-worker3")[n % 2]
        return {"metadata": {"name": f"{dep}-7d9f{n:02d}-x{i}", "namespace": NS,
                             "labels": {"opentelemetry.io/name": dep},
                             "creationTimestamp": "2026-01-01T00:00:00Z"},
                "spec": {"nodeName": node},
                "status": {"phase": "Running",
                           "conditions": [{"type": "Ready", "status": "True"}],
                           "containerStatuses": [{"name": dep, "restartCount": 0,
                                                  "state": {"running": {}}}]}}

    def _sync_pods(self):
        for name, d in self.deps.items():
            mine = sorted(p for p, v in self.pods.items() if v["metadata"]["labels"]["opentelemetry.io/name"] == name)
            want = d["spec"]["replicas"]
            for p in mine[want:]:
                del self.pods[p]
            for i in range(len(mine), want):
                pod = self._new_pod(name, i)
                self.pods[pod["metadata"]["name"]] = pod

    def _status(self, d: dict) -> dict:
        r = d["spec"]["replicas"]
        avail = 0 if d["metadata"]["name"] in self.unavailable else r
        return {"observedGeneration": d["metadata"]["generation"], "replicas": r,
                "readyReplicas": avail, "availableReplicas": avail, "updatedReplicas": r,
                "conditions": [{"type": "Progressing", "status": "True"}]}

    def _out(self, d: dict) -> dict:
        d = copy.deepcopy(d)
        d["status"] = self._status(d)
        return d

    # --- Kube API -------------------------------------------------------------------
    def list_deployments(self):
        return [self._out(d) for d in self.deps.values()]

    def deployment(self, name):
        from tier3.collectors.kube import KubeError
        if name not in self.deps:
            raise KubeError(f"deployment {name!r} not found")
        return self._out(self.deps[name])

    def scale(self, name, replicas):
        self.calls.append(("scale", name, replicas))
        if self.lying:
            return
        self.deps[name]["spec"]["replicas"] = replicas
        if replicas == 0 and name in self.sticky:
            self.unavailable.add(name)
        self._sync_pods()

    def replace_template(self, name, template):
        self.calls.append(("replace_template", name))
        if self.lying:
            return
        self.deps[name]["spec"]["template"] = copy.deepcopy(template)
        self.deps[name]["metadata"]["generation"] += 1

    def list_pods(self, selector=None):
        out = [copy.deepcopy(p) for p in self.pods.values()]
        if selector:
            out = [p for p in out if all(p["metadata"]["labels"].get(k) == v for k, v in selector.items())]
        return sorted(out, key=lambda p: p["metadata"]["name"])

    def delete_pod(self, name):
        self.calls.append(("delete_pod", name))
        if self.lying:
            return
        self.pods.pop(name, None)
        self.files = {f for f in self.files if f[0] != name}
        self._sync_pods()

    def exec(self, pod, command):
        self.calls.append(("exec", pod, tuple(command)))
        if self.exec_rc:
            return self.exec_rc, "sh: not found"
        text = " ".join(command)
        if "dd " in text:
            self.files.add((pod, text.split("of=")[1].split()[0]))
        elif command[:2] == ["rm", "-f"]:
            self.files.discard((pod, command[2]))
        return 0, ""

    def list_nodes(self):
        return [copy.deepcopy(n) for n in self.nodes.values()]

    def node(self, name):
        return copy.deepcopy(self.nodes[name])

    def set_taints(self, name, taints):
        self.calls.append(("set_taints", name))
        if self.lying:
            return
        self.nodes[name]["spec"]["taints"] = copy.deepcopy(taints)

    def list_network_policies(self):
        return [copy.deepcopy(p) for p in self.policies.values()]

    def create_network_policy(self, body):
        self.calls.append(("create_network_policy", body["metadata"]["name"]))
        if not self.lying:
            self.policies[body["metadata"]["name"]] = copy.deepcopy(body)

    def delete_network_policy(self, name):
        return self.policies.pop(name, None) is not None

    def list_warning_events(self):
        return copy.deepcopy(self.events)

    # --- for assertions -----------------------------------------------------------
    def snapshot(self) -> dict:
        """Everything a fault may change, except pod names (kill_pod replaces pods)."""
        return copy.deepcopy({"deps": {n: d["spec"] for n, d in self.deps.items()},
                              "taints": {n: v["spec"].get("taints") or [] for n, v in self.nodes.items()},
                              "policies": sorted(self.policies), "files": sorted(self.files)})


def warning_event(reason, obj, message, ts="2026-01-01T00:02:00Z", ns=NS, count=1) -> dict:
    kind, name = obj.split("/", 1)
    return {"metadata": {"namespace": ns, "creationTimestamp": ts}, "type": "Warning",
            "reason": reason, "involvedObject": {"kind": kind, "name": name, "namespace": ns},
            "message": message, "count": count, "lastTimestamp": ts}


def contaminated_events(experiment_id: str = "K009") -> list[dict]:
    """Events as a real cluster may emit them, deliberately carrying every scrub term,
    plus the only two Warning events the cluster emitted across all ten fault probes
    (task 1.10, kind, 2026-10-14; messages verbatim, pod names shortened)."""
    return [
        warning_event("BackOff", "Pod/cart-5dd4c7fd4f-6tvxh",
                      "Back-off restarting failed container cart in pod cart-5dd4c7fd4f-6tvxh_otel-demo"
                      "(416ecb39-5019-4c30-ab58-28c9462f45fb)", count=6),
        warning_event("Unhealthy", "Pod/cart-5845946d4f-xrszw", "Readiness probe failed: ", count=16),
        warning_event("FailedScheduling", "Pod/cart-7d9f01-x0",
                      "0/4 nodes are available: 1 node(s) had untolerated taint {jev-fault: true}, "
                      "1 node(s) had untolerated taint {node-role.kubernetes.io/control-plane: }."),
        warning_event("BackOff", "Pod/cart-7d9f01-x0", "Back-off restarting failed container cart",
                      count=4),
        warning_event("Unhealthy", "Pod/cart-7d9f01-x0", "Readiness probe failed: exec false"),
        warning_event("OOMKilling", "Node/jev-worker2",
                      f"Memory cgroup out of memory: Killed process 4242 (cart) {experiment_id}",
                      ns="default"),
        warning_event("PolicyNote", "Pod/cart-7d9f01-x0",
                      f"traffic denied by jev-fault-np ({' '.join(FAULT_TYPES)})"),
        warning_event("FailedMount", "Pod/coredns-abc", "noise from kube-system", ns="kube-system"),
        warning_event("BackOff", "Pod/load-generator-abc-x0", "harness pod event"),
    ]


def telemetry(faulty: bool = True, experiment_id: str = "K001") -> Telemetry:
    """Service telemetry that carries fault metadata on purpose (scrubber test)."""
    services = {"frontend": ServiceStats(5.0, 0.30 if faulty else 0.0, 20, 300),
                "checkout": ServiceStats(0.3, 0.50 if faulty else 0.0, 50, 900)}
    return Telemetry(
        window_start="2026-01-01T00:00:00+00:00", window_end="2026-01-01T00:03:00+00:00",
        baseline_start="2025-12-31T23:57:00+00:00", baseline_end="2026-01-01T00:00:00+00:00",
        services=services,
        baseline_services={"frontend": ServiceStats(5.0, 0.0, 20, 250),
                           "checkout": ServiceStats(0.3, 0.0, 50, 200)},
        edges=[{"from": "frontend", "to": "cart", "calls": 9, "errors": 9 if faulty else 0}],
        logs=[{"ts": "t1", "service": "checkout", "level": "ERROR",
               "message": f"failed to call cart: connection refused jev-fault {experiment_id} scale_zero"}],
        traces=[{"trace_id": "abcd1234", "path": ["frontend", "cart"], "duration_ms": 30.0,
                 "failed_spans": [{"service": "frontend", "operation": "GetCart",
                                   "error": "rpc.grpc.status_code=14 jev-fault-np"}]}] if faulty else [],
    )


def ok_decision(**kw) -> Decision:
    base = dict(status="ok", incident_detected=True, affected_component="cart",
                diagnosis="service_unavailable", recommended_action="scale_up",
                confidence=0.8, reasoning=None, raw_response={"answers": {}},
                telemetry={"input_tokens": 1000, "output_tokens": 20, "cost_usd": None})
    base.update(kw)
    return Decision(**base)


QUIET = dict(incident_detected=False, affected_component="none", diagnosis="none",
             recommended_action="no_action")


class SeqEngine:
    """Returns the given decisions (or raises the given exceptions) in order; the last repeats."""

    def __init__(self, *items):
        self.items, self.calls = list(items) or [ok_decision()], []

    def decide(self, observation):
        self.calls.append(observation)
        item = self.items[min(len(self.calls) - 1, len(self.items) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


class FakeDeadman:
    def __init__(self):
        self.armed, self.disarmed = [], []

    def arm(self, seconds):
        self.armed.append(seconds)
        return "jev-fault-reset-test"

    def disarm(self, unit):
        self.disarmed.append(unit)
