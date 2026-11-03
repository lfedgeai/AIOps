"""Thin Kubernetes API client and the pure functions that turn API objects into
collector rows. No `kubernetes` package: plain HTTPS with `requests`, using the
client certificate from the kind kubeconfig (tier3/testbed/kubeconfig).

Pod exec (needed only by the pvc_full fault) speaks a streaming protocol that
`requests` cannot, so it shells out to `kubectl exec` with the same kubeconfig.

Object shapes are the stable core/v1, apps/v1 and networking.k8s.io/v1 APIs.
Chart-specific names (labels, infrastructure deployments) are checked in task 1.4.
"""
from __future__ import annotations

import base64
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

JSON = "application/json"
MERGE_PATCH = "application/merge-patch+json"
JSON_PATCH = "application/json-patch+json"

# Deployments that are harness, load or observability infrastructure, not part of
# the application the engine reasons about. Matched as name prefixes because the
# chart's subcharts add suffixes (prometheus-server, jaeger-all-in-one, ...).
INFRA_PREFIXES = ("load-generator", "flagd", "opamp-server", "otel-collector", "opentelemetry-collector",
                  "jaeger", "prometheus", "grafana", "opensearch", "telemetry-docs")
# Namespaces never shown to the engine (design D10).
EXCLUDED_NAMESPACES = frozenset({"kube-system", "kube-public", "kube-node-lease",
                                 "local-path-storage", "calico-system", "tigera-operator",
                                 "calico-apiserver"})
# The demo's own label first: subchart pods set app.kubernetes.io/component to
# values like "server" (prometheus) or "all-in-one" (jaeger).
SERVICE_LABELS = ("opentelemetry.io/name", "app.kubernetes.io/name", "app.kubernetes.io/component")
PRESSURE_CONDITIONS = ("MemoryPressure", "DiskPressure", "PIDPressure", "NetworkUnavailable")


class KubeError(RuntimeError):
    pass


def is_infra(name: str) -> bool:
    return any(name == p or name.startswith(p + "-") for p in INFRA_PREFIXES)


# --- transport ----------------------------------------------------------------
RequestFn = Callable[..., object]  # (method, url, *, params, body, content_type) -> dict | None


class KubeConfig:
    """Server URL and client credentials from a kubeconfig's current context."""

    def __init__(self, server: str, ca_file: str | None, cert_file: str | None,
                 key_file: str | None, path: str | None = None):
        self.server, self.ca_file, self.cert_file, self.key_file = server, ca_file, cert_file, key_file
        self.path = path

    @classmethod
    def load(cls, path: str | os.PathLike) -> "KubeConfig":
        path = Path(path)
        if not path.exists():
            raise KubeError(f"kubeconfig not found: {path} (run ./setup-k8s.sh)")
        cfg = yaml.safe_load(path.read_text())
        ctx_name = cfg.get("current-context")
        ctx = next((c["context"] for c in cfg.get("contexts", []) if c["name"] == ctx_name), None)
        if ctx is None:
            raise KubeError(f"{path}: no current context")
        cluster = next(c["cluster"] for c in cfg["clusters"] if c["name"] == ctx["cluster"])
        user = next(u["user"] for u in cfg["users"] if u["name"] == ctx["user"])
        tmp = Path(tempfile.mkdtemp(prefix="tier3-kube-"))

        def material(data_key: str, file_key: str, src: dict, name: str) -> str | None:
            if src.get(file_key):
                return src[file_key]
            if src.get(data_key):
                p = tmp / name
                p.write_bytes(base64.b64decode(src[data_key]))
                p.chmod(0o600)
                return str(p)
            return None

        return cls(server=cluster["server"].rstrip("/"),
                   ca_file=material("certificate-authority-data", "certificate-authority", cluster, "ca.crt"),
                   cert_file=material("client-certificate-data", "client-certificate", user, "client.crt"),
                   key_file=material("client-key-data", "client-key", user, "client.key"),
                   path=str(path))


def requests_transport(cfg: KubeConfig, timeout: float = 20.0) -> RequestFn:
    import json

    import requests

    session = requests.Session()
    session.verify = cfg.ca_file or True
    if cfg.cert_file and cfg.key_file:
        session.cert = (cfg.cert_file, cfg.key_file)

    def call(method, path, *, params=None, body=None, content_type=JSON):
        data = None if body is None else json.dumps(body)
        r = session.request(method, cfg.server + path, params=params, data=data,
                            headers={"Content-Type": content_type, "Accept": JSON}, timeout=timeout)
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            raise KubeError(f"{method} {path}: HTTP {r.status_code} {r.text[:300]}")
        return r.json() if r.text else {}
    return call


def kubectl_exec(kubeconfig: str | None, namespace: str):
    def run(pod: str, command: list[str], timeout: float = 120.0) -> tuple[int, str]:
        cmd = ["kubectl"] + (["--kubeconfig", kubeconfig] if kubeconfig else []) + \
              ["-n", namespace, "exec", pod, "--"] + list(command)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return r.returncode, (r.stdout + r.stderr)[-2000:]
    return run


# --- API ------------------------------------------------------------------------
class Kube:
    """The few Kubernetes operations Tier 3 needs, in one namespace."""

    def __init__(self, request: RequestFn, namespace: str = "otel-demo",
                 exec_fn: Callable[[str, list[str]], tuple[int, str]] | None = None):
        self._req, self.ns = request, namespace
        self._exec = exec_fn

    @classmethod
    def from_kubeconfig(cls, path, namespace: str = "otel-demo") -> "Kube":
        cfg = KubeConfig.load(path)
        return cls(requests_transport(cfg), namespace, kubectl_exec(cfg.path, namespace))

    # paths
    def _apps(self, rest=""):
        return f"/apis/apps/v1/namespaces/{self.ns}/deployments{rest}"

    def _netpol(self, rest=""):
        return f"/apis/networking.k8s.io/v1/namespaces/{self.ns}/networkpolicies{rest}"

    # deployments
    def list_deployments(self) -> list[dict]:
        return (self._req("GET", self._apps()) or {}).get("items", [])

    def deployment(self, name: str) -> dict:
        d = self._req("GET", self._apps(f"/{name}"))
        if d is None:
            raise KubeError(f"deployment {name!r} not found in {self.ns}")
        return d

    def scale(self, name: str, replicas: int) -> None:
        self._req("PATCH", self._apps(f"/{name}/scale"),
                  body={"spec": {"replicas": int(replicas)}}, content_type=MERGE_PATCH)

    def replace_template(self, name: str, template: dict) -> None:
        """Replace the whole pod template, so a restore is exact (no merge leftovers)."""
        self._req("PATCH", self._apps(f"/{name}"),
                  body=[{"op": "replace", "path": "/spec/template", "value": template}],
                  content_type=JSON_PATCH)

    # pods
    def list_pods(self, selector: dict | None = None) -> list[dict]:
        params = {"labelSelector": ",".join(f"{k}={v}" for k, v in sorted(selector.items()))} \
            if selector else None
        return (self._req("GET", f"/api/v1/namespaces/{self.ns}/pods", params=params) or {}).get("items", [])

    def delete_pod(self, name: str) -> None:
        self._req("DELETE", f"/api/v1/namespaces/{self.ns}/pods/{name}")

    def exec(self, pod: str, command: list[str]) -> tuple[int, str]:
        if self._exec is None:
            raise KubeError("pod exec is not configured")
        return self._exec(pod, command)

    # nodes
    def list_nodes(self) -> list[dict]:
        return (self._req("GET", "/api/v1/nodes") or {}).get("items", [])

    def node(self, name: str) -> dict:
        n = self._req("GET", f"/api/v1/nodes/{name}")
        if n is None:
            raise KubeError(f"node {name!r} not found")
        return n

    def set_taints(self, name: str, taints: list[dict]) -> None:
        # A merge patch replaces the whole list, so the caller passes the full new list.
        self._req("PATCH", f"/api/v1/nodes/{name}", body={"spec": {"taints": taints or None}},
                  content_type=MERGE_PATCH)

    # network policies
    def list_network_policies(self) -> list[dict]:
        return (self._req("GET", self._netpol()) or {}).get("items", [])

    def create_network_policy(self, body: dict) -> None:
        self._req("POST", self._netpol(), body=body)

    def delete_network_policy(self, name: str) -> bool:
        return self._req("DELETE", self._netpol(f"/{name}")) is not None

    # events: the demo namespace plus `default`, where node events land
    def list_warning_events(self) -> list[dict]:
        out = []
        for ns in (self.ns, "default"):
            body = self._req("GET", f"/api/v1/namespaces/{ns}/events",
                             params={"fieldSelector": "type=Warning"}) or {}
            out.extend(body.get("items", []))
        return out


# --- pure conversion: API objects -> collector rows ----------------------------------
def _ts(s: str | None) -> float | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def service_of(obj: dict) -> str:
    labels = (obj.get("metadata") or {}).get("labels") or {}
    for k in SERVICE_LABELS:
        if labels.get(k):
            return labels[k]
    name = obj["metadata"]["name"]
    parts = name.split("-")
    return "-".join(parts[:-2]) if len(parts) > 2 else name  # deployment-<rs hash>-<pod hash>


def deployment_rows(items: list[dict]) -> dict[str, dict]:
    out = {}
    for d in items:
        name = d["metadata"]["name"]
        if is_infra(name):
            continue
        spec, st = d.get("spec") or {}, d.get("status") or {}
        desired = int(spec.get("replicas", 1))
        updated = int(st.get("updatedReplicas") or 0)
        avail = int(st.get("availableReplicas") or 0)
        conds = {c.get("type"): c for c in st.get("conditions") or []}
        prog = conds.get("Progressing") or {}
        if prog.get("status") == "False" and prog.get("reason") == "ProgressDeadlineExceeded":
            rollout = "stalled"
        elif (int(st.get("observedGeneration") or 0) < int(d["metadata"].get("generation") or 0)
              or updated < desired or int(st.get("replicas") or 0) > desired or avail < desired):
            rollout = "progressing"
        else:
            rollout = "complete"
        out[name] = {"desired": desired, "ready": int(st.get("readyReplicas") or 0),
                     "available": avail, "updated": updated, "rollout": rollout}
    return out


def pod_rows(items: list[dict]) -> list[dict]:
    out = []
    for p in items:
        md, st, spec = p["metadata"], p.get("status") or {}, p.get("spec") or {}
        svc = service_of(p)
        if is_infra(svc):
            continue
        statuses = st.get("containerStatuses") or []
        waiting = next((c["state"]["waiting"].get("reason") for c in statuses
                        if (c.get("state") or {}).get("waiting")), None)
        last = next((c["lastState"]["terminated"].get("reason") for c in statuses
                     if (c.get("lastState") or {}).get("terminated")), None)
        ready = any(c.get("type") == "Ready" and c.get("status") == "True"
                    for c in st.get("conditions") or [])
        out.append({"name": md["name"], "namespace": md.get("namespace"), "service": svc,
                    "node": spec.get("nodeName"), "phase": st.get("phase"), "ready": ready,
                    "restarts": sum(int(c.get("restartCount") or 0) for c in statuses),
                    "waiting_reason": waiting, "last_termination_reason": last,
                    "deleting": bool(md.get("deletionTimestamp")),
                    "started": md.get("creationTimestamp")})
    return sorted(out, key=lambda r: r["name"])


def node_rows(items: list[dict], pods: list[dict] | None = None) -> list[dict]:
    counts: dict[str, int] = {}
    for p in pods or []:
        if p.get("node"):
            counts[p["node"]] = counts.get(p["node"], 0) + 1
    out = []
    for n in items:
        conds = {c.get("type"): c.get("status") for c in (n.get("status") or {}).get("conditions") or []}
        taints = (n.get("spec") or {}).get("taints") or []
        out.append({"name": n["metadata"]["name"], "ready": conds.get("Ready") == "True",
                    "pressure": sorted(k for k in PRESSURE_CONDITIONS if conds.get(k) == "True"),
                    "schedulable": not (n.get("spec") or {}).get("unschedulable", False),
                    "taints": [{"key": t.get("key"), "effect": t.get("effect")} for t in taints],
                    "pods": counts.get(n["metadata"]["name"], 0)})
    return sorted(out, key=lambda r: r["name"])


def event_rows(items: list[dict], start_s: float, end_s: float) -> list[dict]:
    out = []
    for e in items:
        md = e.get("metadata") or {}
        ns = md.get("namespace") or (e.get("involvedObject") or {}).get("namespace")
        if ns in EXCLUDED_NAMESPACES:
            continue
        t = _ts(e.get("lastTimestamp")) or _ts(e.get("eventTime")) or _ts(md.get("creationTimestamp"))
        if t is None or not (start_s <= t < end_s):
            continue
        obj = e.get("involvedObject") or {}
        if is_infra(obj.get("name", "")):
            continue
        out.append({"namespace": ns, "reason": e.get("reason"),
                    "object": f"{obj.get('kind')}/{obj.get('name')}",
                    "message": e.get("message") or "", "count": int(e.get("count") or 1),
                    "last_seen": datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")})
    return out
