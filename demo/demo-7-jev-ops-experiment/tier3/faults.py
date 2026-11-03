"""The LF Edge harness's ten Kubernetes faults, with exact save and restore (design D5).

Differences from the group's inject_k8s_fault, on purpose:
  - nothing names the fault: no HARNESS_INJECTED_FAULT variable, no `harness-*`
    objects; harness objects use the `jev-fault` prefix, which the scrubber removes;
  - reset restores the saved state (replica counts, whole pod template), not fixed values;
  - node_taint targets an application worker and adds one taint instead of
    replacing the node's taint list;
  - pvc_full checks the write succeeded instead of ignoring failure (`|| true`).

Before any change, the saved state goes to a ledger file, so `python -m tier3
reset-faults` (run by hand or by the dead-man timer) can restore it if the
harness dies.
"""
from __future__ import annotations

import copy
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from tier3.scenarios import FaultSpec

PREFIX = "jev-fault"
TAINT = {"key": PREFIX, "value": "true", "effect": "NoExecute"}
POLICY_NAME = f"{PREFIX}-np"
FILL_PATH = f"/tmp/{PREFIX}-fill"
FILL_MB = 500
RESTART_WAIT_S = 30
_sleep = time.sleep  # replaced in tests
MEMORY_LIMIT = "10Mi"
BAD_DB_HOST = "invalid-host-that-does-not-exist.local"  # the group's value, kept for comparability
LOAD_GENERATOR = "load-generator"
OVERLOAD_REPLICAS = 3

DEFAULT_LEDGER = Path("results") / "tier3" / ".active-fault.json"


class FaultError(RuntimeError):
    pass


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


# --- template transforms (inject) ---------------------------------------------
def _containers(tpl: dict) -> list[dict]:
    return tpl["spec"]["containers"]


def _with_memory_limit(tpl: dict) -> dict:
    tpl = copy.deepcopy(tpl)
    for c in _containers(tpl):
        c.setdefault("resources", {}).setdefault("limits", {})["memory"] = MEMORY_LIMIT
        req = c["resources"].get("requests") or {}
        if "memory" in req:  # a request above the limit is rejected by the API server
            req["memory"] = MEMORY_LIMIT
    return tpl


def _with_failing_readiness(tpl: dict) -> dict:
    tpl = copy.deepcopy(tpl)
    for c in _containers(tpl):
        c["readinessProbe"] = {"exec": {"command": ["false"]}, "periodSeconds": 5,
                               "failureThreshold": 1}
    return tpl


def _with_bad_db_host(tpl: dict) -> dict:
    tpl = copy.deepcopy(tpl)
    for c in _containers(tpl):
        env = [e for e in c.get("env") or [] if e.get("name") != "DATABASE_HOST"]
        env.append({"name": "DATABASE_HOST", "value": BAD_DB_HOST})
        c["env"] = env
    return tpl


TEMPLATE_FAULTS = {"memory_limit": _with_memory_limit,
                   "readiness_probe_fail": _with_failing_readiness,
                   "config_corruption": _with_bad_db_host}


def _template_ok(fault_type: str, tpl: dict) -> bool:
    cs = _containers(tpl)
    if fault_type == "memory_limit":
        return all(((c.get("resources") or {}).get("limits") or {}).get("memory") == MEMORY_LIMIT
                   for c in cs)
    if fault_type == "readiness_probe_fail":
        return all(((c.get("readinessProbe") or {}).get("exec") or {}).get("command") == ["false"]
                   for c in cs)
    return all(any(e.get("name") == "DATABASE_HOST" and e.get("value") == BAD_DB_HOST
                   for e in c.get("env") or []) for c in cs)


def _has_our_taint(node: dict) -> bool:
    return any(t.get("key") == PREFIX for t in (node.get("spec") or {}).get("taints") or [])


# --- injector ------------------------------------------------------------------
class FaultInjector:
    def __init__(self, kube, ledger: Path = DEFAULT_LEDGER):
        self.kube, self.ledger = kube, Path(ledger)
        self._entry: dict | None = None

    def leftovers(self) -> list[str]:
        """Anything the harness leaves behind: ledger, policies, taints."""
        out = []
        if self.ledger.exists():
            out.append(f"fault ledger {self.ledger}")
        out += [f"NetworkPolicy {p['metadata']['name']}" for p in self.kube.list_network_policies()
                if p["metadata"]["name"].startswith(PREFIX)]
        out += [f"taint {PREFIX} on {n['metadata']['name']}" for n in self.kube.list_nodes()
                if _has_our_taint(n)]
        return out

    def _selector(self, name: str) -> dict:
        return self.kube.deployment(name)["spec"]["selector"]["matchLabels"]

    def _save(self, fault: FaultSpec) -> dict:
        k, t = self.kube, fault.target
        if fault.type in ("scale_zero", "dependency_removal"):
            return {"replicas": {t: k.deployment(t)["spec"].get("replicas", 1)}}
        if fault.type == "replica_overload":
            return {"replicas": {t: k.deployment(t)["spec"].get("replicas", 1),
                                 LOAD_GENERATOR: k.deployment(LOAD_GENERATOR)["spec"].get("replicas", 1)}}
        if fault.type in TEMPLATE_FAULTS:
            return {"templates": {t: copy.deepcopy(k.deployment(t)["spec"]["template"])}}
        if fault.type == "kill_pod":
            return {"killed_pods": sorted(p["metadata"]["name"] for p in k.list_pods(self._selector(t)))}
        if fault.type == "network_partition":
            return {"network_policy": POLICY_NAME}
        if fault.type == "node_taint":
            return {"taint_node": t}
        if fault.type == "pvc_full":
            pods = [p for p in k.list_pods(self._selector(t))
                    if (p.get("status") or {}).get("phase") == "Running"
                    and not p["metadata"].get("deletionTimestamp")]
            if not pods:
                raise FaultError(f"pvc_full: no running pod for {t}")
            pod = sorted(pods, key=lambda p: p["metadata"]["name"])[0]
            return {"fill": {"pod": pod["metadata"]["name"], "path": FILL_PATH,
                             "restarts": _restarts(pod)}}
        raise FaultError(f"unknown fault type {fault.type!r}")

    def inject(self, fault: FaultSpec) -> dict:
        """Save state to the ledger, apply the fault, read it back. Returns the saved state.
        Raises FaultError on a read-back mismatch; the caller still runs reset()."""
        if self._entry is not None or self.ledger.exists():
            raise FaultError(f"a fault is already active (ledger {self.ledger}); run reset-faults")
        saved = self._save(fault)
        self._entry = {"fault": {"type": fault.type, "target": fault.target},
                       "namespace": getattr(self.kube, "ns", None), "saved": saved,
                       "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        _write_json_atomic(self.ledger, self._entry)
        self._apply(fault, saved)
        self._read_back(fault, saved)
        return copy.deepcopy(saved)

    def _apply(self, fault: FaultSpec, saved: dict) -> None:
        k, t = self.kube, fault.target
        if fault.type in ("scale_zero", "dependency_removal"):
            k.scale(t, 0)
        elif fault.type == "replica_overload":
            k.scale(t, 0)
            k.scale(LOAD_GENERATOR, OVERLOAD_REPLICAS)
        elif fault.type in TEMPLATE_FAULTS:
            k.replace_template(t, TEMPLATE_FAULTS[fault.type](saved["templates"][t]))
        elif fault.type == "kill_pod":
            for name in saved["killed_pods"]:
                k.delete_pod(name)
        elif fault.type == "network_partition":
            k.create_network_policy({
                "apiVersion": "networking.k8s.io/v1", "kind": "NetworkPolicy",
                "metadata": {"name": POLICY_NAME},
                "spec": {"podSelector": {"matchLabels": self._selector(t)},
                         "policyTypes": ["Ingress"], "ingress": []}})
        elif fault.type == "node_taint":
            taints = [x for x in (k.node(t).get("spec") or {}).get("taints") or []
                      if x.get("key") != PREFIX]
            k.set_taints(t, taints + [dict(TAINT)])
        elif fault.type == "pvc_full":
            f = saved["fill"]
            rc, out = k.exec(f["pod"], ["sh", "-c", f"dd if=/dev/zero of={f['path']} bs=1M "
                                                    f"count={FILL_MB}"])
            if rc != 0 and not _was_killed_by_write(k, f, rc):
                raise FaultError(f"pvc_full: write failed in {f['pod']} (rc={rc}): {out.strip()[-200:]}")

    def _read_back(self, fault: FaultSpec, saved: dict) -> None:
        k, t = self.kube, fault.target
        if fault.type in ("scale_zero", "dependency_removal", "replica_overload"):
            got = k.deployment(t)["spec"].get("replicas")
            if got != 0:
                raise FaultError(f"read-back mismatch: {t} replicas {got}, wanted 0")
            if fault.type == "replica_overload":
                got = k.deployment(LOAD_GENERATOR)["spec"].get("replicas")
                if got != OVERLOAD_REPLICAS:
                    raise FaultError(f"read-back mismatch: {LOAD_GENERATOR} replicas {got}, "
                                     f"wanted {OVERLOAD_REPLICAS}")
        elif fault.type in TEMPLATE_FAULTS:
            if not _template_ok(fault.type, k.deployment(t)["spec"]["template"]):
                raise FaultError(f"read-back mismatch: {t} pod template does not carry the change")
        elif fault.type == "kill_pod":
            live = {p["metadata"]["name"] for p in k.list_pods(self._selector(t))
                    if not p["metadata"].get("deletionTimestamp")}
            still = live & set(saved["killed_pods"])
            if still:
                raise FaultError(f"read-back mismatch: pods not deleted: {sorted(still)}")
        elif fault.type == "network_partition":
            if POLICY_NAME not in {p["metadata"]["name"] for p in k.list_network_policies()}:
                raise FaultError("read-back mismatch: NetworkPolicy not present")
        elif fault.type == "node_taint":
            if not _has_our_taint(k.node(t)):
                raise FaultError(f"read-back mismatch: taint missing on {t}")

    def reset(self) -> list[str]:
        """Restore exactly what was saved, read it back, remove the ledger.
        Returns what was restored. Raises FaultError if the restore does not hold."""
        entry = self._entry
        if entry is None and self.ledger.exists():
            entry = json.loads(self.ledger.read_text())
        if entry is None:
            return []
        done = restore(self.kube, entry["saved"])
        self._entry = None
        self.ledger.unlink(missing_ok=True)
        return done


def restore(kube, saved: dict) -> list[str]:
    """Undo a saved state. Idempotent. Raises FaultError if read-back disagrees."""
    done = []
    for name, tpl in (saved.get("templates") or {}).items():
        if not _same(kube.deployment(name)["spec"]["template"], tpl):
            kube.replace_template(name, tpl)
            done.append(f"pod template of {name}")
        if not _same(kube.deployment(name)["spec"]["template"], tpl):
            raise FaultError(f"restore mismatch: pod template of {name}")
    for name, n in (saved.get("replicas") or {}).items():
        if kube.deployment(name)["spec"].get("replicas") != n:
            kube.scale(name, n)
            done.append(f"{name} replicas -> {n}")
        got = kube.deployment(name)["spec"].get("replicas")
        if got != n:
            raise FaultError(f"restore mismatch: {name} replicas {got}, wanted {n}")
    if saved.get("network_policy"):
        if kube.delete_network_policy(saved["network_policy"]):
            done.append(f"NetworkPolicy {saved['network_policy']} deleted")
    if saved.get("taint_node"):
        node = saved["taint_node"]
        taints = (kube.node(node).get("spec") or {}).get("taints") or []
        if any(t.get("key") == PREFIX for t in taints):
            kube.set_taints(node, [t for t in taints if t.get("key") != PREFIX])
            done.append(f"taint removed from {node}")
        if _has_our_taint(kube.node(node)):
            raise FaultError(f"restore mismatch: taint still on {node}")
    if saved.get("fill"):
        f = saved["fill"]
        live = {p["metadata"]["name"]: p for p in kube.list_pods()}
        # A replaced pod, or a restarted container (new writable layer), took the file
        # with it. The write itself can get the container OOMKilled (page cache counts
        # against its memory limit; seen on cart, 2026-10-14).
        if f["pod"] in live and _restarts(live[f["pod"]]) > f.get("restarts", _restarts(live[f["pod"]])):
            done.append(f"container in {f['pod']} restarted; fill file gone with it")
        elif f["pod"] in live:
            rc, out = kube.exec(f["pod"], ["rm", "-f", f["path"]])
            if rc == 0:
                done.append(f"fill file removed from {f['pod']}")
            elif _container_restarted(kube, f):  # exec raced the restart
                done.append(f"container in {f['pod']} restarted; fill file gone with it")
            else:
                raise FaultError(f"restore failed: could not remove fill file in {f['pod']}: {out[-200:]}")
    return done


def _was_killed_by_write(kube, fill: dict, rc: int) -> bool:
    """The write is the fault. On a pod with a small memory limit, the page cache it fills
    counts against the limit and the container is OOMKilled mid-write (exit 137; seen on
    cart, 2026-10-14). That is the fault taking effect, not a failed inject."""
    return rc == 137 and _container_restarted(kube, fill)


def _container_restarted(kube, fill: dict) -> bool:
    """Whether the fill pod's container restarted since inject. The kubelet reports the
    restart a few seconds after the kill, so wait for it (up to RESTART_WAIT_S)."""
    for i in range(RESTART_WAIT_S // 2 + 1):
        pod = next((p for p in kube.list_pods() if p["metadata"]["name"] == fill["pod"]), None)
        if pod is None:
            return False
        if _restarts(pod) > fill.get("restarts", _restarts(pod)):
            return True
        if i < RESTART_WAIT_S // 2:
            _sleep(2)
    return False


def _restarts(pod: dict) -> int:
    return sum(int(c.get("restartCount") or 0)
               for c in (pod.get("status") or {}).get("containerStatuses") or [])


def reset_faults(kube, ledger: Path = DEFAULT_LEDGER) -> list[str]:
    """Global, idempotent cleanup: ledger restore, every jev-fault policy and taint."""
    ledger = Path(ledger)
    changed = []
    if ledger.exists():
        entry = json.loads(ledger.read_text())
        if entry.get("action"):  # Tier 4: undo the action first, then the fault
            changed += restore(kube, entry["action"]["saved"])
        changed += restore(kube, entry.get("saved") or {})
        ledger.unlink()
        changed.append(f"ledger {ledger} cleared")
    for p in kube.list_network_policies():
        name = p["metadata"]["name"]
        if name.startswith(PREFIX) and kube.delete_network_policy(name):
            changed.append(f"NetworkPolicy {name} deleted")
    for n in kube.list_nodes():
        taints = (n.get("spec") or {}).get("taints") or []
        if any(t.get("key") == PREFIX for t in taints):
            kube.set_taints(n["metadata"]["name"], [t for t in taints if t.get("key") != PREFIX])
            changed.append(f"taint removed from {n['metadata']['name']}")
    return changed
