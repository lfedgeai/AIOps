"""The only code in this repo that acts on an engine's recommendation (design D2-D4, D9).

One action per run, from a fixed allowlist, on a Deployment in the demo namespace:
  scale_up        replicas := baseline if below it, else current + 1; never above REPLICA_CAP
  restart_service rollout restart (the pod template annotation kubectl writes)
  rollback_change `kubectl rollout undo` to the previous revision in the cluster's own
                  ReplicaSet history; never the harness's ledger or saved state
  investigate, escalate, no_action: nothing

Before acting, the target's replicas and pod template go to an `action` entry in the
Tier 3 ledger, so `python -m tier3 reset-faults` and the dead-man timer can undo the
action (then the fault) from the ledger alone. Only tier4/runner.py and tier4/cli.py
import this module (tests/tier4/test_isolation.py).
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from tier3.collectors.kube import deployment_rows, is_infra
from tier3.faults import DEFAULT_LEDGER, restore
from tier3.vocab import NODES

EXECUTABLE = ("scale_up", "restart_service", "rollback_change")
NOT_ACTED = ("investigate", "escalate", "no_action")
REPLICA_CAP = 3
RESTART_ANNOTATION = "kubectl.kubernetes.io/restartedAt"
REVISION_ANNOTATION = "deployment.kubernetes.io/revision"
ROLLOUT_TIMEOUT_S = 120.0
ROLLOUT_POLL_S = 2.0

Undo = Callable[[str], tuple[int, str]]  # deployment name -> (returncode, output)


class ExecutorError(RuntimeError):
    pass


def kubectl_undo(kubeconfig: str | None, namespace: str) -> Undo:
    def run(name: str, timeout: float = 60.0) -> tuple[int, str]:
        cmd = ["kubectl"] + (["--kubeconfig", kubeconfig] if kubeconfig else []) + \
              ["-n", namespace, "rollout", "undo", f"deployment/{name}"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return r.returncode, (r.stdout + r.stderr)[-2000:]
    return run


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="milliseconds")


def _revision(dep: dict) -> str | None:
    return ((dep.get("metadata") or {}).get("annotations") or {}).get(REVISION_ANNOTATION)


class Executor:
    def __init__(self, kube, ledger: Path = DEFAULT_LEDGER, undo: Undo | None = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 rollout_timeout_s: float = ROLLOUT_TIMEOUT_S, rollout_poll_s: float = ROLLOUT_POLL_S):
        self.kube, self.ledger, self._undo = kube, Path(ledger), undo
        self._clock, self._sleep = clock, sleep
        self.rollout_timeout_s, self.rollout_poll_s = rollout_timeout_s, rollout_poll_s
        self._used = False

    # --- decision -> operation ---------------------------------------------------
    def _refusal(self, action: str, component: str | None) -> str | None:
        """Why the action cannot run on this component; None when it can."""
        if not component or component == "none":
            return "no component named"
        if component in NODES:
            return "not a Deployment (a node)"
        if is_infra(component):
            return "infrastructure"
        if component not in {d["metadata"]["name"] for d in self.kube.list_deployments()}:
            return "not a Deployment in the namespace"
        return None

    def execute(self, action: str | None, component: str | None,
                baseline_replicas: dict[str, int]) -> dict:
        """Carry out one allowlisted action. Returns the action record. Raises
        ExecutorError on a second call or a read-back mismatch (the caller still resets)."""
        if self._used:
            raise ExecutorError("the executor already ran in this run (one action per run)")
        self._used = True
        out = {"action": action, "component": component, "executed": False,
               "not_executable": None, "operation": None, "t_action_start": None,
               "t_action_done": None, "rollout_complete": None}
        if action not in EXECUTABLE:
            out["not_executable"] = "not auto-remediated" if action in NOT_ACTED else \
                f"action {action!r} is not in the allowlist"
            return out
        why = self._refusal(action, component)
        if why:
            out["not_executable"] = why
            return out
        dep = self.kube.deployment(component)
        cur = int(dep["spec"].get("replicas", 1))
        if action == "scale_up":
            base = int(baseline_replicas.get(component, cur))
            target = base if cur < base else cur + 1
            if target > REPLICA_CAP:
                out["not_executable"] = "replica cap"
                return out
            out["operation"] = f"scale {component} {cur} -> {target}"
        elif action == "restart_service":
            out["operation"] = f"rollout restart {component}"
        else:
            out["operation"] = f"rollout undo {component}"
            out["revision_before"] = _revision(dep)

        saved = {"replicas": {component: cur},
                 "templates": {component: copy.deepcopy(dep["spec"]["template"])}}
        self._save(action, component, saved)
        out["saved"] = saved
        t0 = self._clock()
        out["t_action_start"] = _iso(t0)

        if action == "scale_up":
            self.kube.scale(component, target)
            got = self.kube.deployment(component)["spec"].get("replicas")
            if got != target:
                raise ExecutorError(f"read-back mismatch: {component} replicas {got}, wanted {target}")
            out["read_back"] = {"replicas": got}
        elif action == "restart_service":
            tpl = copy.deepcopy(dep["spec"]["template"])
            stamp = _iso(t0)
            tpl.setdefault("metadata", {}).setdefault("annotations", {})[RESTART_ANNOTATION] = stamp
            self.kube.replace_template(component, tpl)
            got = ((self.kube.deployment(component)["spec"]["template"].get("metadata") or {})
                   .get("annotations") or {}).get(RESTART_ANNOTATION)
            if got != stamp:
                raise ExecutorError(f"read-back mismatch: {component} restart annotation {got!r}")
            out["read_back"] = {RESTART_ANNOTATION: got}
        else:
            if self._undo is None:
                raise ExecutorError("rollback is not configured (no kubectl undo)")
            rc, text = self._undo(component)
            out["undo_output"] = text.strip()[-500:]
            if rc != 0:
                low = text.lower()
                if "no rollout history" in low or "unable to find" in low or "no revision" in low:
                    out["not_executable"] = "no previous revision"
                    self._clear()
                    out.pop("saved")
                    out["t_action_start"] = None
                    return out
                raise ExecutorError(f"rollout undo of {component} failed (rc={rc}): {text.strip()[-200:]}")
        out["executed"] = True
        if action == "scale_up":
            out["t_action_done"] = _iso(self._clock())
        else:
            out["rollout_complete"] = self._wait_rollout(component)
            out["t_action_done"] = _iso(self._clock())
            if action == "rollback_change":  # the controller bumps the revision during the rollout
                out["revision_after"] = _revision(self.kube.deployment(component))
        return out

    def _wait_rollout(self, name: str) -> bool:
        waited = 0.0
        while True:
            row = deployment_rows([self.kube.deployment(name)]).get(name)
            if row and row["rollout"] == "complete":
                return True
            if waited >= self.rollout_timeout_s:
                return False
            self._sleep(self.rollout_poll_s)
            waited += self.rollout_poll_s

    # --- ledger -------------------------------------------------------------------
    def _save(self, action: str, component: str, saved: dict) -> None:
        entry = json.loads(self.ledger.read_text()) if self.ledger.exists() else {}
        if entry.get("action"):
            raise ExecutorError(f"an action is already recorded in {self.ledger}; run reset-faults")
        entry["action"] = {"action": action, "target": component, "saved": saved,
                           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        _write_json_atomic(self.ledger, entry)

    def _clear(self) -> None:
        if not self.ledger.exists():
            return
        entry = json.loads(self.ledger.read_text())
        entry.pop("action", None)
        if entry:
            _write_json_atomic(self.ledger, entry)
        else:
            self.ledger.unlink()

    def restore(self) -> list[str]:
        """Undo the recorded action exactly (Tier 3 restore, with read-back), then drop it
        from the ledger. Runs before the fault's reset. Idempotent."""
        if not self.ledger.exists():
            return []
        entry = json.loads(self.ledger.read_text())
        act = entry.get("action")
        if not act:
            return []
        done = restore(self.kube, act["saved"])
        self._clear()
        return [f"action {act['action']} on {act['target']}: " + (", ".join(done) or "nothing to undo")]
