"""Remediation modes — four mutually exclusive configurations.

Purpose
-------
The graduated autonomy ladder, as one exclusive choice:

  hil            proposals only — the model suggests, a human decides (default)
  governed       execution via reviewed Ansible playbooks from a catalog
                 (the model SELECTS a playbook + params; it never writes commands)
  direct         execution via the guarded MCP/K8s write tools (structured tool
                 selection, G1–G3 guardrails) — requires arming via
                 ``AIOPS_RT_ENABLE_WRITE_TOOLS=1``
  image_rebuild  rebuild the faulty service's container image from source and
                 redeploy it (OpenShift BuildConfig → internal registry →
                 rollout) — the model only names the service; the reviewed
                 ``image_rebuild_deploy`` playbook does the build+deploy

How it connects
---------------
- Seeded from ``remediation.mode`` in the assembly YAML; live-switchable via
  ``GET|PUT /remediation`` and the UI's mutually exclusive buttons.
- ``slots/remediator.py`` reads the mode per cycle and validates the model's
  output against the mode's schema (RemediationPlan / GovernedPlan / DirectPlan /
  ImageRebuildPlan).
- Governed execution runs playbooks from ``playbooks/`` via ansible-runner.
- Image-rebuild execution runs the cataloged ``image_rebuild_deploy`` playbook
  via the same ansible-runner path (audit artifacts identical to governed).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Optional

PLAYBOOKS_DIR = Path(__file__).resolve().parent.parent / "playbooks"

# fault type → default playbook in the catalog
FAULT_PLAYBOOKS: dict[str, str] = {
    "scale_zero": "scale_restore",
    "kill_pod": "restart_deployment",
    "config_corruption": "patch_remove_env",
}


class RemediationMode(str, Enum):
    HIL = "hil"
    GOVERNED = "governed"
    DIRECT = "direct"
    IMAGE_REBUILD = "image_rebuild"


_MODE_LOCK = threading.Lock()
_MODE: RemediationMode = RemediationMode.HIL
_MODE_INITIALIZED = False


def get_mode() -> RemediationMode:
    with _MODE_LOCK:
        return _MODE


def set_mode(mode: str | RemediationMode) -> RemediationMode:
    global _MODE
    m = mode if isinstance(mode, RemediationMode) else RemediationMode(str(mode).lower())
    with _MODE_LOCK:
        _MODE = m
    return m


def init_mode_from_config(config: dict[str, Any]) -> RemediationMode:
    """Seed the mode from YAML once per process; live changes win afterwards."""
    global _MODE, _MODE_INITIALIZED
    with _MODE_LOCK:
        if _MODE_INITIALIZED:
            return _MODE
        _MODE = RemediationMode(str((config.get("remediation") or {}).get("mode", "hil")).lower())
        _MODE_INITIALIZED = True
        return _MODE


def mode_status() -> dict[str, Any]:
    from runtime_harness.write_tools import writes_armed

    return {
        "mode": get_mode().value,
        "modes": ["hil", "governed", "direct", "image_rebuild"],
        "descriptions": {
            "hil": "proposals only — model suggests, human decides",
            "governed": "execution via reviewed Ansible playbooks (model selects from catalog)",
            "direct": "execution via guarded MCP/K8s write tools",
            "image_rebuild": (
                "rebuild the faulty service's image from source and redeploy it "
                "(reviewed BuildConfig playbook; model names the service only)"
            ),
        },
        "direct_armed": writes_armed(),
        "playbooks": list_playbooks(),
        "fault_playbook_map": FAULT_PLAYBOOKS,
    }


# ---------------------------------------------------------------------------
# Playbook catalog + execution (governed mode)
# ---------------------------------------------------------------------------


def list_playbooks() -> list[str]:
    if not PLAYBOOKS_DIR.is_dir():
        return []
    return sorted(p.stem for p in PLAYBOOKS_DIR.glob("*.yml"))


def playbook_for_fault(fault: Optional[str]) -> Optional[str]:
    """Default playbook for a fault type (None → model must choose or propose)."""
    if not fault:
        return None
    key = fault.lower().strip()
    for k, pb in FAULT_PLAYBOOKS.items():
        if k in key:
            return pb if pb in list_playbooks() else None
    return None


def run_playbook(
    name: str, extravars: dict[str, Any], timeout_s: int = 240
) -> dict[str, Any]:
    """Run one catalog playbook via ansible-runner (or ansible-playbook).

    Returns {ok, rc, status, output_tail, playbook, extravars, runner,
    artifact_dir} — the artifact directory contains stdout/stderr, rc, status
    and the command, and is rendered in the remediate stage detail for audit.
    """
    path = PLAYBOOKS_DIR / f"{name}.yml"
    if not path.is_file():
        return {"ok": False, "error": f"playbook '{name}' not in catalog"}
    if name not in list_playbooks():
        return {"ok": False, "error": f"playbook '{name}' not allowed"}

    env = dict(os.environ)
    env["PATH"] = env.get("PATH", "") + f":{os.path.expanduser('~/.local/bin')}"

    try:
        import ansible_runner  # type: ignore

        result = ansible_runner.run(
            playbook=str(path),
            extravars=extravars,
            private_data_dir=None,
            quiet=True,
        )
        artifact_dir = getattr(getattr(result, "config", None), "artifact_dir", None)
        output = ""
        event_count = 0
        try:
            for ev in result.events or []:
                event_count += 1
                if ev.get("event") in ("runner_on_ok", "runner_on_failed", "runner_on_error", "runner_on_start"):
                    output += f"{ev.get('event')}: {json.dumps(ev.get('event_data', {}))[:300]}\n"
        except Exception:  # noqa: BLE001
            pass
        return {
            "ok": result.rc == 0 and result.status == "successful",
            "rc": result.rc,
            "status": result.status,
            "event_count": event_count,
            "output_tail": output[-1500:],
            "playbook": name,
            "extravars": extravars,
            "runner": "ansible-runner",
            "artifact_dir": artifact_dir,
            "stdout_path": f"{artifact_dir}/stdout" if artifact_dir else None,
        }
    except ImportError:
        # fallback: ansible-playbook subprocess
        cmd = [
            "ansible-playbook",
            str(path),
            "-i",
            "localhost,",
            "-c",
            "local",
        ]
        for k, v in extravars.items():
            cmd += ["-e", f"{k}={v}"]
        try:
            proc = subprocess.run(  # noqa: S603 — catalog allowlisted above
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )
            return {
                "ok": proc.returncode == 0,
                "rc": proc.returncode,
                "output_tail": (proc.stdout + proc.stderr)[-1500:],
                "playbook": name,
                "extravars": extravars,
                "runner": "ansible-playbook",
                "artifact_dir": None,
                "stdout_path": None,
            }
        except FileNotFoundError:
            return {
                "ok": False,
                "error": "neither ansible-runner nor ansible-playbook is installed",
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"playbook timed out after {timeout_s}s"}


# ---------------------------------------------------------------------------
# Direct mode execution (guarded MCP tooling)
# ---------------------------------------------------------------------------


def execute_direct(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a validated direct-mode tool selection through the guarded writes."""
    from runtime_harness.write_tools import GUARDED_WRITE_TOOLS, execute_write, writes_armed

    if not writes_armed():
        return {
            "ok": False,
            "executed": False,
            "error": (
                "direct mode is not armed — set AIOPS_RT_ENABLE_WRITE_TOOLS=1 to "
                "allow direct execution; recorded as proposal only"
            ),
        }
    if tool not in GUARDED_WRITE_TOOLS:
        return {"ok": False, "executed": False, "error": f"tool '{tool}' not in guarded allowlist"}
    result = execute_write(tool, args)
    return {
        "ok": not result.startswith("Error"),
        "executed": not result.startswith("Error"),
        "result": result,
        "tool": tool,
        "args": args,
    }


# ---------------------------------------------------------------------------
# Image-rebuild mode execution (reviewed BuildConfig playbook)
# ---------------------------------------------------------------------------

IMAGE_REBUILD_PLAYBOOK = "image_rebuild_deploy"
IMAGE_REBUILD_GIT_URL = "https://github.com/open-telemetry/opentelemetry-demo.git"
_SERVICE_RE = __import__("re").compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")


def execute_image_rebuild(
    service: str,
    namespace: str = "otel-demo",
    git_ref: str = "main",
    timeout_s: int = 900,
) -> dict[str, Any]:
    """Rebuild a service's container image from source and redeploy it.

    Runs the reviewed ``image_rebuild_deploy`` catalog playbook: create/apply
    an OpenShift BuildConfig (docker strategy, source = the pinned demo repo,
    contextDir = src/<service>), build to the internal registry, then roll the
    deployment onto the new image. The model's only input is the service name
    (+ optional git ref); the harness sanitizes it — no model-authored build
    commands ever run. Audit trail identical to governed mode (ansible-runner
    artifact dir rendered in the remediate stage detail).
    """
    service = str(service or "").strip().lower()
    if not _SERVICE_RE.match(service):
        return {
            "ok": False,
            "executed": False,
            "error": f"invalid service name '{service!r}' — must be a lowercase k8s name",
        }
    if IMAGE_REBUILD_PLAYBOOK not in list_playbooks():
        return {
            "ok": False,
            "executed": False,
            "error": f"playbook '{IMAGE_REBUILD_PLAYBOOK}' not in catalog",
        }
    extravars = {
        "namespace": namespace or "otel-demo",
        "target": service,
        "git_url": IMAGE_REBUILD_GIT_URL,
        "git_ref": str(git_ref or "main").strip() or "main",
        "build_timeout": timeout_s,
    }
    result = run_playbook(IMAGE_REBUILD_PLAYBOOK, extravars, timeout_s=timeout_s + 120)
    result["executed"] = bool(result.get("ok"))
    result["service"] = service
    result["namespace"] = extravars["namespace"]
    result["mode"] = "image_rebuild"
    return result


__all__ = [
    "RemediationMode",
    "get_mode",
    "set_mode",
    "init_mode_from_config",
    "mode_status",
    "list_playbooks",
    "playbook_for_fault",
    "run_playbook",
    "execute_direct",
    "execute_image_rebuild",
    "FAULT_PLAYBOOKS",
    "PLAYBOOKS_DIR",
    "IMAGE_REBUILD_PLAYBOOK",
]
