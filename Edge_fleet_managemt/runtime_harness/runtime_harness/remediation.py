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
  image_rebuild  EDGE use case: rebuild the bootc (Bootable Container) OS image
                 and roll it out to the edge fleet via FlightCtl. The model only
                 names the target fleet; the harness owns the build/push/deploy
                 commands (``execute_image_rebuild`` — no model-authored commands
                 ever run)

How it connects
---------------
- Seeded from ``remediation.mode`` in the assembly YAML; live-switchable via
  ``GET|PUT /remediation`` and the UI's mutually exclusive buttons.
- ``slots/remediator.py`` reads the mode per cycle and validates the model's
  output against the mode's schema (RemediationPlan / GovernedPlan / DirectPlan).
- Governed execution runs playbooks from ``playbooks/`` via ansible-runner.
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
    """Seed the mode (and image-rebuild settings) from YAML once per process."""
    global _MODE, _MODE_INITIALIZED
    with _MODE_LOCK:
        rem_cfg = config.get("remediation") or {}
        if not _MODE_INITIALIZED:
            _MODE = RemediationMode(str(rem_cfg.get("mode", "hil").lower())
                                    if str(rem_cfg.get("mode", "hil")).lower()
                                    in ("hil", "governed", "direct", "image_rebuild")
                                    else "hil")
            _MODE_INITIALIZED = True
        ir = rem_cfg.get("image_rebuild") or {}
        if ir:
            _IMAGE_REBUILD_CFG.update(ir)
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
                "edge: rebuild the bootc OS image and roll it out to a FlightCtl "
                "fleet (model names the fleet only; harness runs build/push/deploy)"
            ),
            "image_rebuild_ready": image_rebuild_ready(),
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

    # Persistent artifact dir (audit): ansible-runner defaults to a /tmp mkdtemp
    # that is lost on reboot; keep stdout/stderr/rc/status/events under
    # data/ansible-artifacts/<ts>-<playbook>/ so playbook executions stay
    # reviewable after the fact.
    artifact_root = PLAYBOOKS_DIR.parent / "data" / "ansible-artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    from datetime import datetime as _dt
    private_dir = artifact_root / (
        _dt.now().strftime("%Y%m%d-%H%M%S") + f"-{name}"
    )

    try:
        import ansible_runner  # type: ignore

        result = ansible_runner.run(
            playbook=str(path),
            extravars=extravars,
            private_data_dir=str(private_dir),
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
# Image-rebuild mode (edge): bootc image rebuild + FlightCtl fleet rollout
# ---------------------------------------------------------------------------

import re as _re
import shutil as _shutil

_IMAGE_REBUILD_CFG: dict[str, Any] = {
    "flightctl_cli": "flightctl",       # CLI binary (env FLIGHTCTL_CLI overrides)
    "flightctl_args": [],               # extra CLI args, e.g. ["--context", "edge"]
    "fleet_allowlist": [],              # empty -> discover via `flightctl get fleets`
    "build_tool": "podman",             # bootc build backend (podman | buildah)
    "containerfile": "Containerfile",   # bootc Containerfile path (in context_dir)
    "context_dir": "",                  # build context (local dir; required w/o git_repo)
    "git_repo": "",                     # optional: git clone before build
    "registry": "",                     # push target, e.g. quay.io/org/edge-os (required)
    "tag_prefix": "aiops-rt",
    "build_timeout_s": 1200,
    "deploy_timeout_s": 300,
}
_FLEET_RE = _re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")


def _ir_cfg() -> dict[str, Any]:
    cfg = dict(_IMAGE_REBUILD_CFG)
    if os.environ.get("FLIGHTCTL_CLI"):
        cfg["flightctl_cli"] = os.environ["FLIGHTCTL_CLI"]
    if os.environ.get("AIOPS_RT_IMAGE_REGISTRY"):
        cfg["registry"] = os.environ["AIOPS_RT_IMAGE_REGISTRY"]
    return cfg


def image_rebuild_ready() -> bool:
    """True when the build tool + flightctl CLI + registry target are configured."""
    cfg = _ir_cfg()
    return bool(
        _shutil.which(cfg["flightctl_cli"])
        and _shutil.which(cfg["build_tool"])
        and cfg.get("registry")
        and (cfg.get("context_dir") or cfg.get("git_repo"))
    )


def _run_cmd(cmd: list[str], timeout_s: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(  # noqa: S603 — harness-owned, config-sourced command
            cmd, capture_output=True, text=True, timeout=timeout_s
        )
        return {
            "rc": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout_tail": (proc.stdout or "")[-1500:],
            "stderr_tail": (proc.stderr or "")[-1500:],
        }
    except FileNotFoundError:
        return {"rc": 127, "ok": False, "error": f"'{cmd[0]}' not found"}
    except subprocess.TimeoutExpired:
        return {"rc": 124, "ok": False, "error": f"timed out after {timeout_s}s"}


def _flightctl(cfg: dict[str, Any], args: list[str], timeout_s: int = 120) -> dict[str, Any]:
    cmd = [cfg["flightctl_cli"]] + list(cfg.get("flightctl_args") or []) + args
    return _run_cmd(cmd, timeout_s)


def list_fleets() -> list[str]:
    """Known FlightCtl fleets (allowlist from config, else discovered live)."""
    cfg = _ir_cfg()
    allow = cfg.get("fleet_allowlist") or []
    if allow:
        return list(allow)
    res = _flightctl(cfg, ["get", "fleets", "-o", "name"])
    if not res["ok"]:
        return []
    names = []
    for line in (res.get("stdout_tail") or "").splitlines():
        line = line.strip()
        if line.startswith("fleet/"):
            names.append(line[len("fleet/"):])
    return sorted(names)


def execute_image_rebuild(fleet: str, git_ref: str = "") -> dict[str, Any]:
    """Rebuild the bootc OS image and roll it out to a FlightCtl fleet (edge).

    Harness-owned pipeline — the model only chose ``fleet`` (+ optional source
    ref). Steps, all fail-loud (no mock):
      1. validate the fleet name + membership (config allowlist or live list)
      2. optional: git clone the bootc image source at ``git_ref``
      3. build the Bootable Container image (podman/buildah + the bootc
         Containerfile from config)
      4. push to the configured registry as <registry>:<tag_prefix>-<ts>
      5. patch the fleet's ``spec.template.spec.os.image`` to the new ref and
         ``flightctl apply`` — FlightCtl rolls it out to the enrolled devices
    Returns a full audit dict; ``executed`` is true only if every step passed.
    """
    cfg = _ir_cfg()
    fleet = str(fleet or "").strip().lower()
    if not _FLEET_RE.match(fleet):
        return {"ok": False, "executed": False, "error": f"invalid fleet name {fleet!r}"}
    fleets = list_fleets()
    if fleets and fleet not in fleets:
        return {
            "ok": False,
            "executed": False,
            "error": f"fleet '{fleet}' not in known fleets {fleets}",
        }
    if not (cfg.get("context_dir") or cfg.get("git_repo")):
        return {"ok": False, "executed": False,
                "error": "image_rebuild not configured: set remediation.image_rebuild.context_dir (or git_repo)"}
    if not cfg.get("registry"):
        return {"ok": False, "executed": False,
                "error": "image_rebuild not configured: set remediation.image_rebuild.registry"}

    steps: dict[str, Any] = {"fleet": fleet, "git_ref": git_ref or None}

    # (2) optional source checkout
    context_dir = cfg.get("context_dir") or ""
    if cfg.get("git_repo"):
        ref = git_ref or "main"
        clone_dir = f"/tmp/aiops-rt-bootc-{fleet}"
        _shutil.rmtree(clone_dir, ignore_errors=True)
        res = _run_cmd(["git", "clone", "--depth", "1", "--branch", ref,
                        cfg["git_repo"], clone_dir], 300)
        steps["git_clone"] = res
        if not res["ok"]:
            return {"ok": False, "executed": False, "error": "git clone failed", "steps": steps}
        context_dir = clone_dir

    # (3) build the bootc image
    from datetime import datetime, timezone as _tz
    tag = f"{cfg['tag_prefix']}-{datetime.now(_tz.utc).strftime('%Y%m%d-%H%M%S')}"
    image_ref = f"{cfg['registry']}:{tag}"
    build_tool = cfg.get("build_tool", "podman")
    build_cmd = {
        "podman": ["podman", "build", "-f", cfg.get("containerfile") or "Containerfile",
                   "-t", image_ref, context_dir],
        "buildah": ["buildah", "bud", "-f", cfg.get("containerfile") or "Containerfile",
                    "-t", image_ref, context_dir],
    }[build_tool]
    res = _run_cmd(build_cmd, int(cfg.get("build_timeout_s", 1200)))
    steps["build"] = res
    if not res["ok"]:
        return {"ok": False, "executed": False, "error": "bootc image build failed", "steps": steps}

    # (4) push
    res = _run_cmd([build_tool, "push", image_ref], int(cfg.get("build_timeout_s", 1200)))
    steps["push"] = res
    if not res["ok"]:
        return {"ok": False, "executed": False, "error": f"push of {image_ref} failed", "steps": steps}

    # (5) deploy: patch the fleet's os.image and apply
    get = _flightctl(cfg, ["get", "fleet", fleet, "-o", "yaml"])
    steps["fleet_get"] = get
    if not get["ok"]:
        return {"ok": False, "executed": False, "error": f"flightctl get fleet {fleet} failed", "steps": steps}
    import yaml as _yaml
    try:
        fleet_doc = _yaml.safe_load(get.get("stdout_tail") or "")
    except _yaml.YAMLError:
        fleet_doc = None
    if not isinstance(fleet_doc, dict):
        return {"ok": False, "executed": False, "error": "could not parse fleet YAML", "steps": steps}
    try:
        fleet_doc["spec"]["template"]["spec"]["os"]["image"] = image_ref
    except (KeyError, TypeError):
        return {"ok": False, "executed": False,
                "error": f"fleet {fleet} has no spec.template.spec.os.image to patch", "steps": steps}
    fleet_yaml = _yaml.safe_dump(fleet_doc)
    apply_cmd = [cfg["flightctl_cli"]] + list(cfg.get("flightctl_args") or []) + ["apply", "-f", "-"]
    try:
        proc = subprocess.run(  # noqa: S603 — harness-owned flightctl apply
            apply_cmd, input=fleet_yaml, capture_output=True, text=True,
            timeout=int(cfg.get("deploy_timeout_s", 300)),
        )
        steps["fleet_apply"] = {"rc": proc.returncode, "ok": proc.returncode == 0,
                                "stdout_tail": (proc.stdout or "")[-800:],
                                "stderr_tail": (proc.stderr or "")[-800:]}
    except FileNotFoundError:
        return {"ok": False, "executed": False, "error": "flightctl CLI not found", "steps": steps}
    except subprocess.TimeoutExpired:
        return {"ok": False, "executed": False, "error": "flightctl apply timed out", "steps": steps}
    if not steps["fleet_apply"]["ok"]:
        return {"ok": False, "executed": False, "error": "flightctl apply failed", "steps": steps}

    return {
        "ok": True,
        "executed": True,
        "fleet": fleet,
        "image_ref": image_ref,
        "steps": steps,
        "note": "FlightCtl is rolling the new bootc image out to the fleet's devices",
    }


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
    "image_rebuild_ready",
    "list_fleets",
    "FAULT_PLAYBOOKS",
    "PLAYBOOKS_DIR",
]
