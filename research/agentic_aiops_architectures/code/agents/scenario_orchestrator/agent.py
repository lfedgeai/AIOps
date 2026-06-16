#!/usr/bin/env python3
"""
Scenario orchestrator: runs multi-agent scenarios (b, c) by invoking a base LLM
agent subprocess once per role with gated tool profiles, then merges results.

Scenario A is handled directly by the harness (single agent + TOOL_PROFILE).

Env:
  SCENARIO=b|c
  BASE_AGENT_SCRIPT=path/to/qwen3_agent/agent.py
  TOOL_PROFILE set per role by orchestrator
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.harness.scenarios import get_scenario
from code.tools.tool_profiles import allowed_tool_names


def _python() -> str:
    """Return real CPython for agent subprocesses (avoid IDE AppImage as sys.executable)."""
    for key in ("HARNESS_PYTHON", "PYTHON_BIN"):
        v = (os.environ.get(key) or "").strip()
        if v:
            return v
    exe = (sys.executable or "").lower()
    bad_substrings = ("cursor", "appimage", "electron", "code-insiders", "vscode")
    if exe and not any(b in exe for b in bad_substrings) and "python" in exe:
        return sys.executable
    for cand in ("python3.13", "python3.12", "python3"):
        w = shutil.which(cand)
        if w:
            return w
    return "python3"


@dataclass
class DetectionResult:
    detected: bool
    first_alert_time: str | None
    confidence: float
    suggested_root_cause: str | None
    suggested_remediations: list[str]
    signals: dict[str, Any]
    llm_invoked: bool = False


def _parse_json_stdout(raw: str) -> dict | None:
    if not raw or not raw.strip():
        return None
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _invoke_role(
    base_script: Path,
    since_ts: str,
    role_id: str,
    tool_profile: str,
    timeout: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["TOOL_PROFILE"] = tool_profile
    env["SCENARIO_ROLE"] = role_id
    env["PYTHONUNBUFFERED"] = "1"
    args = [_python(), "-u", str(base_script), "--since", since_ts, "--json"]
    print(
        f"[orchestrator] role={role_id} profile={tool_profile} tools={allowed_tool_names(tool_profile)}",
        file=sys.stderr,
        flush=True,
    )
    try:
        r = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(ROOT),
        )
        parsed = _parse_json_stdout(r.stdout or "")
        if parsed is None:
            return {
                "role": role_id,
                "tool_profile": tool_profile,
                "detected": False,
                "agent_error": (r.stderr or r.stdout or f"exit {r.returncode}")[:1500],
                "llm_invoked": False,
            }
        out = dict(parsed)
        out["role"] = role_id
        out["tool_profile"] = tool_profile
        out["allowed_tools"] = allowed_tool_names(tool_profile)
        if "llm_invoked" not in out:
            out["llm_invoked"] = False
        return out
    except subprocess.TimeoutExpired as e:
        return {
            "role": role_id,
            "tool_profile": tool_profile,
            "detected": False,
            "agent_error": f"timeout after {timeout}s",
            "llm_invoked": False,
        }


def merge_role_results(role_results: list[dict[str, Any]]) -> dict[str, Any]:
    detected = any(bool(r.get("detected")) for r in role_results)
    confidences = [float(r.get("confidence") or 0) for r in role_results]
    root_causes = [
        r.get("suggested_root_cause")
        for r in role_results
        if r.get("suggested_root_cause")
    ]
    remediations: list[str] = []
    for r in role_results:
        remediations.extend(r.get("suggested_remediations") or [])
    # Deduplicate preserving order
    seen: set[str] = set()
    unique_rems: list[str] = []
    for rem in remediations:
        if rem not in seen:
            seen.add(rem)
            unique_rems.append(rem)
    root_joined = ", ".join(dict.fromkeys(root_causes)) if root_causes else None
    return {
        "detected": detected,
        "confidence": max(confidences) if confidences else 0.0,
        "suggested_root_cause": root_joined,
        "suggested_remediations": unique_rems,
        "llm_invoked": any(r.get("llm_invoked") for r in role_results),
        "signals": {
            "scenario_roles": role_results,
            "scenario_mode": "multi",
        },
    }


def run_scenario(
    scenario_id: str,
    since_ts: str,
    base_script: Path,
    timeout: int,
) -> dict[str, Any]:
    spec = get_scenario(scenario_id)
    if spec.get("mode") != "multi":
        raise ValueError(f"Scenario {scenario_id} is not multi-agent mode")
    roles = spec.get("roles") or []
    results = []
    for role in roles:
        rid = role["id"]
        profile = role["tool_profile"]
        results.append(_invoke_role(base_script, since_ts, rid, profile, timeout))
    return merge_role_results(results)


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-agent scenario orchestrator")
    ap.add_argument("--since", required=True)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--scenario", default=os.environ.get("SCENARIO", "b"))
    args = ap.parse_args()

    scenario_id = args.scenario
    base = os.environ.get("BASE_AGENT_SCRIPT", "").strip()
    if not base:
        base = str(ROOT / "code" / "agents" / "qwen3_agent" / "agent.py")
    base_script = Path(base)
    timeout = int(os.environ.get("AGENT_TIMEOUT", "1200"))

    merged = run_scenario(scenario_id, args.since, base_script, timeout)
    detected = merged.get("detected", False)
    result = DetectionResult(
        detected=detected,
        first_alert_time=datetime.now(timezone.utc).isoformat() if detected else None,
        confidence=float(merged.get("confidence") or 0),
        suggested_root_cause=merged.get("suggested_root_cause"),
        suggested_remediations=merged.get("suggested_remediations") or [],
        signals=merged.get("signals") or {},
        llm_invoked=bool(merged.get("llm_invoked")),
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2), flush=True)
    return 0 if detected else 1


if __name__ == "__main__":
    sys.exit(main())
