"""RCA accuracy and remediation scoring against harness ground truth."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH_CONFIG = ROOT / "config" / "fault_ground_truth.yaml"

K8S_FAULT_TYPES = {"kill_pod", "scale_zero", "memory_limit"}


def _load_ground_truth_config() -> dict[str, Any]:
    if not GROUND_TRUTH_CONFIG.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(GROUND_TRUTH_CONFIG.read_text()) or {}
    except Exception:
        return {}


def normalize_component(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"-service$", "", s)
    s = re.sub(r"service$", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def split_root_causes(text: str | None) -> set[str]:
    if not text:
        return set()
    parts = re.split(r"[,;/]+|\band\b", text, flags=re.IGNORECASE)
    return {normalize_component(p) for p in parts if normalize_component(p)}


def expected_root_cause(fault_flag: str, fault_variant: str) -> tuple[str | None, str]:
    """Return (expected_root_cause, layer) for a fault injection."""
    cfg = _load_ground_truth_config()
    if fault_flag in K8S_FAULT_TYPES:
        block = (cfg.get("k8s_faults") or {}).get(fault_flag) or {}
        entry = block.get(fault_variant) or block.get("*") or {}
        rc = entry.get("root_cause") or fault_variant
        layer = entry.get("layer") or "platform"
        return normalize_component(rc) or None, layer

    block = (cfg.get("flagd_faults") or {}).get(fault_flag) or {}
    entry = block.get(fault_variant) or block.get("*") or {}
    rc = entry.get("root_cause")
    layer = entry.get("layer") or "application"
    return (normalize_component(rc) if rc else None), layer


def score_rca(
    fault_flag: str,
    fault_variant: str,
    suggested_root_cause: str | None,
) -> dict[str, Any]:
    expected, layer = expected_root_cause(fault_flag, fault_variant)
    predicted = split_root_causes(suggested_root_cause)
    if not expected:
        return {
            "expected_root_cause": None,
            "expected_layer": layer,
            "rca_correct": None,
            "rca_partial": None,
        }
    correct = expected in predicted
    return {
        "expected_root_cause": expected,
        "expected_layer": layer,
        "rca_correct": correct,
        "rca_partial": correct,
        "predicted_components": sorted(predicted),
    }


def score_remediation(
    fault_flag: str,
    fault_variant: str,
    suggested_remediations: list[str] | None,
) -> dict[str, Any]:
    expected, _layer = expected_root_cause(fault_flag, fault_variant)
    if not expected:
        return {"remediation_correct": None}
    text = " ".join(suggested_remediations or []).lower()
    norm_expected = normalize_component(expected)
    # Match deployment name or service substring in remediation text
    ok = norm_expected in re.sub(r"[^a-z0-9]+", "", text)
    return {"remediation_correct": ok}


def score_scenario_roles(
    fault_flag: str,
    fault_variant: str,
    agent_result: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Score RCA per orchestrator role (scenarios b/c). Returns [] for single-agent runs."""
    if not agent_result:
        return []
    signals = agent_result.get("signals") or {}
    roles = signals.get("scenario_roles")
    if not isinstance(roles, list) or not roles:
        return []

    scored: list[dict[str, Any]] = []
    for role in roles:
        if not isinstance(role, dict):
            continue
        role_id = str(role.get("role") or "unknown")
        rca = score_rca(fault_flag, fault_variant, role.get("suggested_root_cause"))
        rem = score_remediation(fault_flag, fault_variant, role.get("suggested_remediations"))
        scored.append(
            {
                "role": role_id,
                "tool_profile": role.get("tool_profile"),
                "detected": bool(role.get("detected")),
                "suggested_root_cause": role.get("suggested_root_cause"),
                "expected_root_cause": rca.get("expected_root_cause"),
                "rca_correct": rca.get("rca_correct"),
                "remediation_correct": rem.get("remediation_correct"),
                "predicted_components": rca.get("predicted_components"),
                "llm_invoked": bool(role.get("llm_invoked")),
                "agent_error": (role.get("agent_error") or "")[:500] or None,
            }
        )
    return scored
