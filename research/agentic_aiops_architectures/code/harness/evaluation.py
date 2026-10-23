"""RCA accuracy and remediation scoring against harness ground truth."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH_CONFIG = ROOT / "config" / "fault_ground_truth.yaml"

K8S_FAULT_TYPES = {
    "kill_pod", "scale_zero", "memory_limit", "network_partition",
    "readiness_probe_fail", "config_corruption", "pvc_full", "node_taint",
    "replica_overload", "dependency_removal",
}


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


def flatten_suggested_remediations(value: Any) -> list[str]:
    """Flatten nested remediation lists so scoring never crashes on list-of-lists."""
    out: list[str] = []

    def walk(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, str):
            out.append(node)
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
            return
        out.append(str(node))

    walk(value)
    return out


def score_remediation(
    fault_flag: str,
    fault_variant: str,
    suggested_remediations: list[str] | None,
) -> dict[str, Any]:
    """Score remediation from declared text only (legacy field: remediation_correct)."""
    expected, _layer = expected_root_cause(fault_flag, fault_variant)
    if not expected:
        return {"remediation_correct": None, "remediation_declared": None}
    flat = flatten_suggested_remediations(suggested_remediations)
    text = " ".join(flat).lower()
    norm_expected = normalize_component(expected)
    ok = norm_expected in re.sub(r"[^a-z0-9]+", "", text)
    return {"remediation_correct": ok, "remediation_declared": ok}


def _write_matches_fault_action(fault_flag: str, entry: dict[str, Any]) -> bool:
    """True when a successful write tool is an appropriate fix for the fault type."""
    if not entry.get("success"):
        return False
    tool = entry.get("tool") or ""
    args = entry.get("arguments") or {}
    if fault_flag == "scale_zero":
        return tool == "scale_deployment" and int(args.get("replicas") or 0) >= 1
    if fault_flag == "kill_pod":
        return tool in ("restart_deployment", "scale_deployment", "delete_pod")
    if fault_flag in ("memory_limit", "readiness_probe_fail", "config_corruption"):
        return tool in ("restart_deployment", "scale_deployment", "patch_deployment")
    if fault_flag == "network_partition":
        return tool in ("delete_pod", "restart_deployment", "scale_deployment")
    if fault_flag in ("pvc_full", "node_taint", "replica_overload", "dependency_removal"):
        return tool in ("restart_deployment", "scale_deployment", "delete_pod")
    return tool in ("restart_deployment", "scale_deployment", "delete_pod", "patch_deployment")


def first_effective_remediation_write(
    fault_flag: str,
    fault_variant: str,
    remediation_writes: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Earliest successful write that targets the expected component AND matches the fault action.

    Single source of truth for both `remediation_executed` scoring and `mttr_seconds`:
    MTTR must be present exactly when a fault-appropriate remediation was executed,
    so the two can never diverge (the old mttr_seconds counted any successful write,
    including wrong-target/wrong-action ones).
    """
    from code.harness.fault_verification import _write_targets_expected, _norm_target

    expected, _ = expected_root_cause(fault_flag, fault_variant)
    if not expected or not remediation_writes:
        return None
    target = _norm_target(fault_variant)
    if not target:
        return None
    candidates = [
        w
        for w in remediation_writes
        if w.get("success")
        and _write_matches_fault_action(fault_flag, w)
        and _write_targets_expected(w, target)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda w: w.get("completed_at") or w.get("started_at") or "")


def score_remediation_executed(
    fault_flag: str,
    fault_variant: str,
    remediation_writes: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Score remediation from successful write tools targeting the fault variant."""
    from code.harness.fault_verification import write_targeted_expected

    expected, _ = expected_root_cause(fault_flag, fault_variant)
    if not expected:
        return {"remediation_executed": None}
    writes = remediation_writes or []
    effective = first_effective_remediation_write(fault_flag, fault_variant, writes)
    targeted = write_targeted_expected(writes, fault_variant)
    action_ok = any(_write_matches_fault_action(fault_flag, w) for w in writes)
    executed = effective is not None
    return {
        "remediation_executed": executed,
        "remediation_write_targeted": targeted,
        "remediation_action_appropriate": action_ok,
        "remediation_executed_write_time": (effective or {}).get("completed_at")
        or (effective or {}).get("started_at"),
    }


def score_rca_inferred(
    fault_flag: str,
    fault_variant: str,
    suggested_root_cause: str | None,
    remediation_writes: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Infer RCA from declared text or successful targeted remediation writes."""
    from code.harness.fault_verification import write_targeted_expected

    expected, layer = expected_root_cause(fault_flag, fault_variant)
    if not expected:
        return {"rca_inferred": None}
    predicted = split_root_causes(suggested_root_cause)
    if expected in predicted:
        return {"rca_inferred": True, "rca_inferred_source": "declared"}
    writes = remediation_writes or []
    if write_targeted_expected(writes, fault_variant) and any(
        w.get("success") for w in writes
    ):
        return {"rca_inferred": True, "rca_inferred_source": "remediation_writes"}
    return {"rca_inferred": False, "rca_inferred_source": None}


def score_hybrid(
    fault_flag: str,
    fault_variant: str,
    agent_result: dict[str, Any],
    agent_signals: dict[str, Any],
    judge_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Combine declared, executed, inferred, and oracle-judge scores.
    Headline rca_correct / remediation_correct = any layer agrees.
    """
    rca = score_rca(fault_flag, fault_variant, agent_result.get("suggested_root_cause"))
    rem = score_remediation(
        fault_flag, fault_variant, agent_result.get("suggested_remediations")
    )
    executed = score_remediation_executed(
        fault_flag, fault_variant, agent_signals.get("remediation_writes")
    )
    inferred = score_rca_inferred(
        fault_flag,
        fault_variant,
        agent_result.get("suggested_root_cause"),
        agent_signals.get("remediation_writes"),
    )

    rca_declared = rca.get("rca_correct")
    remediation_declared = rem.get("remediation_declared")
    remediation_executed = executed.get("remediation_executed")
    rca_inferred = inferred.get("rca_inferred")

    judge = judge_result or {}
    rca_judge = judge.get("rca_judge_correct")
    remediation_judge = judge.get("remediation_judge_correct")

    def _combine(*vals: bool | None) -> bool | None:
        defined = [v for v in vals if v is not None]
        if not defined:
            return None
        return any(defined)

    return {
        **rca,
        **rem,
        **executed,
        **inferred,
        "rca_declared": rca_declared,
        "remediation_declared": remediation_declared,
        "rca_judge_correct": rca_judge,
        "remediation_judge_correct": remediation_judge,
        "recovery_appropriate": judge.get("recovery_appropriate"),
        "judge_rationale": judge.get("judge_rationale"),
        "judge_model": judge.get("judge_model"),
        "judge_error": judge.get("judge_error"),
        "judge_fallback_same_family": judge.get("judge_fallback_same_family"),
        "judge_skipped": judge.get("judge_skipped"),
        "rca_correct": _combine(rca_declared, rca_inferred, rca_judge),
        "remediation_correct": _combine(
            remediation_declared, remediation_executed, remediation_judge
        ),
    }


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
