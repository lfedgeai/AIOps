"""Oracle LLM judge: score agent RCA/remediation against harness ground truth."""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

from code.harness.evaluation import expected_root_cause
from code.harness.fault_verification import write_targeted_expected

_FAULT_DESCRIPTIONS: dict[str, str] = {
    "scale_zero": "Kubernetes deployment scaled to 0 replicas (service unavailable)",
    "kill_pod": "Pod(s) for the target deployment were deleted",
    "memory_limit": "Container memory limit set too low (OOM risk)",
    "network_partition": "NetworkPolicy blocking traffic to the target",
    "readiness_probe_fail": "Readiness probe configured to always fail",
    "config_corruption": "Invalid environment configuration injected",
    "pvc_full": "Disk filled inside the target pod",
    "node_taint": "Node taint preventing pod scheduling",
    "replica_overload": "Replica count or load-generator scaled abnormally",
    "dependency_removal": "Service dependency removed or broken",
}


def _judge_backend(
    model_env: str,
    model_default: str,
    base_env: str,
    key_env: str,
) -> tuple[str, str, str] | None:
    model = (os.environ.get(model_env) or model_default).strip()
    base = (os.environ.get(base_env) or "").strip()
    key = (os.environ.get(key_env) or "").strip()
    if model and base and key:
        return model, base, key
    return None


def _resolve_judge_candidates(agent_approach: str) -> list[tuple[str, str, str]]:
    """Ordered judge backends; prefer a different provider than the agent under test."""
    override_model = (os.environ.get("JUDGE_MODEL") or "").strip()
    override_base = (os.environ.get("JUDGE_API_BASE") or "").strip()
    override_key = (os.environ.get("JUDGE_API_KEY") or "").strip()
    if override_model and override_base and override_key:
        return [(override_model, override_base, override_key)]

    deepseek = _judge_backend(
        "DEEPSEEK_MODEL",
        "deepseek-r1-distill-qwen-14b",
        "DEEPSEEK_API_BASE",
        "DEEPSEEK_API_KEY",
    )
    qwen3 = _judge_backend(
        "QWEN3_MODEL", "qwen3-14b", "QWEN3_API_BASE", "QWEN3_API_KEY"
    )
    nemotron = _judge_backend(
        "NEMOTRON_MODEL",
        "nvidia/nemotron-3-nano",
        "NEMOTRON_API_BASE",
        "NEMOTRON_API_KEY",
    )

    approach = (agent_approach or "").lower()
    ordered: list[tuple[str, str, str] | None]
    if "qwen3" in approach or "qwen" in approach:
        ordered = [deepseek, nemotron, qwen3]
    elif "deepseek" in approach:
        ordered = [qwen3, nemotron, deepseek]
    elif "llama" in approach or "scout" in approach:
        ordered = [qwen3, nemotron, deepseek]
    elif "granite" in approach:
        ordered = [qwen3, deepseek, nemotron]
    else:
        # nemotron and others: cross-pair first, then nemotron fallback if MaaS is down
        ordered = [deepseek, qwen3, nemotron]

    seen: set[tuple[str, str]] = set()
    candidates: list[tuple[str, str, str]] = []
    for item in ordered:
        if not item:
            continue
        model, base, key = item
        sig = (model, base)
        if sig in seen:
            continue
        seen.add(sig)
        candidates.append(item)
    return candidates


def _fetch_tool_calls_from_mlflow(run_id: str, tracking_uri: str) -> list[dict[str, Any]]:
    if not run_id or not tracking_uri:
        return []
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri)
        for artifact_name in ("agent_llm_tool_calls.json", "artifacts/agent_llm_tool_calls.json"):
            try:
                local = client.download_artifacts(run_id, artifact_name)
                data = json.loads(open(local, encoding="utf-8").read())
                if isinstance(data, dict) and "tool_calls" in data:
                    return list(data["tool_calls"])
                if isinstance(data, list):
                    return data
            except Exception:
                continue
    except Exception:
        pass
    return []


def build_judge_evidence(
    *,
    fault_flag: str,
    fault_variant: str,
    agent_result: dict[str, Any],
    agent_signals: dict[str, Any],
    recovery_verified: bool,
    mlflow_run_id: str | None = None,
    mlflow_tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Assemble oracle + agent evidence for the judge prompt."""
    expected_rc, layer = expected_root_cause(fault_flag, fault_variant)
    tool_calls = _fetch_tool_calls_from_mlflow(
        mlflow_run_id or os.environ.get("MLFLOW_RUN_ID", ""),
        mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", ""),
    )
    rounds = agent_signals.get("ai_metrics_rounds") or []
    if not tool_calls and rounds:
        for rnd in rounds:
            for td in rnd.get("tool_details") or []:
                tool_calls.append({"name": td.get("name"), "note": "timing_only"})

    return {
        "injected_fault": {
            "fault_flag": fault_flag,
            "fault_variant": fault_variant,
            "expected_root_cause": expected_rc,
            "layer": layer,
            "description": _FAULT_DESCRIPTIONS.get(
                fault_flag, f"Harness fault type {fault_flag}"
            ),
        },
        "agent_declared": {
            "detected": agent_result.get("detected"),
            "suggested_root_cause": agent_result.get("suggested_root_cause"),
            "suggested_remediations": agent_result.get("suggested_remediations") or [],
            "confidence": agent_result.get("confidence"),
        },
        "remediation_writes": agent_signals.get("remediation_writes") or [],
        "recovery_verified": recovery_verified,
        "tool_calls": tool_calls[:40],
        "write_targeted_expected": write_targeted_expected(
            agent_signals.get("remediation_writes"), fault_variant
        ),
    }


def _render_evidence_text(evidence: dict[str, Any]) -> str:
    inj = evidence["injected_fault"]
    lines = [
        "=== INJECTED FAULT (oracle ground truth) ===",
        f"fault_flag: {inj['fault_flag']}",
        f"fault_variant (target): {inj['fault_variant']}",
        f"expected_root_cause: {inj['expected_root_cause']}",
        f"layer: {inj['layer']}",
        f"description: {inj['description']}",
        "",
        "=== AGENT DECLARED OUTPUT ===",
        json.dumps(evidence["agent_declared"], indent=2),
        "",
        f"recovery_verified (harness K8s poll): {evidence['recovery_verified']}",
        f"write_targeted_expected: {evidence['write_targeted_expected']}",
        "",
        "=== REMEDIATION WRITE TOOLS ===",
        json.dumps(evidence["remediation_writes"], indent=2)[:8000],
        "",
        "=== TOOL CALL LOG (summary) ===",
    ]
    for i, tc in enumerate(evidence.get("tool_calls") or [], 1):
        name = tc.get("name", "?")
        args = tc.get("arguments", {})
        preview = str(tc.get("result_preview", ""))[:500]
        lines.append(f"#{i} {name} args={json.dumps(args, default=str)[:400]}")
        if preview:
            lines.append(f"   result: {preview[:400]}")
    return "\n".join(lines)


def _parse_judge_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def run_oracle_judge(
    *,
    fault_flag: str,
    fault_variant: str,
    agent_approach: str,
    agent_result: dict[str, Any],
    agent_signals: dict[str, Any],
    recovery_verified: bool,
    mlflow_run_id: str | None = None,
    mlflow_tracking_uri: str | None = None,
) -> dict[str, Any]:
    """
    Call a separate LLM to score RCA/remediation against injected fault (oracle).
    Returns dict with rca_judge_correct, remediation_judge_correct, rationale, judge_model, etc.
    """
    empty: dict[str, Any] = {
        "rca_judge_correct": None,
        "remediation_judge_correct": None,
        "recovery_appropriate": None,
        "judge_rationale": None,
        "judge_model": None,
        "judge_error": None,
        "judge_skipped": True,
    }
    if os.environ.get("HARNESS_DISABLE_JUDGE", "").lower() in ("1", "true", "yes"):
        empty["judge_skipped"] = True
        empty["judge_error"] = "disabled via HARNESS_DISABLE_JUDGE"
        return empty

    resolved_list = _resolve_judge_candidates(agent_approach)
    if not resolved_list:
        empty["judge_error"] = "no judge API credentials"
        return empty

    evidence = build_judge_evidence(
        fault_flag=fault_flag,
        fault_variant=fault_variant,
        agent_result=agent_result,
        agent_signals=agent_signals,
        recovery_verified=recovery_verified,
        mlflow_run_id=mlflow_run_id,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    evidence_text = _render_evidence_text(evidence)

    system = (
        "You are an evaluation judge for an AIOps harness experiment. "
        "The harness injected a KNOWN fault (oracle ground truth). "
        "Score whether the agent's diagnosis and remediation were appropriate for THAT fault. "
        "Base your verdict on declared output AND tool actions (remediation_writes, tool log). "
        "Reply with JSON only, no markdown."
    )
    user = f"""{evidence_text}

Score the agent response. Examples:
- scale_zero on cart: scaling cart deployment back to >=1 replica is correct remediation.
- Wrong target (e.g. fixing load-generator when cart was faulted) = remediation_judge_correct false.

Return JSON:
{{
  "rca_judge_correct": true or false,
  "remediation_judge_correct": true or false,
  "recovery_appropriate": true or false,
  "rationale": "one short paragraph"
}}"""

    from openai import OpenAI

    errors: list[str] = []
    for model, api_base, api_key in resolved_list:
        try:
            client = OpenAI(api_key=api_key, base_url=api_base.rstrip("/"))
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=800,
            )
            raw = (resp.choices[0].message.content or "").strip()
            parsed = _parse_judge_json(raw)
            if not parsed:
                err = f"{model}: failed to parse judge JSON"
                errors.append(err)
                print(f"[harness] oracle judge warning: {err}", file=sys.stderr, flush=True)
                continue
            approach = (agent_approach or "").lower()
            same_family = (
                ("nemotron" in approach and "nemotron" in model.lower())
                or ("deepseek" in approach and "deepseek" in model.lower())
                or (("qwen" in approach) and ("qwen" in model.lower()))
            )
            return {
                "rca_judge_correct": bool(parsed.get("rca_judge_correct")),
                "remediation_judge_correct": bool(parsed.get("remediation_judge_correct")),
                "recovery_appropriate": bool(parsed.get("recovery_appropriate")),
                "judge_rationale": str(parsed.get("rationale", ""))[:4000],
                "judge_model": model,
                "judge_api_base": api_base,
                "judge_fallback_same_family": same_family,
                "judge_error": None,
                "judge_skipped": False,
                "judge_raw_response": raw[:4000],
                "judge_evidence": evidence,
            }
        except Exception as e:
            err = f"{model}: {type(e).__name__}: {str(e)[:200]}"
            errors.append(err)
            print(f"[harness] oracle judge warning: {err}", file=sys.stderr, flush=True)

    return {
        **empty,
        "judge_skipped": False,
        "judge_model": resolved_list[-1][0] if resolved_list else None,
        "judge_error": "; ".join(errors)[:500] if errors else "all judge backends failed",
    }
