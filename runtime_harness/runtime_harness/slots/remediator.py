"""Remediation (executor) slot — GLM-5.3 after confirmed RCA.

Purpose
-------
Propose (and, only when gates allow, mark for execution) a fix for a confirmed
root cause. Oct-26 primary executor: GLM-5.3-Flash (92% exec Run A, 8/8 Run B).
Architectural rule: never hand this stage to diagnose-only models.

How it connects
---------------
- Runs after RCA in ``orchestrator`` when RCA is confirmed.
- Model id from ``stage_models`` (YAML ``models.remediation`` / UI).
- Response validated against ``schemas.RemediationPlan``.
- Respects ``pipeline.dry_run`` + ``human_in_the_loop`` gates; the harness
  (never the model) owns execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from runtime_harness.models import Stage, StageResult
from runtime_harness.prompts import get_prompt_store
from runtime_harness.providers.llm import LLMProvider
from runtime_harness.schemas import RemediationPlan
from runtime_harness.slots.base import Slot, SlotContext
from runtime_harness.stage_models import get_stage_models


class LLMRemediator(Slot):
    """Action slot — configurable executor via stage_models."""

    name = "remediator"
    plugin = "llm_remediate"

    def __init__(self, config: dict[str, Any], provider: LLMProvider):
        super().__init__(config)
        self.provider = provider

    def run(self, ctx: SlotContext) -> StageResult:
        started = datetime.now(timezone.utc)
        primary = get_stage_models().remediation_primary
        dry_run = bool(ctx.get("dry_run", True))
        hil = bool(ctx.get("human_in_the_loop", True))
        rca = ctx.get("rca") or {}
        store = get_prompt_store()
        system = store.get().remediate_system
        user = (
            store.render_user()
            + "\n\nPropose the remediation for the confirmed root cause below. Prefer the "
            "least-invasive action that restores service (e.g. scale back up, remove a "
            "corrupt env var). The harness will execute it only if gates allow.\n"
            f"rca={rca}\nsignals={ctx.get('signals')}\n"
            f"dry_run={dry_run} human_in_the_loop={hil}\n"
            'Return ONLY JSON: {"action": "kubectl ...", "executed": false, '
            '"confidence": 0.0-1.0, "summary": "..."}'
        )
        out: RemediationPlan = self.provider.complete(primary, system, user, RemediationPlan)
        would_exec = bool(out.executed)
        executed = would_exec and not dry_run and not hil
        ctx["remediation"] = {
            "action": out.action or "no-op",
            "proposed": bool(out.action),
            "executed": executed,
            "dry_run": dry_run,
            "human_in_the_loop": hil,
            "agent_id": primary,
            "provider_result": out.model_dump(),
        }
        finished = datetime.now(timezone.utc)
        mode = (
            "executed"
            if executed
            else ("proposed (HIL/dry-run)" if dry_run or hil else "proposed")
        )
        return StageResult(
            stage=Stage.REMEDIATE,
            ok=True,
            agent_id=primary,
            summary=f"Remediation {mode}: {out.action}",
            detail=ctx["remediation"],
            started_at=started,
            finished_at=finished,
            duration_s=(finished - started).total_seconds(),
        )


def build_remediator(config: dict[str, Any], provider: LLMProvider) -> Slot:
    plugin = config.get("plugin", "llm_remediate")
    if plugin != "llm_remediate":
        raise ValueError(f"Unknown remediator plugin: {plugin}")
    return LLMRemediator(config, provider)
