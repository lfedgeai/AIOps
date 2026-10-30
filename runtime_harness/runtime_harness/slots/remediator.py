"""Remediation slot — three mutually exclusive modes (HIL / Governed / Direct).

Purpose
-------
Convert a confirmed RCA into a remediation, per the active mode
(``remediation.get_mode``):

  hil       GLM proposes an action string; nothing executes (human decides).
  governed  GLM SELECTS a playbook from the reviewed catalog (+ params);
            the harness runs it via ansible-runner. Model never writes commands.
  direct    GLM SELECTS a guarded write tool (+ args); the harness executes
            it through write_tools (G1–G3) — requires AIOPS_RT_ENABLE_WRITE_TOOLS=1.

Architectural rule: the harness (never the model) owns execution.

How it connects
---------------
- Runs after RCA in ``orchestrator`` when RCA is confirmed.
- Model id from ``stage_models`` (GLM-5.3-Flash default).
- Response validated against the mode's schema (RemediationPlan /
  GovernedPlan / DirectPlan).
- Writes ``ctx["remediation"]`` for the verifier + cycle history audit.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from runtime_harness.models import Stage, StageResult
from runtime_harness.prompts import get_prompt_store
from runtime_harness.providers.llm import LLMProvider
from runtime_harness.remediation import (
    RemediationMode,
    execute_direct,
    get_mode,
    playbook_for_fault,
    run_playbook,
)
from runtime_harness.schemas import DirectPlan, GovernedPlan, RemediationPlan
from runtime_harness.slots.base import Slot, SlotContext
from runtime_harness.stage_models import get_stage_models


class LLMRemediator(Slot):
    """Action slot — mode-driven (hil | governed | direct)."""

    name = "remediator"
    plugin = "llm_remediate"

    def __init__(self, config: dict[str, Any], provider: LLMProvider):
        super().__init__(config)
        self.provider = provider

    def run(self, ctx: SlotContext) -> StageResult:
        started = datetime.now(timezone.utc)
        primary = get_stage_models().remediation_primary
        rca = ctx.get("rca") or {}
        mode = get_mode()
        store = get_prompt_store()
        system = store.get().remediate_system

        if mode == RemediationMode.HIL:
            return self._run_hil(ctx, primary, system, rca, started)
        if mode == RemediationMode.GOVERNED:
            return self._run_governed(ctx, primary, system, rca, started)
        return self._run_direct(ctx, primary, system, rca, started)

    # -- HIL: proposals only ------------------------------------------------

    def _run_hil(self, ctx, primary, system, rca, started) -> StageResult:
        user = (
            store_user()
            + f"\n\nHIL MODE — propose only, nothing will execute. Propose the "
            f"least-invasive remediation for the confirmed root cause.\n"
            f"rca={rca}\n"
            'Return ONLY JSON: {"action": "kubectl ...", "executed": false, '
            '"confidence": 0.0-1.0, "summary": "..."}'
        )
        out: RemediationPlan = self.provider.complete(primary, system, user, RemediationPlan)
        return self._finish(
            ctx,
            stage_result_kwargs=dict(
                summary=f"Remediation proposed (HIL): {out.action}",
            ),
            mode="hil",
            action=out.action or "no-op",
            proposed=bool(out.action),
            executed=False,
            detail_extra={"plan": out.model_dump()},
            agent_id=primary,
            started=started,
        )

    # -- Governed: Ansible playbook catalog ----------------------------------

    def _run_governed(self, ctx, primary, system, rca, started) -> StageResult:
        from runtime_harness.remediation import list_playbooks

        catalog = list_playbooks()
        fault = (rca.get("fault") or "").lower()
        default_pb = playbook_for_fault(fault)
        user = (
            store_user()
            + f"\n\nGOVERNED MODE — select a remediation playbook from the reviewed "
            f"catalog. You do NOT write commands; you choose a playbook and its "
            f"parameters.\n"
            f"catalog={catalog}\n"
            f"fault_type={fault or 'unknown'} → suggested={default_pb}\n"
            f"rca={rca}\n"
            'Return ONLY JSON: {"playbook": "...", "target": "...", '
            '"params": {"namespace": "...", "replicas": 1, "env_names": [...]}, '
            '"confidence": 0.0-1.0, "summary": "..."}'
        )
        out: GovernedPlan = self.provider.complete(primary, system, user, GovernedPlan)

        # Harness-side allowlist enforcement (never trust the model's selection)
        if out.playbook not in catalog:
            return self._finish(
                ctx,
                stage_result_kwargs=dict(
                    summary=f"Governed remediation refused: playbook '{out.playbook}' "
                    f"not in catalog {catalog} — proposal only",
                    ok=False,
                ),
                mode="governed",
                action=f"REFUSED playbook={out.playbook} (not in catalog)",
                proposed=True,
                executed=False,
                detail_extra={"plan": out.model_dump(), "catalog": catalog, "refused": True},
                agent_id=primary,
                started=started,
            )

        params = dict(out.params or {})
        params.setdefault("target", out.target or rca.get("component") or ctx.get("target"))
        params.setdefault("namespace", ctx.get("namespace", "otel-demo"))
        result = run_playbook(out.playbook, params)
        executed = bool(result.get("ok"))
        return self._finish(
            ctx,
            stage_result_kwargs=dict(
                summary=(
                    f"Governed remediation {'executed' if executed else 'FAILED'}: "
                    f"playbook {out.playbook} ({params.get('target')})"
                    + (f" — {result.get('error', '')}" if not executed else "")
                )
            ),
            mode="governed",
            action=f"ansible-playbook {out.playbook}.yml {params}",
            proposed=True,
            executed=executed,
            detail_extra={"plan": out.model_dump(), "playbook_result": result},
            agent_id=primary,
            started=started,
        )

    # -- Direct: guarded MCP/K8s tooling --------------------------------------

    def _run_direct(self, ctx, primary, system, rca, started) -> StageResult:
        from runtime_harness.write_tools import GUARDED_WRITE_TOOLS

        user = (
            store_user()
            + f"\n\nDIRECT MODE — select one guarded write tool and its arguments "
            f"for the confirmed root cause. Structured selection only.\n"
            f"allowed_tools={GUARDED_WRITE_TOOLS}\n"
            f"rca={rca}\n"
            'Return ONLY JSON: {"tool": "scale_deployment", '
            '"args": {"name": "cart", "replicas": 1, "namespace": "otel-demo"}, '
            '"confidence": 0.0-1.0, "summary": "..."}'
        )
        out: DirectPlan = self.provider.complete(primary, system, user, DirectPlan)
        result = execute_direct(out.tool, out.args or {})
        executed = bool(result.get("executed"))
        return self._finish(
            ctx,
            stage_result_kwargs=dict(
                summary=(
                    f"Direct remediation {'executed' if executed else 'NOT executed'}: "
                    f"{out.tool}({out.args}) — {result.get('result') or result.get('error', '')}"
                )
            ),
            mode="direct",
            action=f"{out.tool} {out.args}",
            proposed=True,
            executed=executed,
            detail_extra={"plan": out.model_dump(), "direct_result": result},
            agent_id=primary,
            started=started,
        )

    # -- shared ------------------------------------------------------------

    def _finish(
        self,
        ctx: SlotContext,
        *,
        stage_result_kwargs: dict[str, Any],
        mode: str,
        action: str,
        proposed: bool,
        executed: bool,
        detail_extra: dict[str, Any],
        agent_id: str,
        started,
    ) -> StageResult:
        finished = datetime.now(timezone.utc)
        ctx["remediation"] = {
            "mode": mode,
            "action": action,
            "proposed": proposed,
            "executed": executed,
            "agent_id": agent_id,
            **detail_extra,
        }
        defaults = dict(
            stage=Stage.REMEDIATE,
            ok=True,
            agent_id=agent_id,
            detail=ctx["remediation"],
            started_at=started,
            finished_at=finished,
            duration_s=(finished - started).total_seconds(),
        )
        defaults.update(stage_result_kwargs)
        return StageResult(**defaults)


def store_user() -> str:
    from runtime_harness.prompts import get_prompt_store

    return get_prompt_store().render_user()


def build_remediator(config: dict[str, Any], provider: LLMProvider) -> Slot:
    plugin = config.get("plugin", "llm_remediate")
    if plugin != "llm_remediate":
        raise ValueError(f"Unknown remediator plugin: {plugin}")
    return LLMRemediator(config, provider)
