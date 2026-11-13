"""Root-cause analysis slot — GLM-5.3 signal correlation with query access.

Purpose
-------
Name the failing workload and fault type by correlating logs / traces /
metrics (and deployment config) — with the same tool-loop access as the
detector. Starts from the detector's ``component`` suspicion instead of
re-discovering it.

How it connects
---------------
- Runs after anomaly detection in ``orchestrator`` (gated on detected=true).
- Response validated against ``schemas.RcaDecision``.
- Writes ``ctx["rca"]``; remediator only proceeds when confirmed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from runtime_harness.agent import ToolLoopAgent, build_tool_executor
from runtime_harness.models import Stage, StageResult
from runtime_harness.prompts import get_prompt_store
from runtime_harness.providers.llm import LLMProvider
from runtime_harness.schemas import RcaDecision
from runtime_harness.slots.base import Slot, SlotContext
from runtime_harness.stage_models import get_stage_models
from runtime_harness.telemetry.clickhouse import ClickHouseClient


class LLMRCA(Slot):
    """RCA slot — signal correlation with full query access."""

    name = "rca"
    plugin = "llm_rca"

    def __init__(self, config: dict[str, Any], provider: LLMProvider):
        super().__init__(config)
        self.provider = provider

    def run(self, ctx: SlotContext) -> StageResult:
        started = datetime.now(timezone.utc)
        models = get_stage_models()
        primary = models.rca_primary
        store = get_prompt_store()
        system = store.get().rca_system
        detection = ctx.get("detection") or {}
        suspect = detection.get("component") or ctx.get("target")

        ch = ClickHouseClient()  # own instance: CLICKHOUSE_RT_HTTP / :38124 default
        agent = ToolLoopAgent(
            self.provider,
            build_tool_executor(ch, ctx.get("namespace", "otel-demo")),
            max_iterations=int(self.config.get("max_tool_iterations", 8)),
        )
        task = (
            store.render_user()
            + "\n\nRCA STAGE — signal correlation. The detector suspects: "
            f"'{suspect}'. Verify and refine: correlate logs, traces and metrics "
            "for the suspect (and its callers), and check its deployment config "
            "(replicas / env) to determine the fault type (e.g. scale_zero, "
            "config_corruption, crashloop, dependency failure).\n"
            f"detection={detection.get('primary')}\n"
            f"adopted_services={ctx.get('adopted_services')}\n"
            'Final JSON: {"component": "...", "fault": "...", '
            '"confidence": 0.0-1.0, "summary": "..."}'
        )
        decision: RcaDecision
        decision, trace = agent.run(primary, system, task, RcaDecision)

        ctx["rca"] = {
            "component": decision.component or suspect,
            "fault": decision.fault or "unknown",
            "confirmed": bool(decision.component),
            "primary": decision.model_dump(),
            "primary_id": primary,
            "tool_calls": trace,
        }
        finished = datetime.now(timezone.utc)
        return StageResult(
            stage=Stage.RCA,
            ok=bool(decision.component),
            agent_id=primary,
            summary=f"RCA: {decision.component}/{decision.fault} "
            f"({len(trace)} tool calls)",
            detail=ctx["rca"],
            started_at=started,
            finished_at=finished,
            duration_s=(finished - started).total_seconds(),
        )


def build_rca(config: dict[str, Any], provider: LLMProvider) -> Slot:
    plugin = config.get("plugin", "llm_rca")
    if plugin != "llm_rca":
        raise ValueError(f"Unknown rca plugin: {plugin}")
    cfg = dict(config)
    cfg.setdefault("max_tool_iterations", 8)
    return LLMRCA(cfg, provider)
