"""Anomaly Detection slot — Granite with full telemetry query access.

Purpose
-------
Decide whether a fault is present AND name the suspect service. The detector
runs a bounded tool-loop (``agent.py``): it can call search_logs /
search_traces / search_metrics / count_errors_by_service / list_services /
get_deployment_config with since_ts windows — fleet-wide, not just the
focus service. Its ``component`` suspicion is handed to RCA.

How it connects
---------------
- Wired by ``assembly.build_detector``; run by ``orchestrator`` after telemetry.
- Model id from ``stage_models`` (Granite default).
- Response validated against ``schemas.DetectDecision`` (with ``component``).
- Writes ``ctx["detection"]`` for the RCA slot and incident view.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from runtime_harness.agent import ToolLoopAgent, build_tool_executor
from runtime_harness.models import Stage, StageResult
from runtime_harness.prompts import get_prompt_store
from runtime_harness.providers.llm import LLMProvider
from runtime_harness.schemas import DetectDecision
from runtime_harness.slots.base import Slot, SlotContext
from runtime_harness.stage_models import get_stage_models
from runtime_harness.telemetry.clickhouse import ClickHouseClient


class LLMDetector(Slot):
    """Detection slot — tool-loop investigation, configurable model."""

    name = "detector"
    plugin = "llm_detect"

    def __init__(self, config: dict[str, Any], provider: LLMProvider):
        super().__init__(config)
        self.provider = provider

    def run(self, ctx: SlotContext) -> StageResult:
        started = datetime.now(timezone.utc)
        models = get_stage_models()
        primary = models.anomaly_detection_primary
        store = get_prompt_store()
        system = store.get().detect_system
        signals = ctx.get("signals") or {}

        ch = ClickHouseClient()  # own instance: CLICKHOUSE_RT_HTTP / :38124 default
        agent = ToolLoopAgent(
            self.provider,
            build_tool_executor(ch, ctx.get("namespace", "otel-demo")),
            max_iterations=int(self.config.get("max_tool_iterations", 8)),
        )
        task = (
            store.render_user()
            + "\n\nDETECT STAGE. Investigate the fleet telemetry with the tools to "
            "decide whether a fault is currently present. Start with "
            "count_errors_by_service to see which service is failing, then drill "
            "into logs/traces/metrics and deployment config for the suspect.\n"
            "If the telemetry looks normal, return detected=false (component=null). "
            "If a fault is present, name the suspect service in 'component'.\n"
            f"adopted_services={ctx.get('adopted_services')}\n"
            f"focus={ctx.get('target')}\n"
            f"injected_fault={ctx.get('injected_fault')}\n"
            'Final JSON: {"detected": bool, "component": "service-or-null", '
            '"confidence": 0.0-1.0, "summary": "..."}'
        )
        decision: DetectDecision
        decision, trace = agent.run(primary, system, task, DetectDecision)

        ctx["detection"] = {
            "detected": decision.detected,
            "component": decision.component,
            "primary": decision.model_dump(),
            "primary_id": primary,
            "tool_calls": trace,
        }
        # Suspect from detection feeds the RCA focus (no re-discovery needed).
        if decision.component:
            ctx["suspect"] = decision.component
        finished = datetime.now(timezone.utc)
        return StageResult(
            stage=Stage.DETECT,
            ok=True,
            agent_id=primary,
            summary=(
                f"Detect={'yes' if decision.detected else 'no'}"
                f" component={decision.component or '—'}"
                f" (confidence={decision.confidence:.2f}, {len(trace)} tool calls)"
            ),
            detail=ctx["detection"],
            started_at=started,
            finished_at=finished,
            duration_s=(finished - started).total_seconds(),
        )


def build_detector(config: dict[str, Any], provider: LLMProvider) -> Slot:
    plugin = config.get("plugin", "llm_detect")
    if plugin != "llm_detect":
        raise ValueError(f"Unknown detector plugin: {plugin}")
    cfg = dict(config)
    cfg.setdefault("max_tool_iterations", 8)
    return LLMDetector(cfg, provider)
