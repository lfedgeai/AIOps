"""Orchestrator — staged AIOps control loop (honest metrics edition).

Purpose
-------
Run one incident cycle: enrich → telemetry (adopt apps) → anomaly detection →
RCA → remediate → verify — against **real** telemetry and **real** model
output. No fabricated fault evidence: healthy systems must report healthy.

MTTD/MTTR are measured from a resolved T0 (``faults.resolve_t0``):
  injected (drill) · telemetry-evidence (production) · cycle-start (fallback).

How it connects
---------------
- Holds an ``Assembly`` (wired slots + shared LLM provider).
- Model choice per LLM stage comes from ``stage_models``; prompts from
  ``prompts``; structured decisions validated via ``schemas.py``.
- Every cycle appends to ``history.get_history()`` for the trends UI.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from runtime_harness.assembly import Assembly
from runtime_harness.faults import FaultSpec, resolve_t0
from runtime_harness.history import get_history, new_cycle_record
from runtime_harness.models import (
    Health,
    IncidentView,
    RuntimeStatus,
    Stage,
    StageResult,
)
from runtime_harness.providers.llm import ProviderError
from runtime_harness.slots.base import SlotContext
from runtime_harness.status_store import StatusStore, get_store
from runtime_harness.telemetry.clickhouse import ClickHouseClient


class Orchestrator:
    """Staged detect → RCA → remediate → verify pipeline."""

    def __init__(self, assembly: Assembly, store: Optional[StatusStore] = None):
        self.assembly = assembly
        self.store = store or get_store()
        from runtime_harness.remediation import init_mode_from_config

        init_mode_from_config(assembly.config)  # hil | governed | direct (exclusive)

    def _publish(
        self,
        *,
        stage: Stage,
        health: Health,
        message: str,
        incident: IncidentView,
        stages: list[StageResult],
    ) -> None:
        from runtime_harness.remediation import get_mode

        mode = get_mode()
        self.store.set(
            RuntimeStatus(
                health=health,
                stage=stage,
                assembly=self.assembly.name,
                remediation_mode=mode.value,
                dry_run=mode.value == "hil",
                human_in_the_loop=mode.value == "hil",
                components=self.assembly.component_refs(),
                incident=incident,
                last_stages=stages[-20:],
                message=message,
                evidence_report=self.assembly.evidence_report(),
            )
        )

    @staticmethod
    def _error_stage(stage: Stage, message: str) -> StageResult:
        now = datetime.now(timezone.utc)
        return StageResult(
            stage=stage,
            ok=False,
            summary=message[:200],
            detail={"error": message},
            started_at=now,
            finished_at=now,
            duration_s=0.0,
        )

    def run_cycle(
        self,
        *,
        target: str = "",
        namespace: str = "otel-demo",
        context_c: Optional[str] = None,
        fault: Optional[FaultSpec] = None,
        t0: Optional[Any] = None,
        source: str = "manual",
    ) -> RuntimeStatus:
        a = self.assembly
        from runtime_harness.remediation import get_mode

        mode = get_mode()
        ctx = SlotContext(
            target=target or None,
            namespace=namespace,
            dry_run=mode.value == "hil",
            human_in_the_loop=mode.value == "hil",
            context_c=context_c,
        )
        if fault is not None:
            ctx["injected_fault"] = {
                "type": fault.type,
                "target": fault.target,
                "injected_at": fault.iso(),
            }

        stages: list[StageResult] = []
        incident = IncidentView()
        record = new_cycle_record(target or "(adopt)", "cycle-start")
        record["source"] = source
        t_cycle = datetime.now(timezone.utc)

        self._publish(
            stage=Stage.TELEMETRY,
            health=Health.UNKNOWN,
            message="Collecting telemetry + enrichment",
            incident=incident,
            stages=stages,
        )
        try:
            stages.append(a.enrichment.run(ctx))
            stages.append(a.telemetry.run(ctx))
        except Exception as exc:  # noqa: BLE001
            stages.append(self._error_stage(Stage.TELEMETRY, f"telemetry stage failed: {exc}"))
            return self._finish(Stage.ERROR, Health.ERROR, str(exc), incident, stages, record)

        # Resolve T0 for honest MTTD/MTTR (injected > trigger > evidence > cycle start)
        ch_cfg = a.config.get("telemetry") or {}
        from runtime_harness.telemetry.clickhouse import clickhouse_http_url_from_config

        ch_client = ClickHouseClient(
            base_url=clickhouse_http_url_from_config(ch_cfg),
            database=str(ch_cfg.get("database", "otel")),
        )
        focus = ctx.get("target")
        if fault is not None:
            t0_resolved = resolve_t0(fault, ch_client, focus, t_cycle)
        elif t0 is not None:
            t0_resolved = t0
        else:
            t0_resolved = resolve_t0(None, ch_client, focus, t_cycle)
        record["t0_source"] = t0_resolved.source
        record["target"] = focus or record["target"]
        ctx["t0"] = t0_resolved

        # -- detect ---------------------------------------------------------
        self._publish(
            stage=Stage.DETECT,
            health=Health.DEGRADED,
            message="Running anomaly detection (Granite)",
            incident=incident,
            stages=stages,
        )
        try:
            detect_res = a.detector.run(ctx)
        except ProviderError as exc:
            detect_res = self._error_stage(Stage.DETECT, str(exc))
        stages.append(detect_res)
        detection = ctx.get("detection") or {}
        incident.detected = bool(detection.get("detected"))
        record["detected"] = incident.detected
        if incident.detected:
            incident.mttd_s = (datetime.now(timezone.utc) - t0_resolved.ts).total_seconds()
            record["mttd_s"] = incident.mttd_s

        if not incident.detected:
            if not detect_res.ok and detect_res.detail.get("error"):
                return self._finish(
                    Stage.ERROR, Health.ERROR, detect_res.summary, incident, stages, record
                )
            return self._finish(
                Stage.COMPLETE,
                Health.HEALTHY,
                "No fault detected — system healthy",
                incident,
                stages,
                record,
            )

        # -- rca ------------------------------------------------------------
        self._publish(
            stage=Stage.RCA,
            health=Health.INCIDENT,
            message="Signal correlation + RCA (GLM-5.3-Flash)",
            incident=incident,
            stages=stages,
        )
        try:
            rca_res = a.rca.run(ctx)
        except ProviderError as exc:
            rca_res = self._error_stage(Stage.RCA, str(exc))
        stages.append(rca_res)
        rca = ctx.get("rca") or {}
        incident.component = rca.get("component")
        incident.fault_hypothesis = rca.get("fault")
        incident.rca_confirmed = bool(rca.get("confirmed"))
        record["rca_confirmed"] = incident.rca_confirmed
        record["rca_component"] = incident.component
        record["fault"] = incident.fault_hypothesis

        if not rca_res.ok:
            return self._finish(
                Stage.ERROR,
                Health.INCIDENT,
                "RCA not confirmed — remediator withheld",
                incident,
                stages,
                record,
            )

        # -- remediate --------------------------------------------------------
        self._publish(
            stage=Stage.REMEDIATE,
            health=Health.INCIDENT,
            message="Remediation (GLM-5.3-Flash, executor)",
            incident=incident,
            stages=stages,
        )
        try:
            rem_res = a.remediator.run(ctx)
        except ProviderError as exc:
            rem_res = self._error_stage(Stage.REMEDIATE, str(exc))
        stages.append(rem_res)
        rem = ctx.get("remediation") or {}
        incident.remediation_proposed = rem.get("action")
        incident.remediation_executed = bool(rem.get("executed"))
        record["remediation"] = rem.get("action")
        if rem.get("proposed") or rem.get("executed"):
            incident.mttr_s = (datetime.now(timezone.utc) - t0_resolved.ts).total_seconds()
            record["mttr_s"] = incident.mttr_s
            record["mttr_mode"] = "executed" if rem.get("executed") else "proposed"

        # -- verify -----------------------------------------------------------
        self._publish(
            stage=Stage.VERIFY,
            health=Health.INCIDENT,
            message="Verifying recovery",
            incident=incident,
            stages=stages,
        )
        ver_res = a.verifier.run(ctx)
        stages.append(ver_res)
        incident.recovery_verified = bool((ctx.get("verification") or {}).get("recovered"))

        health = Health.HEALTHY if incident.recovery_verified else Health.INCIDENT
        msg = (
            "Cycle complete — recovery verified"
            if incident.recovery_verified
            else "Cycle complete — awaiting recovery / human approval"
        )
        return self._finish(Stage.COMPLETE, health, msg, incident, stages, record)

    def _finish(
        self,
        stage: Stage,
        health: Health,
        message: str,
        incident: IncidentView,
        stages: list[StageResult],
        record: dict[str, Any],
    ) -> RuntimeStatus:
        record["health"] = health.value
        record["stage"] = stage.value
        record["error"] = message if health == Health.ERROR else None
        get_history().append(record)
        self._publish(
            stage=stage, health=health, message=message, incident=incident, stages=stages
        )
        return self.store.get()


def bootstrap_status(assembly: Assembly, store: Optional[StatusStore] = None) -> None:
    store = store or get_store()
    store.set(
        RuntimeStatus(
            health=Health.UNKNOWN,
            stage=Stage.IDLE,
            assembly=assembly.name,
            dry_run=bool((assembly.config.get("pipeline") or {}).get("dry_run", True)),
            human_in_the_loop=bool(
                (assembly.config.get("pipeline") or {}).get("human_in_the_loop", True)
            ),
            components=assembly.component_refs(),
            message="Ready — POST /cycle, or enable the scheduler",
            evidence_report=assembly.evidence_report(),
        )
    )
