"""Pydantic status / stage types for the API and UI.

Purpose
-------
Shared schemas: ``Stage``, ``StageResult``, ``RuntimeStatus``, ``IncidentView``.
(Not to be confused with ``stage_models.py``, which selects *which LLM* runs.)

How it connects
---------------
- Slots return ``StageResult``; orchestrator builds ``RuntimeStatus``.
- ``status_store`` holds the latest snapshot for ``GET /status``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Stage(str, Enum):
    IDLE = "idle"
    TELEMETRY = "telemetry"
    ENRICH = "enrich"
    DETECT = "detect"
    RCA = "rca"
    REMEDIATE = "remediate"
    VERIFY = "verify"
    COMPLETE = "complete"
    ERROR = "error"


class Health(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    INCIDENT = "incident"
    UNKNOWN = "unknown"
    ERROR = "error"


class ComponentRef(BaseModel):
    slot: str
    plugin: str
    agent_id: Optional[str] = None
    role: Optional[str] = None
    eval_notes: dict[str, Any] = Field(default_factory=dict)


class StageResult(BaseModel):
    stage: Stage
    ok: bool
    agent_id: Optional[str] = None
    summary: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: Optional[datetime] = None
    duration_s: Optional[float] = None


class IncidentView(BaseModel):
    detected: bool = False
    component: Optional[str] = None
    fault_hypothesis: Optional[str] = None
    rca_confirmed: bool = False
    remediation_proposed: Optional[str] = None
    remediation_executed: bool = False
    recovery_verified: bool = False
    mttd_s: Optional[float] = None
    mttr_s: Optional[float] = None


class RuntimeStatus(BaseModel):
    """Snapshot exposed by the status API / UI."""

    health: Health = Health.UNKNOWN
    stage: Stage = Stage.IDLE
    assembly: str = ""
    dry_run: bool = True
    human_in_the_loop: bool = True
    components: list[ComponentRef] = Field(default_factory=list)
    incident: IncidentView = Field(default_factory=IncidentView)
    last_stages: list[StageResult] = Field(default_factory=list)
    message: str = "Runtime not started"
    updated_at: datetime = Field(default_factory=utc_now)
    evidence_report: Optional[str] = None
