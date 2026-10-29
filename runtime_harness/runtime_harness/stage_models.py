"""Stage → LLM model selection for Anomaly Detection, RCA, and Remediation.

Purpose
-------
Oct-12 evaluation showed a clear split: diagnose-only models (DeepSeek / Qwen3 /
Granite) vs executors (GLM-5.3-Flash / Qwen3.8). This module is the **single
control surface** for which agent id each pipeline stage uses.

How it connects
---------------
- Seeded from ``config/default_assembly.yaml`` → ``models:`` (preferred) or from
  legacy ``detector`` / ``rca`` / ``remediator`` primary ids.
- ``Assembly`` calls ``init_stage_models_from_config`` at wire-up time.
- Slots (``slots/detector.py``, ``rca.py``, ``remediator.py``) call
  ``get_stage_models()`` on each cycle so UI / API changes take effect immediately.
- Status UI + ``GET|PUT /models`` edit the in-memory store (same pattern as prompts).
- ``providers/llm.py`` maps each agent id to MaaS env vars (DEEPSEEK_*, GLM_*, …).

Do **not** assign diagnose-only models to remediation (0% exec on Oct 12).
"""

from __future__ import annotations

import threading
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# Catalog of models — Oct-26 consolidated findings (fixed metric pipeline).
# ``roles`` hint which stages each model is suited for; ``served`` records the
# actual model + gateway (labels lie — see EVALUATION_RESULTS_Oct26.md §6).
MODEL_CATALOG: list[dict[str, Any]] = [
    {
        "id": "maas_granite",
        "label": "Granite",
        "roles": ["anomaly_detection"],
        "served": "granite-3-2-8b-instruct @ RHDP MaaS (shared)",
        "notes": "Oct-26 Run A: detect 96%, RCA 85%, MTTD med 36s — fast detector",
    },
    {
        "id": "maas_glm-53-flash",
        "label": "GLM-5.3-Flash",
        "roles": ["anomaly_detection", "rca", "remediation"],
        "served": "glm-53-flash @ prelude-MaaS (dedicated)",
        "notes": "Oct-26: first near-complete agent — Run A 98/98/92, Run B 8/8 full wins",
    },
    {
        "id": "maas_deepseek",
        "label": "DeepSeek",
        "roles": ["anomaly_detection", "rca"],
        "served": "deepseek-r1-distill-qwen-14b @ RHDP MaaS (shared)",
        "notes": "Oct-26 Run A: detect 100%, RCA 90%, exec 0 — diagnose only",
    },
    {
        "id": "maas_qwen3",
        "label": "Qwen3",
        "roles": ["anomaly_detection", "rca"],
        "served": "qwen3-14b @ RHDP MaaS (shared)",
        "notes": "Oct-26 Run A: detect 94%, RCA 71%, exec 0 (namespace-trap casualty)",
    },
    {
        "id": "maas_qwen36-27b",
        "label": "Qwen3.8-27B (litemaas)",
        "roles": ["remediation"],
        "served": "Qwen3.8-27B @ LiteMaaS (dedicated)",
        "notes": "Oct-26 Run B: 8/8 full wins — executor alternate",
    },
    {
        "id": "maas_qwen38-27b",
        "label": "Qwen3.8-27B (rhdp)",
        "roles": ["remediation"],
        "served": "qwen38-27b @ RHDP MaaS (shared, slow)",
        "notes": "44% timeouts in Run A — avoid for round-hungry faults",
    },
]

DEFAULT_MODELS = {
    "anomaly_detection": {
        "primary": "maas_granite",
        "secondary": None,
        "alternate": None,
    },
    "rca": {
        "primary": "maas_glm-53-flash",
        "confirmer": None,
    },
    "remediation": {
        "primary": "maas_glm-53-flash",
        "alternate": None,
    },
}


@dataclass
class StageModelConfig:
    """Active model ids for the three LLM stages (+ optional confirmers)."""

    anomaly_detection_primary: str = "maas_granite"
    anomaly_detection_secondary: Optional[str] = None
    anomaly_detection_alternate: Optional[str] = None
    rca_primary: str = "maas_glm-53-flash"
    rca_confirmer: Optional[str] = None
    remediation_primary: str = "maas_glm-53-flash"
    remediation_alternate: Optional[str] = None
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_detection": {
                "primary": self.anomaly_detection_primary,
                "secondary": self.anomaly_detection_secondary,
                "alternate": self.anomaly_detection_alternate,
            },
            "rca": {
                "primary": self.rca_primary,
                "confirmer": self.rca_confirmer,
            },
            "remediation": {
                "primary": self.remediation_primary,
                "alternate": self.remediation_alternate,
            },
            "updated_at": self.updated_at,
            "source": self.source,
            "catalog": MODEL_CATALOG,
            "notes": (
                "Oct-26: Granite detects, GLM-5.3-Flash correlates/diagnoses and "
                "executes. Diagnose-only models are never assigned remediation."
            ),
        }

    def as_flat(self) -> dict[str, str]:
        return {k: v for k, v in asdict(self).items() if k not in ("updated_at", "source")}


class StageModelStore:
    """Thread-safe store for stage model selection (YAML seed + UI overrides)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cfg = StageModelConfig()
        self._defaults = StageModelConfig()

    def get(self) -> StageModelConfig:
        with self._lock:
            return deepcopy(self._cfg)

    def seed_from_assembly(self, config: dict[str, Any]) -> StageModelConfig:
        """Load models from assembly YAML ``models:`` or legacy slot sections."""
        models = config.get("models") or {}
        det = config.get("detector") or {}
        rca = config.get("rca") or {}
        rem = config.get("remediator") or {}

        ad = models.get("anomaly_detection") or {}
        rc = models.get("rca") or {}
        rm = models.get("remediation") or {}

        seeded = StageModelConfig(
            anomaly_detection_primary=ad.get("primary")
            or (det.get("primary") or {}).get("id")
            or DEFAULT_MODELS["anomaly_detection"]["primary"],
            anomaly_detection_secondary=ad.get("secondary")
            or (det.get("secondary") or {}).get("id")
            or DEFAULT_MODELS["anomaly_detection"]["secondary"],
            anomaly_detection_alternate=ad.get("alternate")
            or (det.get("alternate") or {}).get("id")
            or DEFAULT_MODELS["anomaly_detection"]["alternate"],
            rca_primary=rc.get("primary")
            or (rca.get("primary") or {}).get("id")
            or DEFAULT_MODELS["rca"]["primary"],
            rca_confirmer=rc.get("confirmer")
            or (rca.get("confirmer") or {}).get("id")
            or DEFAULT_MODELS["rca"]["confirmer"],
            remediation_primary=rm.get("primary")
            or (rem.get("primary") or {}).get("id")
            or DEFAULT_MODELS["remediation"]["primary"],
            remediation_alternate=rm.get("alternate")
            or (rem.get("alternate") or {}).get("id")
            or DEFAULT_MODELS["remediation"]["alternate"],
            source="assembly",
        )
        with self._lock:
            self._cfg = seeded
            self._defaults = deepcopy(seeded)
            return deepcopy(self._cfg)

    def update(
        self,
        *,
        anomaly_detection_primary: Optional[str] = None,
        anomaly_detection_secondary: Optional[str] = None,
        anomaly_detection_alternate: Optional[str] = None,
        rca_primary: Optional[str] = None,
        rca_confirmer: Optional[str] = None,
        remediation_primary: Optional[str] = None,
        remediation_alternate: Optional[str] = None,
        anomaly_detection: Optional[dict[str, str]] = None,
        rca: Optional[dict[str, str]] = None,
        remediation: Optional[dict[str, str]] = None,
    ) -> StageModelConfig:
        if anomaly_detection:
            anomaly_detection_primary = anomaly_detection.get(
                "primary", anomaly_detection_primary
            )
            anomaly_detection_secondary = anomaly_detection.get(
                "secondary", anomaly_detection_secondary
            )
            anomaly_detection_alternate = anomaly_detection.get(
                "alternate", anomaly_detection_alternate
            )
        if rca:
            rca_primary = rca.get("primary", rca_primary)
            rca_confirmer = rca.get("confirmer", rca_confirmer)
        if remediation:
            remediation_primary = remediation.get("primary", remediation_primary)
            remediation_alternate = remediation.get("alternate", remediation_alternate)

        with self._lock:
            if anomaly_detection_primary is not None:
                self._cfg.anomaly_detection_primary = anomaly_detection_primary.strip()
            if anomaly_detection_secondary is not None:
                self._cfg.anomaly_detection_secondary = (
                    anomaly_detection_secondary.strip()
                )
            if anomaly_detection_alternate is not None:
                self._cfg.anomaly_detection_alternate = (
                    anomaly_detection_alternate.strip()
                )
            if rca_primary is not None:
                self._cfg.rca_primary = rca_primary.strip()
            if rca_confirmer is not None:
                self._cfg.rca_confirmer = rca_confirmer.strip()
            if remediation_primary is not None:
                self._cfg.remediation_primary = remediation_primary.strip()
            if remediation_alternate is not None:
                self._cfg.remediation_alternate = remediation_alternate.strip()
            self._cfg.updated_at = datetime.now(timezone.utc).isoformat()
            self._cfg.source = "ui"
            return deepcopy(self._cfg)

    def reset(self) -> StageModelConfig:
        with self._lock:
            self._cfg = deepcopy(self._defaults)
            self._cfg.updated_at = datetime.now(timezone.utc).isoformat()
            self._cfg.source = "assembly"
            return deepcopy(self._cfg)


_STORE: Optional[StageModelStore] = None


def get_stage_model_store() -> StageModelStore:
    global _STORE
    if _STORE is None:
        _STORE = StageModelStore()
    return _STORE


def get_stage_models() -> StageModelConfig:
    return get_stage_model_store().get()


def init_stage_models_from_config(config: dict[str, Any]) -> StageModelConfig:
    return get_stage_model_store().seed_from_assembly(config)
