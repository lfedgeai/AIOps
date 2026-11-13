"""Assembly — YAML → wired pipeline slots.

Purpose
-------
Load ``config/default_assembly.yaml`` (or ``RUNTIME_HARNESS_CONFIG``) and construct
the runnable AIOps composition: telemetry, enrichment, anomaly detection, RCA,
remediation, verify.

How it connects
---------------
- CLI / API call ``load_assembly_config`` → ``Assembly(...)``.
- Seeds ``stage_models`` (Anomaly Detection / RCA / Remediation model ids).
- Instantiates slot plugins via ``slots/*/build_*`` factories.
- Shares one ``LLMProvider`` (mock | maas) across LLM slots.
- ``Orchestrator`` runs the assembled slots in stage order.
- ``component_refs()`` feeds the status UI "Assembled components" table.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from runtime_harness.models import ComponentRef
from runtime_harness.providers.llm import LLMProvider
from runtime_harness.slots.detector import build_detector
from runtime_harness.slots.enrichment import build_enrichment
from runtime_harness.slots.rca import build_rca
from runtime_harness.slots.remediator import build_remediator
from runtime_harness.slots.telemetry import build_telemetry
from runtime_harness.slots.verifier import build_verifier
from runtime_harness.stage_models import get_stage_models, init_stage_models_from_config

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            key, default = match.group(1), match.group(2)
            return os.getenv(key, default if default is not None else "")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "default_assembly.yaml"


def load_assembly_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = os.getenv("RUNTIME_HARNESS_CONFIG") or default_config_path()
    cfg_path = Path(path)
    with cfg_path.open() as f:
        raw = yaml.safe_load(f) or {}
    return _expand_env(raw)


def _sync_slot_ids_from_models(config: dict[str, Any]) -> dict[str, Any]:
    """Copy ``models.*`` ids into detector/rca/remediator sections for consistency."""
    models = config.get("models") or {}
    if not models:
        return config
    cfg = dict(config)
    if models.get("anomaly_detection"):
        det = dict(cfg.get("detector") or {})
        ad = models["anomaly_detection"]
        for key, role in (
            ("primary", "detect"),
            ("secondary", "detect_confirm"),
            ("alternate", "detect_fast"),
        ):
            if ad.get(key):
                block = dict(det.get(key) or {})
                block["id"] = ad[key]
                block.setdefault("role", role)
                det[key] = block
        cfg["detector"] = det
    if models.get("rca"):
        rca = dict(cfg.get("rca") or {})
        rc = models["rca"]
        if rc.get("primary"):
            block = dict(rca.get("primary") or {})
            block["id"] = rc["primary"]
            block.setdefault("role", "rca")
            rca["primary"] = block
        if rc.get("confirmer"):
            block = dict(rca.get("confirmer") or {})
            block["id"] = rc["confirmer"]
            block.setdefault("role", "rca_confirm")
            rca["confirmer"] = block
        cfg["rca"] = rca
    if models.get("remediation"):
        rem = dict(cfg.get("remediator") or {})
        rm = models["remediation"]
        if rm.get("primary"):
            block = dict(rem.get("primary") or {})
            block["id"] = rm["primary"]
            block.setdefault("role", "remediate_execute")
            rem["primary"] = block
        if rm.get("alternate"):
            block = dict(rem.get("alternate") or {})
            block["id"] = rm["alternate"]
            block.setdefault("role", "remediate_execute")
            rem["alternate"] = block
        cfg["remediator"] = rem
    return cfg


class Assembly:
    """Wires pluggable slots from YAML into a runnable pipeline."""

    def __init__(self, config: dict[str, Any]):
        self.config = _sync_slot_ids_from_models(config)
        # Runtime model store (UI can override without rewriting YAML).
        init_stage_models_from_config(self.config)
        backend = (self.config.get("providers") or {}).get("backend", "mock")
        self.provider = LLMProvider(backend=backend)
        self.telemetry = build_telemetry(self.config.get("telemetry") or {})
        self.enrichment = build_enrichment(self.config.get("enrichment") or {})
        self.detector = build_detector(self.config.get("detector") or {}, self.provider)
        self.rca = build_rca(self.config.get("rca") or {}, self.provider)
        self.remediator = build_remediator(
            self.config.get("remediator") or {}, self.provider
        )
        self.verifier = build_verifier(self.config.get("verifier") or {})

    @property
    def name(self) -> str:
        return str(self.config.get("name", "unnamed"))

    def component_refs(self) -> list[ComponentRef]:
        """Refs for status UI — prefer live stage_models over static YAML."""
        m = get_stage_models()
        refs: list[ComponentRef] = []
        # Non-LLM slots from config
        for slot_name, section in (
            ("telemetry", self.config.get("telemetry") or {}),
            ("enrichment", self.config.get("enrichment") or {}),
            ("verifier", self.config.get("verifier") or {}),
        ):
            refs.append(
                ComponentRef(
                    slot=slot_name,
                    plugin=str(section.get("plugin", "")),
                    agent_id=None,
                    role=None,
                    eval_notes={},
                )
            )
        # LLM stages — live model selection
        det = self.config.get("detector") or {}
        rca = self.config.get("rca") or {}
        rem = self.config.get("remediator") or {}
        refs.extend(
            [
                ComponentRef(
                    slot="anomaly_detection",
                    plugin=str(det.get("plugin", "llm_detect")),
                    agent_id=m.anomaly_detection_primary,
                    role="detect",
                    eval_notes=(det.get("primary") or {}).get("eval") or {},
                ),
                ComponentRef(
                    slot="anomaly_detection.secondary",
                    plugin=str(det.get("plugin", "llm_detect")),
                    agent_id=m.anomaly_detection_secondary,
                    role="detect_confirm",
                    eval_notes=(det.get("secondary") or {}).get("eval") or {},
                ),
                ComponentRef(
                    slot="rca",
                    plugin=str(rca.get("plugin", "llm_rca")),
                    agent_id=m.rca_primary,
                    role="rca",
                    eval_notes={},
                ),
                ComponentRef(
                    slot="rca.confirmer",
                    plugin=str(rca.get("plugin", "llm_rca")),
                    agent_id=m.rca_confirmer,
                    role="rca_confirm",
                    eval_notes={},
                ),
                ComponentRef(
                    slot="remediation",
                    plugin=str(rem.get("plugin", "llm_remediate")),
                    agent_id=m.remediation_primary,
                    role="remediate_execute",
                    eval_notes=(rem.get("primary") or {}).get("eval") or {},
                ),
            ]
        )
        return refs

    def evidence_report(self) -> str | None:
        ev = self.config.get("evidence") or {}
        return ev.get("report")
