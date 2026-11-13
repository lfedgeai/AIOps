"""Shared agent prompts — Oct-12 leading practices.

Purpose
-------
Keep task framing identical across Anomaly Detection / RCA / Remediation agents
(matrix fairness). Prompts stay application-agnostic; apps come from telemetry.

How it connects
---------------
- Seeded with research-style ``BASE_SYSTEM_PROMPT`` defaults.
- Status UI + ``GET|PUT /prompts`` edit the in-memory ``PromptStore``.
- Slots read stage overlays (``detect_system``, ``rca_system``, ``remediate_system``)
  on each cycle via ``get_prompt_store()``.
- Orthogonal to ``stage_models`` (which model) — this module is *what we ask*.
"""

from __future__ import annotations

import threading
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# Core framing from Oct-12 shared prompts (diagnose/detect/remediate/report).
# Intentionally generic — apps are adopted from telemetry, not named here.
DEFAULT_SYSTEM_PROMPT = """You are an autonomous AIOps agent.
Your job is to DETECT faults, DIAGNOSE root cause, FIX the problem, and REPORT what you did.

Use ONLY the available tools (logs, metrics, traces, and optional context corpora).
Do not invent tools you cannot call.

Output final JSON only (no markdown fences, no prose):
{"detected": true|false, "confidence": 0.0-1.0, "summary": "..."}

Rules:
- detected=false only when available signals look normal for the fault window.
- Prefer diagnosing with high-signal context (docs, architecture, dependencies) when configured; telemetry-only (C0) is weak for RCA.
- Do not ask diagnose-only models to execute; hand confirmed RCA to an executor.
- Infer workload names and topology only from tool results (and optional RAG/context if enabled).
  Do not assume a particular application or service inventory — applications are self-adopted from telemetry.
"""

DEFAULT_USER_PROMPT_TEMPLATE = (
    "Analyze telemetry since {since_ts}. Use tools to search logs, traces, "
    "and metrics. Determine if there is a failure, remediate if you can, and "
    "verify the fix before finishing. Respond with JSON when done."
)

# Stage-specific overlays (still generic). Operators can edit via UI.
DEFAULT_DETECT_SYSTEM = (
    DEFAULT_SYSTEM_PROMPT
    + "\n\nYou are in the DETECT stage. Confirm whether a fault is present."
)
DEFAULT_RCA_SYSTEM = (
    DEFAULT_SYSTEM_PROMPT
    + "\n\nYou are in the RCA stage. Name the failing workload and fault type from evidence."
)
DEFAULT_REMEDIATE_SYSTEM = (
    DEFAULT_SYSTEM_PROMPT
    + "\n\nYou are in the REMEDIATE (executor) stage. Act only on a confirmed RCA; "
    "respect dry-run / human-in-the-loop gates."
)


@dataclass
class PromptBundle:
    """Editable prompt set exposed to the UI and LLM slots."""

    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE
    detect_system: str = DEFAULT_DETECT_SYSTEM
    rca_system: str = DEFAULT_RCA_SYSTEM
    remediate_system: str = DEFAULT_REMEDIATE_SYSTEM
    notes: str = (
        "Oct-12 practices: shared generic prompts; apps via telemetry; "
        "prefer CONTEXT_C docs/architecture/deps; diagnose≠execute."
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PromptStore:
    """Thread-safe in-memory prompt store (UI/API editable)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bundle = PromptBundle()

    def get(self) -> PromptBundle:
        with self._lock:
            return deepcopy(self._bundle)

    def update(
        self,
        *,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        detect_system: Optional[str] = None,
        rca_system: Optional[str] = None,
        remediate_system: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> PromptBundle:
        with self._lock:
            if system_prompt is not None:
                new_sys = system_prompt.strip()
                # Keep stage overlays aligned when only the core system prompt changes.
                if detect_system is None and self._bundle.detect_system.startswith(
                    self._bundle.system_prompt
                ):
                    suffix = self._bundle.detect_system[len(self._bundle.system_prompt) :]
                    self._bundle.detect_system = new_sys + suffix
                if rca_system is None and self._bundle.rca_system.startswith(
                    self._bundle.system_prompt
                ):
                    suffix = self._bundle.rca_system[len(self._bundle.system_prompt) :]
                    self._bundle.rca_system = new_sys + suffix
                if remediate_system is None and self._bundle.remediate_system.startswith(
                    self._bundle.system_prompt
                ):
                    suffix = self._bundle.remediate_system[
                        len(self._bundle.system_prompt) :
                    ]
                    self._bundle.remediate_system = new_sys + suffix
                self._bundle.system_prompt = new_sys
            if user_prompt_template is not None:
                self._bundle.user_prompt_template = user_prompt_template.strip()
            if detect_system is not None:
                self._bundle.detect_system = detect_system.strip()
            if rca_system is not None:
                self._bundle.rca_system = rca_system.strip()
            if remediate_system is not None:
                self._bundle.remediate_system = remediate_system.strip()
            if notes is not None:
                self._bundle.notes = notes.strip()
            self._bundle.updated_at = datetime.now(timezone.utc).isoformat()
            self._bundle.source = "ui"
            return deepcopy(self._bundle)

    def reset(self) -> PromptBundle:
        with self._lock:
            self._bundle = PromptBundle()
            return deepcopy(self._bundle)

    def render_user(self, since_ts: Optional[str] = None) -> str:
        ts = since_ts or datetime.now(timezone.utc).isoformat()
        with self._lock:
            tmpl = self._bundle.user_prompt_template
        try:
            return tmpl.format(since_ts=ts)
        except (KeyError, ValueError):
            return tmpl.replace("{since_ts}", ts)


_STORE: Optional[PromptStore] = None


def get_prompt_store() -> PromptStore:
    global _STORE
    if _STORE is None:
        _STORE = PromptStore()
    return _STORE
