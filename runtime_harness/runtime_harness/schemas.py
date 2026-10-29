"""Structured stage decisions — the response contract for every LLM stage.

Purpose
-------
Single source of truth for what each pipeline stage's model must return.
Both the provider (parsing/validation) and the slots (consumption) use these
schemas. The eval-harness lesson: never let a stage silently default when a
model's output doesn't parse — surface the error.

How it connects
---------------
- ``providers/llm.py`` validates ``/chat/completions`` content against these.
- ``slots/detector.py|rca.py|remediator.py`` consume validated instances.
- ``api.py`` exposes parse failures in stage results (health=error, not fake ok).
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError

class DetectDecision(BaseModel):
    """Anomaly detection stage output.

    ``component`` lets the detector share WHICH service it believes is
    anomalous — the RCA stage starts from that suspicion instead of
    re-discovering it.
    """

    detected: bool
    component: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = ""


class RcaDecision(BaseModel):
    """Signal correlation + root-cause stage output."""

    component: str
    fault: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = ""


class RemediationPlan(BaseModel):
    """Remediation stage output (proposal; execution is harness-gated)."""

    action: str
    executed: bool = False  # model claim only — harness gates real execution
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = ""


T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(content: str) -> Optional[dict[str, Any]]:
    """Best-effort JSON object extraction from model text."""
    if not content:
        return None
    text = content.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = _OBJ_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def validate(content: str, schema: Type[T]) -> T:
    """Parse + validate; raises ValidationError on failure."""
    obj = extract_json(content)
    if obj is None:
        raise ValueError("no JSON object found in model output")
    return schema.model_validate(obj)


__all__ = [
    "DetectDecision",
    "RcaDecision",
    "RemediationPlan",
    "extract_json",
    "validate",
]
