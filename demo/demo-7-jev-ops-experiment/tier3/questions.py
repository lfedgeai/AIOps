"""Tier 3's question set, shared by every engine.

No engine-specific code here: the Jev adapter and the Claude baseline both ask exactly
these four questions with these options. IDs, types and option keys are fixed here from
tier3/vocab.py; the wording comes from a prompt variant (prompts/tier3/*.yaml). Tier 4
uses the same variants. QUESTIONS is the baseline, exactly what was sent before variants
existed (tests/fixtures/questions-tier3-pre-variants.json).
"""
from __future__ import annotations

from prompts.loader import DEFAULT_VARIANT, NOUL_OPTIONS, QuestionSpec, Variant, load_variant
from tier3.vocab import ACTIONS, COMPONENTS, DIAGNOSES, NO_COMPONENT, NODES, SERVICES

QUESTION_SCHEMA = {
    "incident_detected": QuestionSpec("noul", NOUL_OPTIONS),
    "affected_component": QuestionSpec("choice", (NO_COMPONENT, *COMPONENTS),
                                       {"services": SERVICES, "nodes": NODES}),
    "diagnosis": QuestionSpec("choice", DIAGNOSES),
    "recommended_action": QuestionSpec("choice", ACTIONS),
}


def load_prompt(name: str = DEFAULT_VARIANT) -> Variant:
    """A Tier 3 prompt variant by name. Raises VariantError for an unknown or bad one."""
    return load_variant("tier3", name, QUESTION_SCHEMA)


QUESTIONS = load_prompt().questions
