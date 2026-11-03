"""Prompt, JSON schema and strict parsing shared by every baseline engine.

Everything is built from the question dict of the selected prompt variant (the
engine's `prompt.questions`; jev.client.QUESTIONS, the baseline, by default), so an
LLM is asked exactly what Jev is asked, with the same wording and the same allowed
options. The state is
wrapped as {"server_observations": ...}, the same shape Jev receives.
"""
from __future__ import annotations

import json

from jev.client import QUESTIONS, STATUS_INVALID, STATUS_OK, Decision
from scenarios.definitions import HUMAN_REVIEW

SYSTEM_PROMPT = ("You assess server observations and answer each question with exactly "
                 "one of the allowed options.")

# noul (yes/no) criteria keys map to the human_review values Jev's answers become.
_NOUL = {"true": HUMAN_REVIEW[0], "false": HUMAN_REVIEW[1]}


def options(qid: str, questions: dict = QUESTIONS) -> dict[str, str]:
    """Allowed values for one question, each with its description."""
    q = questions[qid]
    if q["type"] == "noul":
        return {_NOUL[k]: v for k, v in q["criteria"].items()}
    return dict(q["criteria"])


def build_schema(questions: dict = QUESTIONS) -> dict:
    return {
        "type": "object",
        "properties": {qid: {"type": "string", "enum": list(options(qid, questions))} for qid in questions},
        "required": list(questions),
        "additionalProperties": False,
    }


def build_prompt(observations: dict, questions: dict = QUESTIONS) -> str:
    lines = ["Questions. Answer each one with exactly one allowed option.", ""]
    for qid, q in questions.items():
        lines.append(f"{qid}: {q['instructions']}")
        lines += [f"  - {opt}: {desc}" for opt, desc in options(qid, questions).items()]
        lines.append("")
    lines += ["Reply with a JSON object with the keys " + ", ".join(questions) + ".", "",
              "State:", json.dumps({"server_observations": observations}, indent=2)]
    return "\n".join(lines)


def parse_answer(data, raw=None, questions: dict = QUESTIONS) -> Decision:
    """Strict: all four fields, each in its allowed set, or 'invalid'. Never coerces."""
    raw = data if raw is None else raw
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            return Decision(status=STATUS_INVALID, raw_response=_raw(raw),
                            message="reply is not JSON")
    if not isinstance(data, dict):
        return Decision(status=STATUS_INVALID, raw_response=_raw(raw),
                        message="reply is not a JSON object")
    out = Decision(status=STATUS_OK, raw_response=_raw(raw))
    for qid in questions:
        v = data.get(qid)
        if v not in options(qid, questions):
            return Decision(status=STATUS_INVALID, raw_response=_raw(raw),
                            message=f"{qid}: missing or out-of-set value ({v!r})")
        setattr(out, qid, v)
    return out


def _raw(x):
    return x if isinstance(x, dict) else {"text": x}
