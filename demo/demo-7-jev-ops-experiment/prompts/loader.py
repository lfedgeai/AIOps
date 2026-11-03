"""Prompt variants: named, versioned question wording kept as YAML (design D1-D3).

A variant supplies text only. The calling tier supplies the question IDs, types and
option keys (a schema built from its own vocabularies), so a variant can never add,
drop or rename a question or an option, and the parsers and expected outcomes never
change. No engine, HTTP or tier-specific code lives here.

    schema = {"severity": QuestionSpec("choice", ("normal", "degraded", ...)), ...}
    v = load_variant("tier1", "baseline", schema)
    v.questions   # the dict sent to the engine
    v.identity    # {"name", "version", "sha256"} for the run record
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).resolve().parent
TIERS = ("tier1", "tier2", "tier3")
DEFAULT_VARIANT = "baseline"
TYPES = ("choice", "noul")
NOUL_OPTIONS = ("true", "false")
TOP_KEYS = {"name", "version", "tier", "style", "rationale", "dev_records", "questions"}
# "wording": only makes the existing questions clearer. "policy": also states, in general
# operational terms, the rule behind the expected answers. Reports show the two apart.
STYLES = ("wording", "policy")
QUESTION_KEYS = {"instructions", "criteria", "criteria_templates", "order"}
# Option order reaches the engine (Jev's docs say jev-1.13 "leans toward the option that
# comes first"). A variant may ask for the tier's order reversed, to measure that bias with
# the same text and keys. Nothing else: no free ordering, so the order stays checkable.
ORDERS = ("schema", "reversed")


class VariantError(ValueError):
    """A variant file that is missing, malformed or does not fit the tier's schema."""


@dataclass(frozen=True)
class QuestionSpec:
    """What the tier fixes for one question: its type, its option keys in order, and
    optional named groups of options that a variant may describe with one template."""
    type: str
    options: tuple[str, ...]
    groups: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self):
        if self.type not in TYPES:
            raise ValueError(f"unknown question type {self.type!r}")
        if self.type == "noul" and self.options != NOUL_OPTIONS:
            raise ValueError("a noul question's options are exactly ('true', 'false')")
        for g, keys in self.groups.items():
            if not set(keys) <= set(self.options):
                raise ValueError(f"group {g!r} has keys outside the options")


@dataclass(frozen=True)
class Variant:
    name: str
    version: int
    tier: str
    rationale: str
    dev_records: tuple[str, ...]
    questions: dict
    sha256: str
    path: str
    style: str = "wording"

    @property
    def identity(self) -> dict:
        return {"name": self.name, "version": self.version, "sha256": self.sha256, "style": self.style}


def questions_hash(questions: dict) -> str:
    """SHA-256 of the questions as sent: key order kept (option order reaches the
    engines), no whitespace, UTF-8."""
    blob = json.dumps(questions, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def variant_path(tier: str, name: str) -> Path:
    return PROMPTS_DIR / tier / f"{name}.yaml"


def list_variants(tier: str) -> list[str]:
    _check_tier(tier)
    return sorted(p.stem for p in (PROMPTS_DIR / tier).glob("*.yaml"))


def _check_tier(tier: str) -> None:
    if tier not in TIERS:
        raise VariantError(f"unknown tier {tier!r} (one of {', '.join(TIERS)})")


def _text(where: str, v) -> str:
    if not isinstance(v, str) or not v.strip():
        raise VariantError(f"{where}: must be a non-empty string")
    return v


def _build_question(where: str, spec: QuestionSpec, raw) -> dict:
    if not isinstance(raw, dict):
        raise VariantError(f"{where}: must be a mapping")
    extra = set(raw) - QUESTION_KEYS
    if extra:
        raise VariantError(f"{where}: unknown keys {sorted(extra)} (a variant sets text only; "
                           "type and options come from the tier)")
    instructions = _text(f"{where}.instructions", raw.get("instructions"))

    explicit = raw.get("criteria") or {}
    templates = raw.get("criteria_templates") or {}
    if not isinstance(explicit, dict) or not isinstance(templates, dict):
        raise VariantError(f"{where}: criteria and criteria_templates must be mappings")
    if any(not isinstance(k, str) for k in explicit):
        # YAML reads bare true/false/yes/no/on/off (and numbers) as non-strings.
        raise VariantError(f"{where}.criteria: quote every option key (got {sorted(map(repr, explicit))})")

    unknown = set(explicit) - set(spec.options)
    if unknown:
        raise VariantError(f"{where}.criteria: options not in this question: {sorted(unknown)}")
    unknown_groups = set(templates) - set(spec.groups)
    if unknown_groups:
        raise VariantError(f"{where}.criteria_templates: unknown groups {sorted(unknown_groups)} "
                           f"(available: {sorted(spec.groups) or 'none'})")

    from_template: dict[str, str] = {}
    for g, tmpl in templates.items():
        tmpl = _text(f"{where}.criteria_templates.{g}", tmpl)
        if "{option}" not in tmpl:
            raise VariantError(f"{where}.criteria_templates.{g}: must contain {{option}}")
        for k in spec.groups[g]:
            if k in from_template:
                raise VariantError(f"{where}: option {k!r} is in two templated groups")
            from_template[k] = tmpl.replace("{option}", k)

    both = set(explicit) & set(from_template)
    if both:
        raise VariantError(f"{where}: options described twice (criteria and a template): {sorted(both)}")
    missing = [k for k in spec.options if k not in explicit and k not in from_template]
    if missing:
        raise VariantError(f"{where}: no description for options {missing}")

    order = raw.get("order", "schema")
    if order not in ORDERS:
        raise VariantError(f"{where}.order: must be one of {', '.join(ORDERS)}")
    if order != "schema" and spec.type != "choice":
        raise VariantError(f"{where}.order: only a choice question's options can be reordered")
    keys = spec.options if order == "schema" else tuple(reversed(spec.options))
    criteria = {k: _text(f"{where}.criteria.{k}", explicit[k]) if k in explicit else from_template[k]
                for k in keys}
    return {"type": spec.type, "instructions": instructions, "criteria": criteria}


def build_questions(where: str, schema: dict[str, QuestionSpec], raw) -> dict:
    if not isinstance(raw, dict):
        raise VariantError(f"{where}: 'questions' must be a mapping")
    missing = [q for q in schema if q not in raw]
    extra = sorted(set(raw) - set(schema))
    if missing or extra:
        raise VariantError(f"{where}: questions must be exactly {list(schema)} "
                           f"(missing {missing}, extra {extra})")
    return {qid: _build_question(f"{where}.{qid}", spec, raw[qid]) for qid, spec in schema.items()}


def load_variant(tier: str, name: str, schema: dict[str, QuestionSpec]) -> Variant:
    _check_tier(tier)
    path = variant_path(tier, name)
    if not path.is_file():
        raise VariantError(f"no {tier} prompt variant {name!r} "
                           f"(available: {', '.join(list_variants(tier)) or 'none'})")
    where = f"{tier}/{path.name}"
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise VariantError(f"{where}: not valid YAML ({e})") from e
    if not isinstance(doc, dict):
        raise VariantError(f"{where}: must be a mapping")
    extra = set(doc) - TOP_KEYS
    missing = {"name", "version", "tier", "rationale", "questions"} - set(doc)
    if extra or missing:
        raise VariantError(f"{where}: unknown keys {sorted(extra)}, missing keys {sorted(missing)}")
    if doc["name"] != name:
        raise VariantError(f"{where}: name {doc['name']!r} does not match the file name")
    if doc["tier"] != tier:
        raise VariantError(f"{where}: tier {doc['tier']!r} does not match the directory")
    version = doc["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise VariantError(f"{where}: version must be a positive integer")
    dev = doc.get("dev_records") or []
    if not isinstance(dev, list) or not all(isinstance(d, str) for d in dev):
        raise VariantError(f"{where}: dev_records must be a list of paths")
    style = doc.get("style", "wording")
    if style not in STYLES:
        raise VariantError(f"{where}: style must be one of {', '.join(STYLES)}")
    questions = build_questions(where, schema, doc["questions"])
    return Variant(name=name, version=version, tier=tier,
                   rationale=_text(f"{where}.rationale", doc["rationale"]),
                   dev_records=tuple(dev), questions=questions,
                   sha256=questions_hash(questions), path=str(path), style=style)


def engine_prompt(engine) -> dict | None:
    """The identity of the variant an engine asks, for its run record. None for an
    engine without one (test fakes); readers then treat the record as baseline."""
    v = getattr(engine, "prompt", None)
    return v.identity if isinstance(v, Variant) else None


ASSUMED_NOTE = "no prompt field: read as baseline"


def record_prompt(record: dict) -> tuple[dict, bool]:
    """(prompt identity, assumed) for a run record. Records written before variants
    existed have no `prompt` field and are read as baseline version 1, sha256 unknown."""
    p = record.get("prompt")
    if isinstance(p, dict) and p.get("name"):
        return {"name": p["name"], "version": p.get("version"), "sha256": p.get("sha256"),
                "style": p.get("style", "wording")}, False
    return {"name": DEFAULT_VARIANT, "version": 1, "sha256": None, "style": "wording"}, True


def all_variant_text(variant: Variant) -> str:
    """Every string a variant sends to an engine, for the leak checks."""
    parts = []
    for q in variant.questions.values():
        parts.append(q["instructions"])
        parts.extend(q["criteria"].values())
    return "\n".join(parts)
