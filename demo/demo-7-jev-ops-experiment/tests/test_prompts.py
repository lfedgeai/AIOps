"""Prompt variant loader (jev-prompt-variants tasks 2.1-2.4). Offline, no engine."""
import json
from pathlib import Path

import pytest
import yaml

import prompts.loader as loader
from jev import client as tier1
from prompts.loader import QuestionSpec, VariantError, load_variant, questions_hash
from tier2 import jev_adapter as tier2
from tier3 import questions as tier3

FIXTURES = Path(__file__).parent / "fixtures"
TIERS = {"tier1": tier1, "tier2": tier2, "tier3": tier3}


# --- baseline reproduces what was sent before variants (D2) ---------------------

@pytest.mark.parametrize("tier,n", [("tier1", 1), ("tier2", 2), ("tier3", 3)])
def test_baseline_equals_pre_variant_questions_byte_for_byte(tier, n):
    frozen = (FIXTURES / f"questions-tier{n}-pre-variants.json").read_text(encoding="utf-8")
    built = TIERS[tier].load_prompt("baseline").questions
    assert json.dumps(built, indent=2, ensure_ascii=False) + "\n" == frozen
    assert TIERS[tier].QUESTIONS == built


@pytest.mark.parametrize("tier", list(TIERS))
def test_baseline_identity(tier):
    v = TIERS[tier].load_prompt("baseline")
    assert v.identity == {"name": "baseline", "version": 1, "sha256": questions_hash(v.questions),
                          "style": "wording"}
    assert len(v.sha256) == 64


@pytest.mark.parametrize("tier", list(TIERS))
def test_every_variant_file_loads(tier):
    names = loader.list_variants(tier)
    assert "baseline" in names
    for name in names:
        v = TIERS[tier].load_prompt(name)
        assert list(v.questions) == list(TIERS[tier].QUESTION_SCHEMA)


def test_hash_follows_text_and_option_order():
    q = {"a": {"type": "choice", "instructions": "x", "criteria": {"p": "1", "q": "2"}}}
    same = json.loads(json.dumps(q))
    reordered = {"a": {"type": "choice", "instructions": "x", "criteria": {"q": "2", "p": "1"}}}
    reworded = {"a": {"type": "choice", "instructions": "y", "criteria": {"p": "1", "q": "2"}}}
    assert questions_hash(q) == questions_hash(same)
    assert questions_hash(q) != questions_hash(reordered)
    assert questions_hash(q) != questions_hash(reworded)


# --- rejections (D1) --------------------------------------------------------------

SCHEMA = {
    "level": QuestionSpec("choice", ("low", "high")),
    "who": QuestionSpec("choice", ("none", "a", "b", "n1"), {"svc": ("a", "b"), "nodes": ("n1",)}),
    "ok": QuestionSpec("noul", ("true", "false")),
}


def good_doc(**over):
    d = {"name": "v", "version": 1, "tier": "tier1", "rationale": "why", "dev_records": [],
         "questions": {
             "level": {"instructions": "How bad?", "criteria": {"low": "L", "high": "H"}},
             "who": {"instructions": "Where?", "criteria": {"none": "Nothing."},
                     "criteria_templates": {"svc": "In {option}.", "nodes": "In node {option}."}},
             "ok": {"instructions": "Fine?", "criteria": {"true": "Yes.", "false": "No."}},
         }}
    d.update(over)
    return d


@pytest.fixture
def prompts_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "PROMPTS_DIR", tmp_path)
    (tmp_path / "tier1").mkdir()
    return tmp_path


def write(prompts_dir, doc, name="v", raw=None):
    (prompts_dir / "tier1" / f"{name}.yaml").write_text(raw if raw is not None else yaml.safe_dump(doc))


def test_good_variant_builds_in_schema_order(prompts_dir):
    write(prompts_dir, good_doc())
    v = load_variant("tier1", "v", SCHEMA)
    assert list(v.questions) == ["level", "who", "ok"]
    assert v.questions["who"] == {"type": "choice", "instructions": "Where?",
                                  "criteria": {"none": "Nothing.", "a": "In a.", "b": "In b.",
                                               "n1": "In node n1."}}
    assert v.questions["ok"]["type"] == "noul"


def test_order_reversed_keeps_text_and_reverses_options(prompts_dir):
    write(prompts_dir, good_doc())
    base = load_variant("tier1", "v", SCHEMA)
    d = good_doc()
    d["questions"]["who"]["order"] = "reversed"
    d["name"] = "r"
    write(prompts_dir, d, name="r")
    rev = load_variant("tier1", "r", SCHEMA)
    assert list(rev.questions["who"]["criteria"]) == list(reversed(list(base.questions["who"]["criteria"])))
    assert rev.questions["who"]["criteria"] == base.questions["who"]["criteria"]  # same text per key
    assert rev.questions["level"] == base.questions["level"]
    assert rev.sha256 != base.sha256


def mutate(fn):
    d = good_doc()
    fn(d)
    return d


@pytest.mark.parametrize("doc,needle", [
    (mutate(lambda d: d["questions"].pop("ok")), "missing ['ok']"),
    (mutate(lambda d: d["questions"].update(extra={"instructions": "x", "criteria": {}})), "extra ['extra']"),
    (mutate(lambda d: d["questions"]["level"]["criteria"].pop("high")), "no description for options ['high']"),
    (mutate(lambda d: d["questions"]["level"]["criteria"].update(drain="D")), "not in this question: ['drain']"),
    (mutate(lambda d: d["questions"]["level"].update(type="score")), "unknown keys ['type']"),
    (mutate(lambda d: d["questions"]["who"]["criteria_templates"].update(pods="{option}")), "unknown groups ['pods']"),
    (mutate(lambda d: d["questions"]["who"]["criteria_templates"].update(svc="no placeholder")), "must contain {option}"),
    (mutate(lambda d: d["questions"]["who"]["criteria"].update(a="A")), "described twice"),
    (mutate(lambda d: d["questions"]["level"].update(instructions="  ")), "non-empty string"),
    (mutate(lambda d: d["questions"]["level"]["criteria"].update(low=3)), "non-empty string"),
    (good_doc(name="other"), "does not match the file name"),
    (good_doc(tier="tier2"), "does not match the directory"),
    (good_doc(version=0), "positive integer"),
    (good_doc(version="1"), "positive integer"),
    (good_doc(version=True), "positive integer"),
    (good_doc(dev_records="results/x"), "dev_records"),
    (good_doc(prompt="free text"), "unknown keys ['prompt']"),
    (good_doc(style="clever"), "style must be one of wording, policy"),
    (mutate(lambda d: d.pop("rationale")), "missing keys ['rationale']"),
    (mutate(lambda d: d["questions"]["who"].update(order="random")), "order: must be one of schema, reversed"),
    (mutate(lambda d: d["questions"]["ok"].update(order="reversed")), "only a choice question's options"),
])
def test_rejections(prompts_dir, doc, needle):
    write(prompts_dir, doc)
    with pytest.raises(VariantError) as e:
        load_variant("tier1", "v", SCHEMA)
    assert needle in str(e.value)


def test_style_defaults_to_wording_and_is_in_the_identity(prompts_dir):
    write(prompts_dir, good_doc())
    assert load_variant("tier1", "v", SCHEMA).identity["style"] == "wording"
    write(prompts_dir, good_doc(style="policy"))
    v = load_variant("tier1", "v", SCHEMA)
    assert v.style == "policy" and v.identity["style"] == "policy"


def test_unquoted_yaml_boolean_keys_are_rejected(prompts_dir):
    raw = yaml.safe_dump(good_doc()).replace("'true':", "true:").replace("'false':", "false:")
    write(prompts_dir, None, raw=raw)
    with pytest.raises(VariantError, match="quote every option key"):
        load_variant("tier1", "v", SCHEMA)


def test_unknown_variant_lists_the_available_ones(prompts_dir):
    write(prompts_dir, good_doc())
    with pytest.raises(VariantError, match=r"no tier1 prompt variant 'nope' \(available: v\)"):
        load_variant("tier1", "nope", SCHEMA)


def test_unknown_tier():
    with pytest.raises(VariantError, match="unknown tier"):
        load_variant("tier9", "baseline", SCHEMA)


def test_bad_yaml(prompts_dir):
    write(prompts_dir, None, raw="name: [unclosed\n")
    with pytest.raises(VariantError, match="not valid YAML"):
        load_variant("tier1", "v", SCHEMA)


def test_schema_guards():
    with pytest.raises(ValueError):
        QuestionSpec("score", ("a",))
    with pytest.raises(ValueError):
        QuestionSpec("noul", ("yes", "no"))
    with pytest.raises(ValueError):
        QuestionSpec("choice", ("a",), {"g": ("b",)})
