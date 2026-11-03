"""--prompt reaches every engine and every record; variant text leaks nothing
(jev-prompt-variants tasks 3.1-3.5). Offline: fake transports, no engine is called."""
import json
import shutil
from pathlib import Path

import pytest
import yaml

import prompts.loader as loader
from prompts.loader import all_variant_text, engine_prompt, list_variants, record_prompt

MARK = "ZZQ-test-wording"


@pytest.fixture
def test_variant(tmp_path, monkeypatch):
    """A copy of prompts/ with one extra variant per tier, `marked`, whose every
    instruction starts with MARK. The real files are not touched."""
    root = tmp_path / "prompts"
    shutil.copytree(loader.PROMPTS_DIR, root, ignore=shutil.ignore_patterns("*.py", "__pycache__"))
    for tier in loader.TIERS:
        doc = yaml.safe_load((root / tier / "baseline.yaml").read_text())
        doc.update(name="marked", version=3, rationale="test only")
        for q in doc["questions"].values():
            q["instructions"] = f"{MARK} {q['instructions']}"
        (root / tier / "marked.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    monkeypatch.setattr(loader, "PROMPTS_DIR", root)
    return root


def _marked(tier_module):
    v = tier_module.load_prompt("marked")
    assert v.version == 3 and all(q["instructions"].startswith(MARK) for q in v.questions.values())
    return v


# --- the variant reaches every engine (D4) -----------------------------------

def test_tier1_jev_request_uses_the_variant(test_variant):
    from jev import client as t1
    v = _marked(t1)
    c = t1.JevClient(t1.JevConfig(api_key="k"), post=lambda *a: None, prompt=v)
    assert c.build_request({"cpu_pct": 1})["questions"] == v.questions
    assert t1.JevClient(t1.JevConfig(api_key="k"), post=lambda *a: None).build_request({})["questions"] \
        == t1.QUESTIONS
    assert engine_prompt(c) == v.identity


def test_tier1_engine_factory_passes_the_variant(test_variant, monkeypatch):
    from jev import client as t1
    from llm import engines
    v = _marked(t1)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("JEV_API_KEY", "k")
    assert engines.build_engine("jev", prompt=v).prompt is v
    assert engines.build_engine("ollama:qwen3:8b", prompt=v).prompt is v
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert engines.build_engine("claude-opus", prompt=v).prompt is v


def test_tier1_baseline_engines_ask_the_variant(test_variant):
    from jev import client as t1
    from llm.claude import ClaudeAPIEngine, ClaudeCLIEngine
    from llm.ollama import OllamaEngine
    from tests.test_llm import GOOD, FakeMessage, FakeRunner, FakeSDK, cli_body
    v = _marked(t1)

    sdk = FakeSDK([FakeMessage(json.dumps(GOOD))])
    d = ClaudeAPIEngine("claude-sonnet", client=sdk, sleep=lambda _: None, prompt=v).decide({"x": 1})
    assert d.status == "ok" and MARK in sdk.calls[0]["messages"][0]["content"]

    runner = FakeRunner(json.dumps(cli_body()))
    ClaudeCLIEngine("claude-opus", runner=runner, prompt=v).decide({"x": 1})
    assert MARK in runner.calls[0]["input"]

    req = OllamaEngine("qwen3:8b", post=lambda *a: None, prompt=v).build_request({"x": 1})
    assert MARK in req["messages"][1]["content"]


def test_tier2_adapter_asks_the_variant(test_variant):
    from jev.client import JevClient, JevConfig
    from tier2 import jev_adapter as t2
    v = _marked(t2)
    a = t2.JevAdapter(JevClient(JevConfig(api_key="k"), post=lambda *x: None), prompt=v)
    assert a.build_request({"s": 1})["questions"] == v.questions
    assert engine_prompt(a) == v.identity
    base = t2.JevAdapter(JevClient(JevConfig(api_key="k"), post=lambda *x: None))
    assert base.build_request({})["questions"] == t2.QUESTIONS


def test_tier3_engines_ask_the_variant(test_variant):
    from jev.client import JevClient, JevConfig
    from tests.tier3.test_claude_engine import GOOD, cli_body
    from tests.test_llm import FakeMessage, FakeRunner, FakeSDK
    from tier3 import questions as t3
    from tier3.claude_engine import ClaudeAPIEngine, ClaudeCLIEngine
    from tier3.jev_adapter import JevAdapter
    v = _marked(t3)

    a = JevAdapter(JevClient(JevConfig(api_key="k"), post=lambda *x: None), prompt=v)
    assert a.build_request({"s": 1})["questions"] == v.questions

    sdk = FakeSDK([FakeMessage(json.dumps(GOOD))])
    eng = ClaudeAPIEngine("claude-opus", client=sdk, sleep=lambda _: None, prompt=v)
    assert eng.decide({"s": 1}).status == "ok" and MARK in sdk.calls[0]["messages"][0]["content"]

    runner = FakeRunner(json.dumps(cli_body()))
    ClaudeCLIEngine("claude-opus", runner=runner, prompt=v).decide({"s": 1})
    assert MARK in runner.calls[0]["input"]
    assert engine_prompt(eng) == v.identity


def test_tier3_cli_builds_engines_with_the_variant(test_variant, monkeypatch):
    from tier3 import cli, questions as t3
    v = _marked(t3)
    monkeypatch.setenv("JEV_API_KEY", "k")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    assert cli.build_engine("jev", v).prompt is v
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert cli.build_engine("claude-opus", v).prompt is v


# --- the variant is recorded (D3) ---------------------------------------------

class Prompted:
    """Wraps a fake engine and carries a variant, as the real engines do."""

    def __init__(self, inner, prompt):
        self.inner, self.prompt = inner, prompt
        self.info = getattr(inner, "info", {"name": "jev", "transport": "fake"})

    def decide(self, *a, **k):
        return self.inner.decide(*a, **k)


def test_tier1_records_and_summary_carry_the_prompt(test_variant, tmp_path):
    import run
    from jev import client as t1
    from tests.helpers import FakeClient, ok_decision
    v = _marked(t1)
    records, _ = run.run_experiment(Prompted(FakeClient([ok_decision()]), v), "healthy", 2, 7,
                                    tmp_path, out=lambda *_: None)
    assert all(r["prompt"] == v.identity for r in records)
    (summary,) = tmp_path.glob("*-summary.json")
    assert json.loads(summary.read_text())["meta"]["prompt"] == v.identity
    assert summary.name.endswith("-prompt-marked-summary.json")


def test_tier1_baseline_keeps_its_file_names(tmp_path):
    import run
    from jev import client as t1
    from tests.helpers import FakeClient, ok_decision
    run.run_experiment(Prompted(FakeClient([ok_decision()]), t1.load_prompt()), "healthy", 1, 7,
                       tmp_path, out=lambda *_: None)
    assert [p.name for p in tmp_path.glob("*.jsonl")][0].endswith("-1runs.jsonl")


def test_tier2_record_and_summary_carry_the_prompt(test_variant, tmp_path):
    from tests.tier2.helpers import FakeEngine
    from tests.tier2.test_runner import CFG, make
    from tier2 import jev_adapter as t2
    from tier2.runner import run_batch
    from tier2.scenarios import get_scenario
    v = _marked(t2)
    deps, *_ = make(Prompted(FakeEngine(), v))
    summary = run_batch([get_scenario("F000")], deps, CFG, tmp_path, run_id="r")
    assert summary["prompt"] == v.identity
    rec = json.loads(next((tmp_path / "r").glob("F000*.json")).read_text())
    assert rec["prompt"] == v.identity


def test_tier3_record_and_summary_carry_the_prompt(test_variant, tmp_path):
    from tests.tier3.helpers import QUIET, SeqEngine, ok_decision
    from tests.tier3.test_runner import CFG, make
    from tier3 import questions as t3
    from tier3.runner import run_batch
    from tier3.scenarios import get_scenario
    v = _marked(t3)
    deps, *_ = make(tmp_path, Prompted(SeqEngine(ok_decision(**QUIET)), v))
    summary = run_batch([get_scenario("K000")], deps, CFG, tmp_path, run_id="r")
    assert summary["prompt"] == v.identity
    rec = json.loads(next((tmp_path / "r").glob("K000*.json")).read_text())
    assert rec["prompt"] == v.identity


def test_record_without_prompt_reads_as_baseline():
    assert record_prompt({"run_id": "old"}) == ({"name": "baseline", "version": 1, "sha256": None,
                                                 "style": "wording"}, True)
    p = {"name": "marked", "version": 3, "sha256": "ab", "style": "policy"}
    assert record_prompt({"prompt": p}) == (p, False)
    assert record_prompt({"prompt": {"name": "x", "version": 1, "sha256": "c"}})[0]["style"] == "wording"
    assert record_prompt({"prompt": None})[1] is True


# --- unknown variant stops before any run (3.1) ---------------------------------

def _forbid(monkeypatch, target):
    def boom(*a, **k):
        raise AssertionError("built dependencies for an unknown variant")
    monkeypatch.setattr(target, boom)


@pytest.mark.parametrize("argv", [["run", "K001", "--prompt", "nope"],
                                  ["batch", "K001", "--prompt", "nope"]])
def test_tier3_unknown_variant_exits_before_any_run(monkeypatch, capsys, argv):
    from tier3 import cli
    _forbid(monkeypatch, "tier3.cli.build_deps")
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2 and "available: baseline" in capsys.readouterr().err


def test_tier4_unknown_variant_exits_before_any_run(monkeypatch, capsys):
    from tier4 import cli
    _forbid(monkeypatch, "tier4.cli.build_tier4_deps")
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "K001", "--prompt", "nope"])
    assert e.value.code == 2 and "no tier3 prompt variant 'nope'" in capsys.readouterr().err


def test_tier2_unknown_variant_exits_before_any_run(monkeypatch, capsys):
    from tier2 import cli
    _forbid(monkeypatch, "tier2.cli.build_deps")
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "F001", "--prompt", "nope"])
    assert e.value.code == 2 and "no tier2 prompt variant 'nope'" in capsys.readouterr().err


def test_tier1_unknown_variant_exits_before_any_run(monkeypatch, capsys):
    import run
    _forbid(monkeypatch, "run.build_engine")
    with pytest.raises(SystemExit) as e:
        run.main(["--prompt", "nope"])
    assert e.value.code == 2 and "no tier1 prompt variant 'nope'" in capsys.readouterr().err


def test_tier1_known_variant_is_loaded_and_passed(test_variant, monkeypatch, tmp_path):
    import run
    seen = {}

    def fake_build(name, prompt=None):
        seen["prompt"] = prompt
        raise run.ConfigError("stop here")
    monkeypatch.setattr(run, "build_engine", fake_build)
    assert run.main(["--prompt", "marked", "--output-dir", str(tmp_path)]) == 2
    assert seen["prompt"].name == "marked" and seen["prompt"].version == 3


# --- variant text passes the leak rules (D9, 3.4) -------------------------------

def _forbidden_terms():
    from scenarios.definitions import ACTIONS, CAUSES, HUMAN_REVIEW, SEVERITIES, scenario_names
    from tier2.scenarios import load_all as tier2_all
    from tier3.scenarios import load_all as tier3_all
    from tier3.vocab import FAULT_TYPES

    option_words = {*SEVERITIES, *ACTIONS, *CAUSES, *HUMAN_REVIEW}
    terms = {"jev-fault", "flagd", "feature_flag", "ground_truth", "expected", "scenario",
             "experiment", "acceptable_actions", "fault_type", *FAULT_TYPES}
    terms |= {n for n in scenario_names() if n not in option_words}
    for sc in tier2_all().values():
        terms.add(sc.experiment_id)
        if sc.fault:
            terms.add(sc.fault.flag)
    terms |= set(tier3_all())
    return {t.lower() for t in terms}


def _all_variants():
    from jev import client as t1
    from tier2 import jev_adapter as t2
    from tier3 import questions as t3
    mods = {"tier1": t1, "tier2": t2, "tier3": t3}
    return [(tier, name, mods[tier]) for tier in loader.TIERS for name in list_variants(tier)]


@pytest.mark.parametrize("tier,name,mod", _all_variants(), ids=lambda x: x if isinstance(x, str) else "")
def test_variant_text_names_no_scenario_fault_or_harness_term(tier, name, mod):
    text = all_variant_text(mod.load_prompt(name)).lower()
    hits = sorted(t for t in _forbidden_terms() if t in text)
    assert not hits, f"{tier}/{name}.yaml contains {hits}"


def test_leak_check_catches_a_fault_name(test_variant):
    from tier3 import questions as t3
    path = Path(test_variant) / "tier3" / "marked.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["questions"]["diagnosis"]["instructions"] = "Is this scale_zero?"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    text = all_variant_text(t3.load_prompt("marked")).lower()
    assert "scale_zero" in {t for t in _forbidden_terms() if t in text}


@pytest.mark.parametrize("name", list_variants("tier1"))
def test_tier1_prompt_fixed_part_is_the_same_for_every_scenario_per_variant(name):
    """test_prompt_does_not_leak_scenario_or_expected, once per Tier 1 variant."""
    from generator.server_state import generate
    from jev.client import load_prompt
    from llm.prompt import SYSTEM_PROMPT, build_prompt, build_schema
    from scenarios.definitions import get_scenario, scenario_names
    q = load_prompt(name).questions
    fixed = set()
    for s in scenario_names():
        obs = generate(get_scenario(s), 1234).to_dict()
        state = json.dumps({"server_observations": obs}, indent=2)
        p = build_prompt(obs, q)
        assert p.endswith(state)
        fixed.add(SYSTEM_PROMPT + p[: -len(state)] + json.dumps(build_schema(q)))
    assert len(fixed) == 1


@pytest.mark.parametrize("name", list_variants("tier3"))
@pytest.mark.parametrize("eid", ["K000", "K001", "K009"])
def test_tier3_whole_path_leaks_nothing_per_variant(tmp_path, name, eid):
    """The Tier 3 whole-path leak test, with each variant's questions in the request."""
    from jev.client import JevClient, JevConfig
    from tests.tier3.helpers import QUIET, SeqEngine, ok_decision
    from tests.tier3.test_runner import CFG, make
    from tier3.claude_engine import SYSTEM_PROMPT, build_prompt
    from tier3.jev_adapter import JevAdapter
    from tier3.questions import load_prompt
    from tier3.runner import run_experiment
    from tier3.scenarios import get_scenario
    from tier3.vocab import FAULT_TYPES
    scn, v = get_scenario(eid), load_prompt(name)
    deps, _, _, eng = make(tmp_path, SeqEngine(ok_decision(**QUIET)))
    run_experiment(scn, deps, CFG, "r", tmp_path)
    adapter = JevAdapter(JevClient(JevConfig(api_key="k"), post=lambda *a: None), prompt=v)
    for obs in eng.calls:
        for text in (json.dumps(adapter.build_request(obs)), SYSTEM_PROMPT + build_prompt(obs, v.questions)):
            for s in ("jev-fault", "feature_flag", "flagd", scn.experiment_id, "ground_truth",
                      "acceptable_actions", "expected", "fault_type", *FAULT_TYPES):
                assert s not in text, f"{s!r} leaked under {name} for {eid}"
