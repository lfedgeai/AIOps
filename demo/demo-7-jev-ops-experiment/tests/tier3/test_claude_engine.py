"""Offline tests for the Tier 3 Claude baseline engine. No Anthropic API or claude CLI call."""
import ast
import io
import json
from pathlib import Path

import anthropic
import pytest

from tests.test_llm import FakeMessage, FakeRunner, FakeSDK, _status_error
from tests.tier3.helpers import QUIET, SeqEngine, ok_decision
from tests.tier3.test_runner import CFG, make
from tier3 import claude_engine
from tier3.claude_engine import (SYSTEM_PROMPT, ClaudeAPIEngine, ClaudeCLIEngine, ConfigError,
                                 build_claude_engine, build_prompt, build_schema, parse_answer)
from tier3.questions import QUESTIONS
from tier3.runner import run_experiment
from tier3.scenarios import get_scenario, load_all
from tier3.vocab import ACTIONS, COMPONENTS, DIAGNOSES, FAULT_TYPES, NO_COMPONENT

GOOD = {"incident_detected": "true", "affected_component": "cart",
        "diagnosis": "service_unavailable", "recommended_action": "scale_up"}


def api_engine(replies, **kw):
    sdk = FakeSDK(replies)
    return ClaudeAPIEngine("claude-opus", client=sdk, sleep=lambda _: None, **kw), sdk


# --- questions, schema, parser ------------------------------------------------

def test_schema_enums_are_the_jev_answer_sets():
    p = build_schema()["properties"]
    assert set(p) == set(QUESTIONS)
    assert p["incident_detected"]["enum"] == ["true", "false"]
    assert p["affected_component"]["enum"] == [NO_COMPONENT, *COMPONENTS]
    assert p["diagnosis"]["enum"] == list(DIAGNOSES)
    assert p["recommended_action"]["enum"] == list(ACTIONS)


def test_prompt_uses_the_jev_question_wording():
    text = build_prompt({"x": 1})
    for q in QUESTIONS.values():
        assert q["instructions"] in text
        for desc in q["criteria"].values():
            assert desc in text
    assert '"observations"' in text


def test_valid_answer_parses():
    d = parse_answer(GOOD)
    assert d.status == "ok" and d.incident_detected is True and d.affected_component == "cart"
    assert parse_answer({**GOOD, "incident_detected": "false"}).incident_detected is False


@pytest.mark.parametrize("bad", [
    {**GOOD, "affected_component": "database"},
    {**GOOD, "diagnosis": "Service_Unavailable"},
    {**GOOD, "recommended_action": "reboot"},
    {**GOOD, "incident_detected": True},
    {k: v for k, v in GOOD.items() if k != "diagnosis"},
    "not json", ["a list"],
])
def test_out_of_set_or_missing_is_invalid_not_coerced(bad):
    d = parse_answer(bad)
    assert d.status == "invalid" and d.recommended_action is None and d.message


# --- API engine (fake SDK) ------------------------------------------------------

def test_api_request_shape_and_telemetry():
    eng, sdk = api_engine([FakeMessage(json.dumps(GOOD), model="claude-opus-5-5")])
    d = eng.decide({"x": 1})
    assert d.status == "ok" and d.recommended_action == "scale_up"
    call = sdk.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["system"] == SYSTEM_PROMPT
    assert call["output_config"]["format"]["schema"] == build_schema()
    assert "temperature" not in call and "fallbacks" not in call
    t = d.telemetry
    assert t["input_tokens"] == 900 and t["output_tokens"] == 120 and t["model"] == "claude-opus-5-5"
    assert t["cost_usd"] is None and t["engine"]["transport"] == "anthropic-api"


def test_api_cost_only_from_user_prices():
    eng, _ = api_engine([FakeMessage(json.dumps(GOOD))], prices=(4.0, 20.0))
    t = eng.decide({}).telemetry
    assert t["cost_source"] == "estimated" and t["cost_usd"] == pytest.approx((900 * 4 + 120 * 20) / 1e6)


def test_api_refusal_and_out_of_set_are_invalid():
    eng, sdk = api_engine([FakeMessage(stop_reason="refusal", category="cyber"),
                           FakeMessage(json.dumps({**GOOD, "diagnosis": "gremlins"}))])
    assert eng.decide({}).status == "invalid" and len(sdk.calls) == 1
    assert eng.decide({}).status == "invalid"


def test_api_retry_then_ok_and_auth_error_not_retried():
    eng, sdk = api_engine([_status_error(anthropic.RateLimitError, 429), FakeMessage(json.dumps(GOOD))])
    d = eng.decide({})
    assert d.status == "ok" and d.telemetry["attempts"] == 2
    eng, sdk = api_engine([_status_error(anthropic.AuthenticationError, 401)])
    d = eng.decide({})
    assert d.status == "error" and d.telemetry["http_status"] == 401 and len(sdk.calls) == 1


# --- CLI fallback (fake runner) -------------------------------------------------

def cli_body(answer=GOOD):
    return {"type": "result", "is_error": False, "result": json.dumps(answer),
            "structured_output": answer, "duration_api_ms": 3000, "total_cost_usd": 0.05,
            "usage": {"input_tokens": 5, "cache_creation_input_tokens": 1000,
                      "cache_read_input_tokens": 200, "output_tokens": 80},
            "modelUsage": {"claude-opus-5-5": {}}}


def test_cli_isolation_and_parse(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    runner = FakeRunner(json.dumps(cli_body()))
    d = ClaudeCLIEngine("claude-opus", runner=runner).decide({"x": 1})
    assert d.status == "ok" and d.incident_detected is True
    c = runner.calls[0]
    assert c["cwd_files"] == [] and "CLAUDECODE" not in c["env"]
    for flag in ("--tools", "--strict-mcp-config", "--setting-sources", "--no-session-persistence"):
        assert flag in c["cmd"]
    assert c["cmd"][c["cmd"].index("--model") + 1] == "claude-opus-5-5"
    assert c["input"] == build_prompt({"x": 1})
    t = d.telemetry
    assert t["input_tokens"] == 1205 and t["cost_source"] == "reported:claude-cli"
    assert t["engine"]["transport"] == "claude-cli"


def test_cli_out_of_set_and_failures():
    assert ClaudeCLIEngine(runner=FakeRunner(json.dumps(cli_body({**GOOD, "affected_component": "db"})))
                           ).decide({}).status == "invalid"
    assert ClaudeCLIEngine(runner=FakeRunner("", returncode=1, stderr="boom")).decide({}).status == "error"
    assert ClaudeCLIEngine(runner=FakeRunner("not json")).decide({}).status == "error"


def test_factory_api_with_key_else_cli_loudly(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert isinstance(build_claude_engine(), ClaudeAPIEngine)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    err = io.StringIO()
    monkeypatch.setattr(claude_engine.subprocess, "run", lambda *a, **k: type("P", (), {"stdout": "x"})())
    eng = build_claude_engine(which=lambda _: "/bin/claude", err=err)
    assert isinstance(eng, ClaudeCLIEngine) and "claude CLI" in err.getvalue()
    with pytest.raises(ConfigError):
        build_claude_engine(which=lambda _: None, err=io.StringIO())


# --- whole path: what this engine would send leaks nothing ---------------------

@pytest.mark.parametrize("eid", sorted(load_all()))
def test_prompt_leaks_nothing_on_any_poll(tmp_path, eid):
    scn = get_scenario(eid)
    deps, _, _, eng = make(tmp_path, SeqEngine(ok_decision(**QUIET)))
    run_experiment(scn, deps, CFG, "r", tmp_path)
    assert len(eng.calls) >= 2
    for obs in eng.calls:
        text = SYSTEM_PROMPT + build_prompt(obs)
        for s in ("jev-fault", "feature_flag", "flagd", scn.experiment_id, "ground_truth",
                  "acceptable_actions", "expected", "fault_type", *FAULT_TYPES):
            assert s not in text, f"{s!r} leaked into the Claude prompt for {eid}"


def test_engine_module_does_not_import_jev_or_ground_truth():
    tree = ast.parse(Path(claude_engine.__file__).read_text())
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.startswith("jev") for m in mods)
    assert not {"tier3.scenarios", "tier3.faults"} & mods


def test_engine_recorded_on_the_run_and_in_the_lfedge_approach(tmp_path):
    from tier3.report import export_lfedge, format_scorecard

    class Named(SeqEngine):
        info = {"name": "claude-opus", "transport": "claude-cli"}

    deps, _, _, _ = make(tmp_path, Named(ok_decision(telemetry={"model": "claude-opus-5-5"})))
    rec = run_experiment(get_scenario("K001"), deps, CFG, "r", tmp_path)
    assert rec["engine"]["name"] == "claude-opus"
    assert export_lfedge([rec])[0]["approach"] == "claude-opus:claude-opus-5-5"
    assert "Engine: claude-opus (claude-cli)." in format_scorecard([rec])
    old = {**rec, "engine": None}  # records from before engines were recorded are Jev's
    assert export_lfedge([old])[0]["approach"] == "jev:claude-opus-5-5"


def test_cli_engine_choice():
    from tier3.cli import ENGINES
    assert ENGINES == ("jev", "claude-opus")
