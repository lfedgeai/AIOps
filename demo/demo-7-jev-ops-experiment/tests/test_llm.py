"""Offline tests for the baseline LLM engines. No Anthropic, Ollama or claude CLI calls."""
import io
import json
import os
import subprocess
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
import requests

from generator.server_state import generate
from jev.client import QUESTIONS, ConfigError
from llm import engines
from llm.claude import ClaudeAPIEngine, ClaudeCLIEngine
from llm.ollama import OllamaEngine
from llm.prompt import SYSTEM_PROMPT, build_prompt, build_schema, options, parse_answer
from scenarios.definitions import ACTIONS, CAUSES, HUMAN_REVIEW, SEVERITIES, get_scenario, scenario_names

GOOD = {"severity": "high", "action": "investigate", "human_review": "no",
        "probable_cause": "resource_exhaustion"}


# --- prompt, schema, parser -------------------------------------------------

def test_schema_enums_are_the_tier1_option_sets():
    props = build_schema()["properties"]
    assert set(props) == set(QUESTIONS)
    assert props["severity"]["enum"] == list(SEVERITIES)
    assert props["action"]["enum"] == list(ACTIONS)
    assert props["probable_cause"]["enum"] == list(CAUSES)
    assert props["human_review"]["enum"] == list(HUMAN_REVIEW)
    assert build_schema()["additionalProperties"] is False


def test_prompt_uses_jev_question_wording():
    text = build_prompt({"cpu_pct": 1})
    for qid, q in QUESTIONS.items():
        assert q["instructions"] in text
        for desc in options(qid).values():
            assert desc in text
    assert '"server_observations"' in text


def test_prompt_does_not_leak_scenario_or_expected():
    # Everything except the observation is the same text for every scenario, and the
    # observation is exactly what the generator produced (its own leak test covers it).
    # Option names such as "critical" appear, as they do in Jev's request.
    option_words = {o for qid in QUESTIONS for o in options(qid)}
    fixed_parts = set()
    for name in scenario_names():
        obs = generate(get_scenario(name), 1234).to_dict()
        state = json.dumps({"server_observations": obs}, indent=2)
        prompt = build_prompt(obs)
        assert prompt.endswith(state)
        fixed_parts.add(SYSTEM_PROMPT + prompt[: -len(state)] + json.dumps(build_schema()))
    assert len(fixed_parts) == 1
    fixed = fixed_parts.pop().lower()
    assert "expected" not in fixed and "scenario" not in fixed and "rationale" not in fixed
    for name in set(scenario_names()) - option_words:
        assert name.lower() not in fixed


def test_prompt_without_logs_has_no_logs():
    obs = generate(get_scenario("log_only_errors"), 1).to_dict()
    obs.pop("logs")
    assert '"logs"' not in build_prompt(obs)


def test_parser_accepts_valid_answer():
    d = parse_answer(json.dumps(GOOD))
    assert d.status == "ok" and d.severity == "high" and d.human_review == "no"
    assert d.confidence is None and d.human_review_probability is None and d.probabilities == {}


@pytest.mark.parametrize("bad", [
    {**GOOD, "severity": "very high"},
    {**GOOD, "human_review": "maybe"},
    {k: v for k, v in GOOD.items() if k != "action"},
    "not json",
    "[1, 2]",
])
def test_parser_rejects_without_coercing(bad):
    d = parse_answer(bad if isinstance(bad, str) else json.dumps(bad))
    assert d.status == "invalid" and d.severity is None and d.message


# --- Claude API engine (fake SDK client) -------------------------------------

class FakeMessage:
    def __init__(self, text=None, stop_reason="end_turn", category=None, model="claude-sonnet-5-5"):
        self.stop_reason = stop_reason
        self.stop_details = SimpleNamespace(category=category) if category else None
        self.content = [SimpleNamespace(type="text", text=text)] if text is not None else []
        self._model = model

    def to_dict(self):
        return {"model": self._model, "stop_reason": self.stop_reason,
                "usage": {"input_tokens": 900, "output_tokens": 120,
                          "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}


class FakeSDK:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _status_error(cls, code):
    return cls("x", response=httpx2.Response(code, request=httpx2.Request("POST", "https://api.test")),
               body=None)


def api_engine(replies, **kw):
    sdk = FakeSDK(replies)
    return ClaudeAPIEngine("claude-sonnet", client=sdk, sleep=lambda _: None, **kw), sdk


def test_api_happy_path_and_request_shape():
    eng, sdk = api_engine([FakeMessage(json.dumps(GOOD))])
    d = eng.decide({"cpu_pct": 1})
    assert d.status == "ok" and d.action == "investigate"
    assert d.input_tokens == 900 and d.output_tokens == 120 and d.model_version == "claude-sonnet-5-5"
    assert d.cost_usd is None  # no prices set, no invented cost
    call = sdk.calls[0]
    assert call["model"] == "claude-sonnet-5-5" and call["system"] == SYSTEM_PROMPT
    assert call["output_config"]["effort"] == "medium"
    assert call["output_config"]["format"]["schema"] == build_schema()
    assert "temperature" not in call and "fallbacks" not in call
    assert eng.info["transport"] == "anthropic-api"


def test_api_cost_from_user_prices():
    eng, _ = api_engine([FakeMessage(json.dumps(GOOD))], prices=(2.0, 10.0))
    d = eng.decide({})
    assert d.cost_source == "estimated"
    assert d.cost_usd == pytest.approx((900 * 2 + 120 * 10) / 1e6)


def test_api_refusal_is_invalid_and_no_other_model_asked():
    eng, sdk = api_engine([FakeMessage(stop_reason="refusal", category="cyber")])
    d = eng.decide({})
    assert d.status == "invalid" and d.message == "refusal: cyber" and len(sdk.calls) == 1


def test_api_out_of_set_is_invalid():
    eng, _ = api_engine([FakeMessage(json.dumps({**GOOD, "action": "reboot"}))])
    assert eng.decide({}).status == "invalid"


def test_api_retries_rate_limit_then_succeeds():
    eng, sdk = api_engine([_status_error(anthropic.RateLimitError, 429), FakeMessage(json.dumps(GOOD))])
    d = eng.decide({})
    assert d.status == "ok" and d.attempts == 2 and len(sdk.calls) == 2


def test_api_auth_error_is_error_not_retried():
    eng, sdk = api_engine([_status_error(anthropic.AuthenticationError, 401)])
    d = eng.decide({})
    assert d.status == "error" and d.http_status == 401 and len(sdk.calls) == 1


def test_api_connection_failure_is_error_after_retries():
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.test"))
    eng, sdk = api_engine([err] * 4)
    d = eng.decide({})
    assert d.status == "error" and len(sdk.calls) == 4


def test_api_record_has_no_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-secret")
    eng, _ = api_engine([FakeMessage(json.dumps(GOOD))])
    blob = json.dumps(eng.decide({}).to_dict()) + json.dumps(eng.info)
    assert "sk-ant-test-secret" not in blob


# --- Claude CLI fallback (fake subprocess runner) ----------------------------

def cli_body(answer=GOOD, **extra):
    body = {"type": "result", "is_error": False, "result": json.dumps(answer),
            "structured_output": answer, "duration_ms": 4000, "duration_api_ms": 3000,
            "total_cost_usd": 0.0123,
            "usage": {"input_tokens": 5, "cache_creation_input_tokens": 1000,
                      "cache_read_input_tokens": 200, "output_tokens": 80},
            "modelUsage": {"claude-sonnet-5-5": {}}}
    body.update(extra)
    return body


class FakeRunner:
    def __init__(self, stdout="", returncode=0, stderr="", exc=None):
        self.stdout, self.returncode, self.stderr, self.exc = stdout, returncode, stderr, exc
        self.calls = []

    def __call__(self, cmd, *, input, cwd, env, timeout):
        self.calls.append({"cmd": cmd, "input": input, "cwd": cwd, "env": env,
                           "cwd_files": os.listdir(cwd)})
        if self.exc:
            raise self.exc
        return SimpleNamespace(stdout=self.stdout, stderr=self.stderr, returncode=self.returncode)


def test_cli_isolation_flags_and_empty_cwd(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    runner = FakeRunner(json.dumps(cli_body()))
    d = ClaudeCLIEngine("claude-opus", runner=runner).decide({"cpu_pct": 1})
    call = runner.calls[0]
    cmd = call["cmd"]
    assert cmd[:2] == ["claude", "-p"] and cmd[cmd.index("--model") + 1] == "claude-opus-5-5"
    for flag in ("--no-session-persistence", "--strict-mcp-config"):
        assert flag in cmd
    assert cmd[cmd.index("--tools") + 1] == "" and cmd[cmd.index("--setting-sources") + 1] == ""
    assert cmd[cmd.index("--system-prompt") + 1] == SYSTEM_PROMPT
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == build_schema()
    assert call["cwd_files"] == [] and os.path.abspath(call["cwd"]) != os.getcwd()
    assert "CLAUDECODE" not in call["env"]
    assert call["input"] == build_prompt({"cpu_pct": 1})
    assert d.status == "ok"


def test_cli_parses_usage_cost_and_model():
    d = ClaudeCLIEngine("claude-sonnet", runner=FakeRunner(json.dumps(cli_body()))).decide({})
    assert d.input_tokens == 1205 and d.output_tokens == 80
    assert d.cost_usd == 0.0123 and d.cost_source == "reported:claude-cli"
    assert d.model_version == "claude-sonnet-5-5" and d.server_elapsed_ms == 3000


def test_cli_falls_back_to_result_text_when_no_structured_output():
    body = cli_body()
    del body["structured_output"]
    assert ClaudeCLIEngine("claude-sonnet", runner=FakeRunner(json.dumps(body))).decide({}).status == "ok"


def test_cli_out_of_set_is_invalid():
    runner = FakeRunner(json.dumps(cli_body({**GOOD, "severity": "bad"})))
    assert ClaudeCLIEngine("claude-sonnet", runner=runner).decide({}).status == "invalid"


@pytest.mark.parametrize("runner", [
    FakeRunner(returncode=1, stderr="not logged in"),
    FakeRunner(stdout="garbage"),
    FakeRunner(stdout=json.dumps({"is_error": True, "result": "overloaded"})),
    FakeRunner(exc=subprocess.TimeoutExpired("claude", 300)),
    FakeRunner(exc=FileNotFoundError()),
])
def test_cli_failures_are_errors(runner):
    d = ClaudeCLIEngine("claude-sonnet", runner=runner).decide({})
    assert d.status == "error" and d.message


# --- engine factory ----------------------------------------------------------

@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.setattr(engines, "_load_dotenv", lambda: None)
    for k in ("ANTHROPIC_API_KEY", "LLM_EFFORT", "CLAUDE_SONNET_PRICE_INPUT_PER_MTOK",
              "CLAUDE_SONNET_PRICE_OUTPUT_PER_MTOK", "OLLAMA_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(engines, "_cli_version", lambda _: "9.9.9 (Claude Code)")
    return monkeypatch


def test_factory_uses_api_when_key_set(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    err = io.StringIO()
    eng = engines.build_engine("claude-sonnet", which=lambda _: "/usr/bin/claude", err=err)
    assert isinstance(eng, ClaudeAPIEngine) and err.getvalue() == ""


def test_factory_falls_back_to_cli_loudly(clean_env):
    err = io.StringIO()
    eng = engines.build_engine("claude-opus", which=lambda _: "/usr/bin/claude", err=err)
    assert isinstance(eng, ClaudeCLIEngine)
    assert "ANTHROPIC_API_KEY not set" in err.getvalue() and "claude -p" in err.getvalue()
    assert eng.info["transport"] == "claude-cli" and "notice" in eng.info
    assert eng.info["cli_version"] == "9.9.9 (Claude Code)"


def test_factory_neither_key_nor_cli_is_config_error(clean_env):
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY.*claude CLI"):
        engines.build_engine("claude-sonnet", which=lambda _: None, err=io.StringIO())


def test_factory_rejects_bad_effort(clean_env):
    clean_env.setenv("LLM_EFFORT", "turbo")
    with pytest.raises(ConfigError, match="LLM_EFFORT"):
        engines.build_engine("claude-sonnet", which=lambda _: "/usr/bin/claude", err=io.StringIO())


def test_factory_per_engine_prices(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    clean_env.setenv("CLAUDE_SONNET_PRICE_INPUT_PER_MTOK", "2")
    clean_env.setenv("CLAUDE_SONNET_PRICE_OUTPUT_PER_MTOK", "10")
    eng = engines.build_engine("claude-sonnet", err=io.StringIO())
    assert eng.prices == (2.0, 10.0)


def test_factory_ollama_uses_env_url(clean_env):
    clean_env.setenv("OLLAMA_URL", "http://gpu-box:11434/")
    eng = engines.build_engine("ollama:qwen3:8b")
    assert isinstance(eng, OllamaEngine) and eng.model == "qwen3:8b" and eng.url == "http://gpu-box:11434"


@pytest.mark.parametrize("value", ["jev", "claude-sonnet", "claude-opus", "ollama:qwen3:8b"])
def test_parse_engine_accepts(value):
    assert engines.parse_engine(value) == value


@pytest.mark.parametrize("value", ["gpt", "claude", "ollama:", "ollama:  "])
def test_parse_engine_rejects(value):
    with pytest.raises(ValueError, match="valid"):
        engines.parse_engine(value)


def test_engine_info_defaults_to_jev():
    assert engines.engine_info(object())["name"] == "jev"


# --- Ollama engine (canned HTTP) ---------------------------------------------

class Resp:
    def __init__(self, status=200, body=None, text=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        if self._body is None:
            raise ValueError
        return self._body


def ollama(replies):
    calls = []

    def post(url, payload, timeout):
        calls.append((url, payload))
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    return OllamaEngine("qwen3:8b", post=post), calls


def ollama_body(answer=GOOD):
    return {"model": "qwen3:8b", "message": {"role": "assistant", "content": json.dumps(answer)},
            "done": True, "prompt_eval_count": 800, "eval_count": 40, "total_duration": 2_500_000_000}


def test_ollama_request_is_reproducible():
    eng, calls = ollama([Resp(body=ollama_body())])
    d = eng.decide({"cpu_pct": 1}, seed=1234)
    url, payload = calls[0]
    assert url == "http://localhost:11434/api/chat"
    assert payload["options"] == {"temperature": 0, "seed": 1234}
    assert payload["format"] == build_schema() and payload["stream"] is False
    assert payload["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert d.status == "ok" and d.input_tokens == 800 and d.output_tokens == 40
    assert d.server_elapsed_ms == 2500 and d.cost_usd is None


def test_ollama_out_of_set_is_invalid():
    eng, _ = ollama([Resp(body=ollama_body({**GOOD, "action": "panic"}))])
    assert eng.decide({}).status == "invalid"


def test_ollama_unreachable_is_error_naming_url():
    eng, _ = ollama([requests.ConnectionError("refused")])
    d = eng.decide({})
    assert d.status == "error" and "OLLAMA_URL" in d.message


def test_ollama_unknown_model_is_error():
    eng, _ = ollama([Resp(404, text='{"error":"model not found"}')])
    d = eng.decide({})
    assert d.status == "error" and d.http_status == 404


def test_haiku_engine_uses_the_haiku_model_id():
    sdk = FakeSDK([FakeMessage(json.dumps(GOOD), model="claude-haiku-5-5")])
    eng = ClaudeAPIEngine("claude-haiku", client=sdk, sleep=lambda _: None)
    d = eng.decide({"cpu_pct": 1})
    call = sdk.calls[0]
    assert call["model"] == "claude-haiku-5-5" and call["output_config"]["effort"] == "medium"
    assert eng.info["transport"] == "anthropic-api" and d.model_version == "claude-haiku-5-5"
