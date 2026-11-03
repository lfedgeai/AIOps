"""Claude baseline engine for Tier 3 (and Tier 4): the same observation and the same four
questions as the Jev adapter, answered by Claude through the Anthropic API, or through
`claude -p` when ANTHROPIC_API_KEY is not set (as in Tier 1, labelled transport=claude-cli).

The schema restricts every answer to the question's options and the parser is strict:
an out-of-set or missing answer is 'invalid'. Settings (model IDs, effort, max_tokens,
retries, CLI isolation flags) are llm/claude.py's; API facts are in docs/llm-baseline-notes.md.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable

from llm.claude import ClaudeCLIEngine as _Tier1CLIEngine
from llm.claude import EFFORTS, MAX_RETRIES, MAX_TOKENS, MODELS, TIMEOUT_S, _num, _run, total_input_tokens
from llm.engines import _prices, fallback_notice
from tier3.engine import STATUS_ERROR, STATUS_INVALID, STATUS_OK, Decision
from tier3.questions import QUESTIONS, load_prompt

SYSTEM_PROMPT = ("You assess the observed state of an application running on a Kubernetes "
                 "cluster and answer each question with exactly one of the allowed options.")
STATE_KEY = "observations"  # the key the Jev adapter wraps the observation in


class ConfigError(Exception):
    pass  # a bad setting, raised before any call


def options(qid: str, questions: dict = QUESTIONS) -> dict[str, str]:
    return dict(questions[qid]["criteria"])


def build_schema(questions: dict = QUESTIONS) -> dict:
    return {
        "type": "object",
        "properties": {qid: {"type": "string", "enum": list(options(qid, questions))} for qid in questions},
        "required": list(questions),
        "additionalProperties": False,
    }


def build_prompt(observation: dict, questions: dict = QUESTIONS) -> str:
    lines = ["Questions. Answer each one with exactly one allowed option.", ""]
    for qid, q in questions.items():
        lines.append(f"{qid}: {q['instructions']}")
        lines += [f"  - {opt}: {desc}" for opt, desc in options(qid, questions).items()]
        lines.append("")
    lines += ["Reply with a JSON object with the keys " + ", ".join(questions) + ".", "",
              "State:", json.dumps({STATE_KEY: observation}, indent=2)]
    return "\n".join(lines)


def parse_answer(data, raw=None, questions: dict = QUESTIONS) -> Decision:
    """Strict: all four fields, each in its allowed set, or 'invalid'. Never coerces."""
    raw = data if raw is None else raw
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            return Decision(status=STATUS_INVALID, raw_response=raw, message="reply is not JSON")
    if not isinstance(data, dict):
        return Decision(status=STATUS_INVALID, raw_response=raw, message="reply is not a JSON object")
    out = Decision(status=STATUS_OK, raw_response=raw)
    for qid in questions:
        v = data.get(qid)
        if not isinstance(v, str) or v not in options(qid, questions):
            return Decision(status=STATUS_INVALID, raw_response=raw,
                            message=f"{qid}: missing or out-of-set value ({v!r})")
        setattr(out, qid, v == "true" if qid == "incident_detected" else v)
    return out


def _apply_prices(tele: dict, prices) -> None:
    pin, pout = prices
    if (tele.get("cost_usd") is None and pin is not None and pout is not None
            and tele.get("input_tokens") is not None and tele.get("output_tokens") is not None):
        tele["cost_usd"] = (tele["input_tokens"] * pin + tele["output_tokens"] * pout) / 1_000_000
        tele["cost_source"] = "estimated"


class ClaudeAPIEngine:
    """AIOpsEngine: Claude through the official SDK, JSON-schema structured output."""

    def __init__(self, name: str = "claude-opus", effort: str = "medium", client=None,
                 prices: tuple[float | None, float | None] = (None, None),
                 sleep: Callable[[float], None] = time.sleep, prompt=None):
        import anthropic

        self._anthropic = anthropic
        self.model = MODELS[name]
        self.effort = effort
        self.prices = prices
        self._client = client or anthropic.Anthropic(max_retries=0, timeout=TIMEOUT_S)
        self._sleep = sleep
        self.prompt = prompt or load_prompt()
        self.info = {"name": name, "transport": "anthropic-api", "model_requested": self.model,
                     "effort": effort}

    def request(self, observation: dict) -> dict:
        return dict(model=self.model, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": build_prompt(observation, self.prompt.questions)}],
                    output_config={"format": {"type": "json_schema",
                                                "schema": build_schema(self.prompt.questions)},
                                   "effort": self.effort})

    def decide(self, observation: dict) -> Decision:
        a = self._anthropic
        kwargs = self.request(observation)
        started = time.perf_counter()
        attempts, last, status = 0, "no attempt made", None
        for attempt in range(MAX_RETRIES + 1):
            if attempt:
                self._sleep(min(2 ** (attempt - 1), 30))
            attempts += 1
            t0 = time.perf_counter()
            try:
                resp = self._client.messages.create(**kwargs)
            except (a.RateLimitError, a.InternalServerError) as e:
                last, status = f"HTTP {e.status_code}", e.status_code
                continue
            except a.APIConnectionError as e:
                last, status = type(e).__name__, None
                continue
            except a.APIStatusError as e:
                d = Decision(status=STATUS_ERROR, message=f"HTTP {e.status_code}: {str(e.message)[:200]}")
                return self._finish(d, kwargs, started, attempts, e.status_code)
            d = self._parse(resp)
            d.telemetry["latency_ms"] = (time.perf_counter() - t0) * 1000
            return self._finish(d, kwargs, started, attempts, 200)
        d = Decision(status=STATUS_ERROR, message=f"gave up after retries: {last}")
        return self._finish(d, kwargs, started, attempts, status)

    def _parse(self, resp) -> Decision:
        raw = resp.to_dict()
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            d = Decision(status=STATUS_INVALID, raw_response=raw,
                         message=f"refusal: {getattr(details, 'category', None)}")
        elif resp.stop_reason == "max_tokens":
            d = Decision(status=STATUS_INVALID, raw_response=raw, message="reply cut off at max_tokens")
        else:
            text = next((b.text for b in resp.content if b.type == "text"), None)
            d = parse_answer(text, raw=raw, questions=self.prompt.questions) if text is not None else Decision(
                status=STATUS_INVALID, raw_response=raw, message="reply has no text block")
            d.raw_response = raw
        usage = raw.get("usage") or {}
        out = usage.get("output_tokens")
        d.telemetry.update(usage=usage, input_tokens=total_input_tokens(usage),
                           output_tokens=out if isinstance(out, int) else None,
                           cost_usd=None, cost_source=None, model=raw.get("model"),
                           response_bytes=len(json.dumps(raw).encode()))
        _apply_prices(d.telemetry, self.prices)
        return d

    def _finish(self, d: Decision, kwargs: dict, started: float, attempts: int, status) -> Decision:
        d.telemetry.update(total_ms=(time.perf_counter() - started) * 1000, attempts=attempts,
                           http_status=status, engine=dict(self.info),
                           request_bytes=len((kwargs["system"] + kwargs["messages"][0]["content"]).encode()))
        return d


class ClaudeCLIEngine:
    """AIOpsEngine fallback without ANTHROPIC_API_KEY: `claude -p`, isolated exactly as the
    Tier 1 CLI engine (fresh empty directory, no tools, settings, MCP or session)."""

    def __init__(self, name: str = "claude-opus", effort: str = "medium", claude: str = "claude",
                 runner=_run, prices: tuple[float | None, float | None] = (None, None),
                 cli_version: str | None = None, prompt=None):
        self.model = MODELS[name]
        self.effort = effort
        self.claude = claude
        self.prices = prices
        self._run = runner
        self.prompt = prompt or load_prompt()
        self.info = {"name": name, "transport": "claude-cli", "model_requested": self.model,
                     "effort": effort, "cli_version": cli_version}

    def command(self) -> list[str]:
        return [self.claude, "-p", "--model", self.model, "--output-format", "json",
                "--json-schema", json.dumps(build_schema(self.prompt.questions)),
                "--system-prompt", SYSTEM_PROMPT,
                "--tools", "", "--effort", self.effort, "--no-session-persistence",
                "--strict-mcp-config", "--setting-sources", ""]

    def decide(self, observation: dict) -> Decision:
        prompt = build_prompt(observation, self.prompt.questions)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="llm-baseline-") as cwd:
            try:
                proc = self._run(self.command(), input=prompt, cwd=cwd, env=_Tier1CLIEngine.env(),
                                 timeout=TIMEOUT_S)
            except (OSError, subprocess.SubprocessError) as e:
                return self._finish(Decision(status=STATUS_ERROR, message=f"claude CLI: {type(e).__name__}"),
                                    prompt, started)
        latency = (time.perf_counter() - started) * 1000
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip()[:200]
            return self._finish(Decision(status=STATUS_ERROR,
                                         message=f"claude CLI exit {proc.returncode}: {msg}"), prompt, started)
        try:
            body = json.loads(proc.stdout)
        except ValueError:
            return self._finish(Decision(status=STATUS_ERROR, message="claude CLI output is not JSON"),
                                prompt, started)
        if not isinstance(body, dict) or body.get("is_error"):
            msg = str(body.get("result") if isinstance(body, dict) else body)[:200]
            return self._finish(Decision(status=STATUS_ERROR, message=f"claude CLI error: {msg}"),
                                prompt, started)
        answer = body.get("structured_output")
        d = parse_answer(answer if isinstance(answer, dict) else body.get("result"), raw=body,
                         questions=self.prompt.questions)
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        models = body.get("modelUsage")
        out = usage.get("output_tokens")
        cost = _num(body.get("total_cost_usd"))
        d.telemetry.update(usage=usage, input_tokens=total_input_tokens(usage),
                           output_tokens=out if isinstance(out, int) else None,
                           model=next(iter(models), None) if isinstance(models, dict) and models else None,
                           server_elapsed_ms=_num(body.get("duration_api_ms")),
                           cost_usd=cost, cost_source="reported:claude-cli" if cost is not None else None,
                           latency_ms=latency, response_bytes=len(proc.stdout.encode()))
        _apply_prices(d.telemetry, self.prices)
        return self._finish(d, prompt, started)

    def _finish(self, d: Decision, prompt: str, started: float) -> Decision:
        d.telemetry.update(total_ms=(time.perf_counter() - started) * 1000, attempts=1,
                           engine=dict(self.info),
                           request_bytes=len((SYSTEM_PROMPT + prompt).encode()))
        return d


def build_claude_engine(name: str = "claude-opus", *, which=shutil.which, err=sys.stderr,
                        prompt=None):
    """API engine when ANTHROPIC_API_KEY is set, else the CLI fallback with a loud notice.
    Raises ConfigError before any call."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    effort = os.environ.get("LLM_EFFORT", "medium").strip() or "medium"
    if effort not in EFFORTS:
        raise ConfigError(f"LLM_EFFORT must be one of {', '.join(EFFORTS)}, got {effort!r}")
    try:
        prices = _prices(name)
    except Exception as e:  # noqa: BLE001 - a malformed price variable
        raise ConfigError(str(e)) from e
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return ClaudeAPIEngine(name, effort=effort, prices=prices, prompt=prompt)
    claude = which("claude")
    if not claude:
        raise ConfigError(f"{name} needs either ANTHROPIC_API_KEY (Anthropic API) or the claude "
                          "CLI on PATH (fallback). Neither was found.")
    notice = fallback_notice(MODELS[name])
    bar = "*" * 78
    print(f"\n{bar}\n*** {notice}\n{bar}\n", file=err)
    try:
        version = subprocess.run([claude, "--version"], capture_output=True, text=True,
                                 timeout=15).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        version = None
    engine = ClaudeCLIEngine(name, effort=effort, claude=claude, prices=prices, cli_version=version,
                             prompt=prompt)
    engine.info["notice"] = notice
    return engine
