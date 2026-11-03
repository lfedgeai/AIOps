"""Claude baseline engines: the Anthropic API (official SDK) or the `claude -p` CLI.

API facts and sources are in docs/llm-baseline-notes.md. Server-side refusal
fallbacks are deliberately not enabled: they would let another model answer,
and the benchmark must know which model made each decision.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from typing import Callable

from jev.client import STATUS_ERROR, STATUS_INVALID, Decision
from jev.client import load_prompt
from llm.prompt import SYSTEM_PROMPT, build_prompt, build_schema, parse_answer

MODELS = {"claude-haiku": "claude-haiku-5-5", "claude-sonnet": "claude-sonnet-5-5",
          "claude-opus": "claude-opus-5-5"}
EFFORTS = ("low", "medium", "high", "xhigh", "max")
MAX_TOKENS = 16000  # room for adaptive thinking before the short JSON answer
MAX_RETRIES = 3
TIMEOUT_S = 300.0


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def total_input_tokens(usage: dict) -> int | None:
    """Uncached + cache-write + cache-read input tokens: the whole prompt the model read."""
    parts = [usage.get(k) for k in ("input_tokens", "cache_creation_input_tokens",
                                    "cache_read_input_tokens")]
    parts = [p for p in parts if isinstance(p, int)]
    return sum(parts) if parts else None


def apply_prices(d: Decision, prices: tuple[float | None, float | None]) -> None:
    """Estimate cost only from user-set prices. Never a built-in default."""
    pin, pout = prices
    if (d.cost_usd is None and pin is not None and pout is not None
            and d.input_tokens is not None and d.output_tokens is not None):
        d.cost_usd = (d.input_tokens * pin + d.output_tokens * pout) / 1_000_000
        d.cost_source = "estimated"


class ClaudeAPIEngine:
    """Claude through the official SDK, with JSON-schema structured output."""

    accepts_seed = False

    def __init__(self, name: str, effort: str = "medium", client=None,
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

    def decide(self, observations: dict) -> Decision:
        a = self._anthropic
        prompt = build_prompt(observations, self.prompt.questions)
        kwargs = dict(model=self.model, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
                      messages=[{"role": "user", "content": prompt}],
                      output_config={"format": {"type": "json_schema",
                                                "schema": build_schema(self.prompt.questions)},
                                     "effort": self.effort})
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
                return self._finish(d, prompt, started, attempts, e.status_code)
            latency = (time.perf_counter() - t0) * 1000
            d = self._parse(resp)
            d.latency_ms = latency
            return self._finish(d, prompt, started, attempts, 200)
        d = Decision(status=STATUS_ERROR, message=f"gave up after retries: {last}")
        return self._finish(d, prompt, started, attempts, status)

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
        d.usage = usage
        d.input_tokens = total_input_tokens(usage)
        d.output_tokens = usage.get("output_tokens") if isinstance(usage.get("output_tokens"), int) else None
        d.model_version = raw.get("model")
        d.response_bytes = len(json.dumps(raw).encode())
        apply_prices(d, self.prices)
        return d

    def _finish(self, d: Decision, prompt: str, started: float, attempts: int, status) -> Decision:
        d.total_ms = (time.perf_counter() - started) * 1000
        d.attempts, d.http_status = attempts, status
        d.request_bytes = len((SYSTEM_PROMPT + prompt).encode())
        return d


def _run(cmd, *, input, cwd, env, timeout):
    return subprocess.run(cmd, input=input, cwd=cwd, env=env, timeout=timeout,
                          capture_output=True, text=True)


class ClaudeCLIEngine:
    """Fallback without ANTHROPIC_API_KEY: `claude -p`, isolated from this repo.

    Runs in a fresh empty directory with a replacing system prompt, no tools, no
    setting sources (so no hooks or plugins), no MCP servers and no session
    persistence, so nothing from the repository, CLAUDE.md or memory reaches the model.
    """

    accepts_seed = False

    def __init__(self, name: str, effort: str = "medium", claude: str = "claude",
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

    @staticmethod
    def env() -> dict:
        # Drop markers of the parent Claude Code session so the child runs standalone.
        return {k: v for k, v in os.environ.items()
                if k != "CLAUDECODE" and not k.startswith("CLAUDE_CODE_")}

    def decide(self, observations: dict) -> Decision:
        prompt = build_prompt(observations, self.prompt.questions)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="llm-baseline-") as cwd:
            try:
                proc = self._run(self.command(), input=prompt, cwd=cwd, env=self.env(),
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
        d.usage = usage
        d.input_tokens = total_input_tokens(usage)
        d.output_tokens = usage.get("output_tokens") if isinstance(usage.get("output_tokens"), int) else None
        models = body.get("modelUsage")
        d.model_version = next(iter(models), None) if isinstance(models, dict) and models else None
        d.server_elapsed_ms = _num(body.get("duration_api_ms"))
        cost = _num(body.get("total_cost_usd"))
        if cost is not None:
            d.cost_usd, d.cost_source = cost, "reported:claude-cli"
        apply_prices(d, self.prices)
        d.latency_ms = latency
        d.response_bytes = len(proc.stdout.encode())
        return self._finish(d, prompt, started)

    def _finish(self, d: Decision, prompt: str, started: float) -> Decision:
        d.total_ms = (time.perf_counter() - started) * 1000
        d.attempts = 1
        d.request_bytes = len((SYSTEM_PROMPT + prompt).encode())
        return d
