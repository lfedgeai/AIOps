"""Jev adapter. The only module that knows about Jev's HTTP API.

API details are in docs/jev-api-notes.md (source: https://docs.typesafe.ai/api).
One request per decision, three typed questions:
  severity      -> choice
  action        -> choice
  human_review  -> noul (probability of "yes"; yes if >= HUMAN_REVIEW_THRESHOLD)
  probable_cause -> choice (diagnosis; scored separately from the decision)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import requests

from prompts.loader import DEFAULT_VARIANT, NOUL_OPTIONS, QuestionSpec, Variant, load_variant
from scenarios.definitions import ACTIONS, CAUSES, HUMAN_REVIEW, SEVERITIES

DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
PLACEHOLDER_KEY = "your-key-here"  # the value in .env.example
HUMAN_REVIEW_THRESHOLD = 0.5
RETRY_STATUSES = {429, 500, 502, 503, 504, 529}

STATUS_OK = "ok"
STATUS_INVALID = "invalid"
STATUS_ERROR = "error"

# Tier 1's questions: IDs, types and option keys fixed here; wording from a prompt
# variant (prompts/tier1/*.yaml). QUESTIONS is the baseline, exactly what was sent
# before variants existed (tests/fixtures/questions-tier1-pre-variants.json).
QUESTION_SCHEMA = {
    "severity": QuestionSpec("choice", SEVERITIES),
    "action": QuestionSpec("choice", ACTIONS),
    "probable_cause": QuestionSpec("choice", CAUSES),
    "human_review": QuestionSpec("noul", NOUL_OPTIONS),
}


def load_prompt(name: str = DEFAULT_VARIANT) -> Variant:
    """A Tier 1 prompt variant by name. Raises VariantError for an unknown or bad one."""
    return load_variant("tier1", name, QUESTION_SCHEMA)


QUESTIONS = load_prompt().questions


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class JevConfig:
    api_key: str = field(repr=False)
    url: str = DEFAULT_URL
    model: str = DEFAULT_MODEL
    timeout_s: float = 30.0
    max_retries: int = 3
    # Optional, user-supplied USD prices per million tokens. No default, even
    # though TypeSafe now publishes one (docs/jev-api-notes.md).
    price_input_per_mtok: float | None = None
    price_output_per_mtok: float | None = None


def load_config() -> JevConfig:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # dotenv is optional at runtime
        pass
    key = os.environ.get("JEV_API_KEY", "").strip()
    if not key:
        raise ConfigError(
            "JEV_API_KEY is not set. Copy .env.example to .env and add your key "
            "(get one at https://console.typesafe.ai/)."
        )
    if key == PLACEHOLDER_KEY:
        raise ConfigError(
            "JEV_API_KEY in .env is still the placeholder. Get a key at "
            "https://console.typesafe.ai/"
        )
    return JevConfig(
        api_key=key,
        url=os.environ.get("JEV_API_URL", DEFAULT_URL),
        model=os.environ.get("JEV_MODEL", DEFAULT_MODEL),
        timeout_s=float(os.environ.get("JEV_TIMEOUT_S", "30")),
        max_retries=int(os.environ.get("JEV_MAX_RETRIES", "3")),
        price_input_per_mtok=_env_float("JEV_PRICE_INPUT_PER_MTOK"),
        price_output_per_mtok=_env_float("JEV_PRICE_OUTPUT_PER_MTOK"),
    )


def _env_float(name: str) -> float | None:
    v = os.environ.get(name, "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        raise ConfigError(f"{name} must be a number (USD per million tokens), got {v!r}") from None


@dataclass
class Decision:
    status: str  # ok | invalid | error
    severity: str | None = None
    action: str | None = None
    human_review: str | None = None
    probable_cause: str | None = None
    confidence: float | None = None  # mean of severity and action confidence
    field_confidence: dict = field(default_factory=dict)
    human_review_probability: float | None = None
    probabilities: dict = field(default_factory=dict)
    latency_ms: float | None = None  # round trip of the answering attempt
    # Telemetry
    total_ms: float | None = None  # whole decide() call, incl. retries and backoff
    server_elapsed_ms: float | None = None  # Jev's own elapsedMs
    attempts: int = 0
    http_status: int | None = None
    request_bytes: int | None = None
    response_bytes: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    usage: dict | None = None  # raw usage object, kept whatever it contains
    cost_usd: float | None = None
    cost_source: str | None = None  # reported:usage.<key> | estimated | None
    model_version: str | None = None
    raw_response: dict | None = None
    message: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _num(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def parse_response(body: dict) -> Decision:
    """Validate a Jev response strictly. Anything unexpected is 'invalid'."""

    def bad(msg: str) -> Decision:
        return Decision(status=STATUS_INVALID, raw_response=body, message=msg,
                        model_version=body.get("model") if isinstance(body, dict) else None)

    if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
        return bad("response has no 'answers' object")
    answers = body["answers"]
    out = Decision(status=STATUS_OK, raw_response=body, model_version=body.get("model"))

    for qid, allowed in (("severity", SEVERITIES), ("action", ACTIONS), ("probable_cause", CAUSES)):
        a = answers.get(qid)
        if not isinstance(a, dict) or a.get("choice") not in allowed:
            return bad(f"{qid}: missing or out-of-set choice ({a!r})")
        setattr(out, qid, a["choice"])
        out.field_confidence[qid] = _num(a.get("confidence"))
        if isinstance(a.get("probabilities"), dict):
            out.probabilities[qid] = a["probabilities"]

    hr = answers.get("human_review")
    p = _num(hr.get("noul")) if isinstance(hr, dict) else None
    if p is None or not 0.0 <= p <= 1.0:
        return bad(f"human_review: missing or out-of-range probability ({hr!r})")
    out.human_review_probability = p
    out.human_review = HUMAN_REVIEW[0] if p >= HUMAN_REVIEW_THRESHOLD else HUMAN_REVIEW[1]

    # Overall confidence is for the operational decision only (severity, action).
    confs = [out.field_confidence[q] for q in ("severity", "action")
             if out.field_confidence.get(q) is not None]
    out.confidence = sum(confs) / len(confs) if confs else None
    return out


def _int(x) -> int | None:
    return int(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def apply_usage(d: Decision, body, cfg: JevConfig) -> None:
    """Copy token, timing and cost telemetry from a response body onto a decision.

    Cost: the docs say usage "may include cost in USD" without naming the
    field, so a numeric usage key containing "cost" is used as reported cost.
    Otherwise cost is estimated only if the user set prices. Never invented.
    """
    if not isinstance(body, dict):
        return
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
    d.usage = usage
    elapsed = body.get("elapsedMs", (usage or {}).get("elapsedMs"))
    d.server_elapsed_ms = _num(elapsed)
    if usage:
        d.input_tokens = _int(usage.get("input_tokens"))
        d.output_tokens = _int(usage.get("output_tokens"))
        for key in sorted(usage):
            if "cost" in key.lower() and _num(usage[key]) is not None:
                d.cost_usd, d.cost_source = _num(usage[key]), f"reported:usage.{key}"
                return
    if (cfg.price_input_per_mtok is not None and cfg.price_output_per_mtok is not None
            and d.input_tokens is not None and d.output_tokens is not None):
        d.cost_usd = (d.input_tokens * cfg.price_input_per_mtok
                      + d.output_tokens * cfg.price_output_per_mtok) / 1_000_000
        d.cost_source = "estimated"


PostFn = Callable[[str, dict, dict, float], "requests.Response"]


def _default_post(url, headers, payload, timeout):
    return requests.post(url, headers=headers, json=payload, timeout=timeout)


@dataclass
class Reply:
    """Transport result of one send(): what came back, before any interpretation."""
    status: str = STATUS_ERROR  # ok (HTTP 200 + JSON) | invalid (not JSON) | error
    body: dict | None = None
    message: str = ""
    attempts: int = 0
    http_status: int | None = None
    latency_ms: float | None = None  # round trip of the answering attempt
    total_ms: float | None = None  # whole call, incl. retries and backoff
    request_bytes: int | None = None
    response_bytes: int | None = None


class JevClient:
    def __init__(self, config: JevConfig | None = None, post: PostFn = _default_post,
                 sleep: Callable[[float], None] = time.sleep, prompt: Variant | None = None):
        self.config = config or load_config()
        self.prompt = prompt or load_prompt()  # Tier 1 wording; Tier 2/3 adapters pass their own questions
        self._post = post
        self._sleep = sleep

    def build_request(self, observations: dict, questions: dict | None = None,
                      state_key: str = "server_observations") -> dict:
        """Tier 1 defaults ask this client's prompt variant; Tier 2 and 3 pass their own questions."""
        return {
            "state": {state_key: observations},
            "model": self.config.model,
            "questions": self.prompt.questions if questions is None else questions,
        }

    def send(self, payload: dict) -> Reply:
        """POST one payload with bounded retries. Interprets nothing in the answer."""
        cfg = self.config
        headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
        reply = Reply(request_bytes=len(json.dumps(payload).encode()))
        started = time.perf_counter()
        last = "no attempt made"

        def finish(**kw) -> Reply:
            reply.total_ms = (time.perf_counter() - started) * 1000
            for k, v in kw.items():
                setattr(reply, k, v)
            return reply

        for attempt in range(cfg.max_retries + 1):
            if attempt:
                self._sleep(min(2 ** (attempt - 1), 30))
            reply.attempts += 1
            t0 = time.perf_counter()
            try:
                resp = self._post(cfg.url, headers, payload, cfg.timeout_s)
            except requests.RequestException as e:
                last, reply.http_status = f"{type(e).__name__}", None
                continue
            latency = (time.perf_counter() - t0) * 1000
            reply.http_status = resp.status_code
            if resp.status_code in RETRY_STATUSES:
                last = f"HTTP {resp.status_code}"
                continue
            reply.latency_ms = latency
            reply.response_bytes = len((resp.text or "").encode())
            if resp.status_code != 200:
                return finish(status=STATUS_ERROR,
                              message=f"HTTP {resp.status_code}: {resp.text[:200]}")
            try:
                reply.body = resp.json()
            except ValueError:
                return finish(status=STATUS_INVALID, message="response body is not JSON")
            return finish(status=STATUS_OK)
        reply.latency_ms = reply.response_bytes = None
        return finish(status=STATUS_ERROR, message=f"gave up after retries: {last}")

    def decide(self, observations: dict) -> Decision:
        r = self.send(self.build_request(observations))
        if r.status == STATUS_OK:
            d = parse_response(r.body)
            apply_usage(d, r.body, self.config)  # tokens are billed even for an invalid answer
        else:
            d = Decision(status=r.status, message=r.message)
        d.attempts, d.http_status, d.request_bytes = r.attempts, r.http_status, r.request_bytes
        d.total_ms = r.total_ms
        if r.latency_ms is not None:
            d.latency_ms = r.latency_ms
        if r.response_bytes is not None:
            d.response_bytes = r.response_bytes
        return d
