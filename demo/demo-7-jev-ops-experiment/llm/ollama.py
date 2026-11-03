"""Local baseline engine: a model served by Ollama, over its HTTP API.

Temperature 0 and the run seed make a local run reproducible. Cost is always
n/a (local). The model has no default: it comes from --engine ollama:<model>.
"""
from __future__ import annotations

import json
import time
from typing import Callable

import requests

from jev.client import STATUS_ERROR, Decision
from jev.client import load_prompt
from llm.prompt import SYSTEM_PROMPT, build_prompt, build_schema, parse_answer

DEFAULT_URL = "http://localhost:11434"
TIMEOUT_S = 300.0


def _default_post(url, payload, timeout):
    return requests.post(url, json=payload, timeout=timeout)


class OllamaEngine:
    accepts_seed = True

    def __init__(self, model: str, url: str = DEFAULT_URL,
                 post: Callable = _default_post, prompt=None):
        self.model = model
        self.url = url.rstrip("/")
        self._post = post
        self.prompt = prompt or load_prompt()
        self.info = {"name": f"ollama:{model}", "transport": "ollama", "model_requested": model,
                     "ollama_url": self.url}

    def build_request(self, observations: dict, seed: int | None = None) -> dict:
        options = {"temperature": 0}
        if seed is not None:
            options["seed"] = seed
        return {"model": self.model, "stream": False, "format": build_schema(self.prompt.questions),
                "options": options,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": build_prompt(observations, self.prompt.questions)}]}

    def decide(self, observations: dict, seed: int | None = None) -> Decision:
        payload = self.build_request(observations, seed)
        started = time.perf_counter()
        try:
            resp = self._post(f"{self.url}/api/chat", payload, TIMEOUT_S)
        except requests.RequestException as e:
            d = Decision(status=STATUS_ERROR,
                         message=f"Ollama not reachable at {self.url} ({type(e).__name__}); check OLLAMA_URL")
            return self._finish(d, payload, started, None)
        latency = (time.perf_counter() - started) * 1000
        if resp.status_code != 200:
            d = Decision(status=STATUS_ERROR, message=f"HTTP {resp.status_code}: {(resp.text or '')[:200]}")
            return self._finish(d, payload, started, resp.status_code)
        try:
            body = resp.json()
        except ValueError:
            d = Decision(status=STATUS_ERROR, message="Ollama response is not JSON")
            return self._finish(d, payload, started, resp.status_code)
        content = (body.get("message") or {}).get("content") if isinstance(body, dict) else None
        d = parse_answer(content, raw=body, questions=self.prompt.questions)
        d.raw_response = body
        if isinstance(body, dict):
            d.input_tokens = body.get("prompt_eval_count") if isinstance(body.get("prompt_eval_count"), int) else None
            d.output_tokens = body.get("eval_count") if isinstance(body.get("eval_count"), int) else None
            dur = body.get("total_duration")
            d.server_elapsed_ms = dur / 1e6 if isinstance(dur, (int, float)) else None
            d.model_version = body.get("model")
        d.latency_ms = latency
        d.response_bytes = len((resp.text or "").encode())
        return self._finish(d, payload, started, resp.status_code)

    @staticmethod
    def _finish(d: Decision, payload: dict, started: float, status) -> Decision:
        d.total_ms = (time.perf_counter() - started) * 1000
        d.attempts, d.http_status = 1, status
        d.request_bytes = len(json.dumps(payload).encode())
        return d
