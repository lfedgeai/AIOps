#!/usr/bin/env python3
"""Preflight: verify configured LLM endpoints return a real chat completion.

Catches the failure mode where an endpoint returns an HTML error page or a
connection error yet the harness records a soft ``detected=false`` and poisons a
full matrix.

By default, agents whose ``*_API_BASE`` or ``*_API_KEY`` is unset are skipped.
Use ``--require-all`` in matrix runs to abort unless every default-lineup agent
is configured and healthy.
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentEndpoint:
    approach: str
    base_env: str
    key_env: str
    model_env: str
    model_default: str


# Keep in sync with AGENTS in code/harness/run_harness.py and the matrix scripts.
DEFAULT_AGENTS: list[AgentEndpoint] = [
    AgentEndpoint("rhdp_deepseek-r1-distill-qwen-14b", "DEEPSEEK_API_BASE", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "deepseek-r1-distill-qwen-14b"),
    AgentEndpoint("litemaas_qwen3.8-27b", "QWEN36_API_BASE", "QWEN36_API_KEY", "QWEN36_MODEL", "Qwen3.8-27B"),
    AgentEndpoint("rhdp_granite-3-2-8b-instruct", "GRANITE_API_BASE", "GRANITE_API_KEY", "GRANITE_MODEL", "granite-3-2-8b-instruct"),
]


def _looks_like_html(text: str) -> bool:
    t = (text or "").lstrip()[:200].lower()
    return t.startswith("<!doctype") or t.startswith("<html") or "<html" in t[:50]


def _post_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> tuple[int, str]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return int(resp.status), body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
        return int(e.code), body


def probe_chat(ep: AgentEndpoint, *, timeout: float) -> tuple[bool, str]:
    base = (os.environ.get(ep.base_env) or "").strip().rstrip("/")
    key = (os.environ.get(ep.key_env) or "").strip()
    model = (os.environ.get(ep.model_env) or ep.model_default).strip()
    if not base:
        return False, f"missing {ep.base_env}"
    if not key:
        return False, f"missing {ep.key_env}"

    url = f"{base}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
        "max_tokens": 16,
        "temperature": 0,
    }
    try:
        status, body = _post_json(url, payload, key, timeout)
    except Exception as e:
        return False, f"request failed: {type(e).__name__}: {e}"

    if _looks_like_html(body):
        return False, f"HTTP {status}: HTML error page (not a chat completion) — endpoint down/auth?"
    if status < 200 or status >= 300:
        return False, f"HTTP {status}: {body[:180].replace(chr(10), ' ')}"

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return False, f"HTTP {status}: non-JSON body: {body[:120]!r}"

    choices = data.get("choices") or []
    if not choices:
        return False, f"HTTP {status}: no choices in response"
    content = ((choices[0].get("message") or {}).get("content")) or ""
    if not str(content).strip() and not (choices[0].get("message") or {}).get("tool_calls"):
        finish = choices[0].get("finish_reason")
        if finish:
            return True, f"OK model={model} finish={finish} (empty content)"
        return False, f"HTTP {status}: empty assistant message"
    return True, f"OK model={model} content={str(content).strip()[:40]!r}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--agents",
        default=os.environ.get("HARNESS_AGENTS", ""),
        help="Comma-separated approach names to probe (default: full default lineup)",
    )
    ap.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout seconds")
    ap.add_argument(
        "--allow-fail",
        action="store_true",
        help="Print failures but exit 0 (smoke only; do not use for matrices)",
    )
    ap.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any default agent's base/key is missing (use for matrices)",
    )
    args = ap.parse_args()

    wanted = {a.strip() for a in args.agents.split(",") if a.strip()}
    endpoints = [e for e in DEFAULT_AGENTS if not wanted or e.approach in wanted]
    if wanted:
        known = {e.approach for e in DEFAULT_AGENTS}
        unknown = wanted - known
        if unknown:
            print(f"[preflight] unknown approaches: {sorted(unknown)}", file=sys.stderr)
            return 1

    to_probe: list[AgentEndpoint] = []
    skipped: list[str] = []
    for ep in endpoints:
        if not os.environ.get(ep.base_env) or not os.environ.get(ep.key_env):
            if args.require_all:
                print(f"[preflight] FAIL {ep.approach}: missing {ep.base_env}/{ep.key_env}", file=sys.stderr)
                return 1
            skipped.append(ep.approach)
            continue
        to_probe.append(ep)

    if not to_probe:
        print("[preflight] no configured agent endpoints (set *_API_BASE and *_API_KEY in .env)", file=sys.stderr)
        return 1

    if skipped:
        print(f"[preflight] skipping unconfigured: {', '.join(skipped)}")
    print(f"[preflight] probing {len(to_probe)} agent API(s) (chat/completions)…")
    failures: list[str] = []
    for ep in to_probe:
        ok, msg = probe_chat(ep, timeout=args.timeout)
        status = "PASS" if ok else "FAIL"
        print(f"[preflight] {status} {ep.approach}: {msg}")
        if not ok:
            failures.append(f"{ep.approach}: {msg}")

    if failures:
        print(
            "[preflight] Agent API preflight FAILED — abort matrix until endpoints are healthy.",
            file=sys.stderr,
        )
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 0 if args.allow_fail else 1

    print("[preflight] all configured agent APIs OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
