"""Model context window lookup and AI metrics enrichment."""
from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from typing import Any

from code.agents.ai_metrics import RoundMetrics, aggregate_metrics

_LIMIT_CACHE: dict[tuple[str, str], int | None] = {}


def lookup_model_context_limit(api_base: str, api_key: str, model: str) -> int | None:
    """Return max_model_len from GET {base}/models when exposed by the endpoint."""
    base = (api_base or "").strip().rstrip("/")
    key = (api_key or "").strip()
    model = (model or "").strip()
    if not base or not key or not model:
        return None
    cache_key = (base, model)
    if cache_key in _LIMIT_CACHE:
        return _LIMIT_CACHE[cache_key]

    url = f"{base}/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    limit: int | None = None
    try:
        with urllib.request.urlopen(req, timeout=20, context=ssl.create_default_context()) as resp:
            data = json.loads(resp.read().decode())
        models = data.get("data") or []
        match = next((m for m in models if m.get("id") == model), models[0] if models else {})
        raw = match.get("max_model_len") or match.get("context_length")
        if raw is not None:
            limit = int(raw)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError):
        limit = None

    _LIMIT_CACHE[cache_key] = limit
    return limit


def build_ai_metrics(
    rounds: list[RoundMetrics],
    api_base: str,
    api_key: str,
    model: str,
) -> dict[str, float]:
    """Aggregate round metrics and attach peak prompt usage vs model context limit."""
    limit = lookup_model_context_limit(api_base, api_key, model)
    metrics = aggregate_metrics(rounds, model_context_limit=limit)
    peak = int(metrics.get("ai_peak_prompt_tokens") or 0)
    if limit and limit > 0:
        avail = max(0, limit - peak)
        util = round(100.0 * peak / limit, 2) if peak else 0.0
        print(
            f"[agent] context window: peak_prompt={peak} limit={limit} "
            f"available={avail} utilization={util}%",
            flush=True,
        )
    elif peak:
        print(
            f"[agent] context window: peak_prompt={peak} limit=unknown (endpoint did not expose max_model_len)",
            flush=True,
        )
    return metrics
