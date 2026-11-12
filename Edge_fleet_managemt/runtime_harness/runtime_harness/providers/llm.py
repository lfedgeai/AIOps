"""LLM provider — MaaS chat completions with structured-output validation.

Purpose
-------
Give every LLM stage a single ``complete(agent_id, system, user, schema)``
entry point that returns a **validated pydantic instance**. No mock backend:
missing credentials fail loudly, unparseable output surfaces as an error
after one repair retry.

How it connects
---------------
- Constructed once in ``Assembly`` from ``providers.backend`` (maas only).
- Slots call ``complete`` with their stage schema (``schemas.py``).
- Env var mapping matches the research harness preflight
  (GRANITE_*, GLM_*, DEEPSEEK_*, QWEN3_*, ...).
"""

from __future__ import annotations

import os
from typing import Any, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from runtime_harness.schemas import extract_json

T = TypeVar("T", bound=BaseModel)


class ProviderError(RuntimeError):
    """Missing credentials / unreachable backend / unparseable output."""


# agent_id -> (base_url_env, api_key_env, model_env)
_ENV_MAP: dict[str, tuple[str, str, str]] = {
    "maas_granite": ("GRANITE_API_BASE", "GRANITE_API_KEY", "GRANITE_MODEL"),
    "maas_glm-53-flash": ("GLM_API_BASE", "GLM_API_KEY", "GLM_MODEL"),
    "maas_deepseek": ("DEEPSEEK_API_BASE", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL"),
    "maas_qwen3": ("QWEN3_API_BASE", "QWEN3_API_KEY", "QWEN3_MODEL"),
    "maas_qwen36-27b": ("QWEN36_API_BASE", "QWEN36_API_KEY", "QWEN36_MODEL"),
    "maas_qwen38-27b": ("QWEN38_API_BASE", "QWEN38_API_KEY", "QWEN38_MODEL"),
}


class LLMProvider:
    """OpenAI-compatible chat client with per-stage response schemas."""

    def __init__(self, backend: str = "maas"):
        if backend not in ("maas", "mock"):
            raise ProviderError(f"unknown backend: {backend}")
        self.backend = "maas"  # mock removed; kept for config compat

    def _env_for(self, agent_id: str) -> tuple[Optional[str], Optional[str], str]:
        base_k, key_k, model_k = _ENV_MAP.get(
            agent_id, ("LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL")
        )
        return os.getenv(base_k), os.getenv(key_k), os.getenv(model_k, agent_id)

    def _chat(self, agent_id: str, system: str, user: str) -> str:
        return self._chat_messages(
            agent_id,
            system,
            [{"role": "user", "content": user}],
        )

    def _chat_messages(self, agent_id: str, system: str, messages: list[dict[str, str]]) -> str:
        base, key, model = self._env_for(agent_id)
        if not base or not key:
            raise ProviderError(
                f"missing credentials for {agent_id} — export "
                f"{_ENV_MAP.get(agent_id, ('LLM_API_BASE','LLM_API_KEY','LLM_MODEL'))[:2]}"
            )
        url = base.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": 0.1,
        }
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"{agent_id}: MaaS request failed: {exc}") from exc
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{agent_id}: unexpected MaaS response shape") from exc

    def complete(self, agent_id: str, system: str, user: str, schema: Type[T]) -> T:
        """One completion + validation, with a single repair retry."""
        content = self._chat(agent_id, system, user)
        try:
            return self._parse(content, schema)
        except (ValueError, ValidationError):
            # repair retry: re-ask with explicit schema instruction
            repair_user = (
                user
                + "\n\nYour previous reply was not valid JSON. Respond with ONLY a "
                f"JSON object matching this schema (no markdown, no prose): "
                f"{_schema_hint(schema)}"
            )
            content2 = self._chat(agent_id, system, repair_user)
            try:
                return self._parse(content2, schema)
            except (ValueError, ValidationError) as exc:
                raise ProviderError(
                    f"{agent_id}: output failed schema validation after repair retry: {exc}"
                ) from exc

    @staticmethod
    def _parse(content: str, schema: Type[T]) -> T:
        obj = extract_json(content)
        if obj is None:
            raise ValueError("no JSON object in output")
        return schema.model_validate(obj)


def _schema_hint(schema: Type[BaseModel]) -> str:
    props = {
        name: (f.field_type.__name__ if hasattr(f, "field_type") else "any")
        for name, f in schema.model_fields.items()
    }
    return str(props)


def wiring_info(agent_ids: list[str]) -> list[dict[str, Any]]:
    """Endpoint wiring for the UI — base URL + served model per agent.

    API keys are NEVER included; only whether one is set.
    """
    out: list[dict[str, Any]] = []
    for aid in dict.fromkeys(agent_ids):  # dedupe, keep order
        base_k, key_k, model_k = _ENV_MAP.get(
            aid, ("LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL")
        )
        out.append(
            {
                "id": aid,
                "base_url": os.getenv(base_k),
                "model": os.getenv(model_k, aid),
                "api_key_set": bool(os.getenv(key_k)),
                "env": {"base": base_k, "key": key_k, "model": model_k},
            }
        )
    return out


__all__ = ["LLMProvider", "ProviderError", "wiring_info"]
