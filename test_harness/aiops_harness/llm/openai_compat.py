from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import httpx
from .base import LLMProvider, LLMMessage, LLMResponse


class OpenAICompatibleProvider(LLMProvider):
	def __init__(self, api_base: Optional[str] = None, api_key: Optional[str] = None, default_model: Optional[str] = None, timeout_s: int = 60):
		self.api_base = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
		if isinstance(self.api_base, str) and "://" not in self.api_base:
			self.api_base = f"https://{self.api_base}"
		self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
		self.default_model = default_model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
		self.timeout_s = timeout_s

	def name(self) -> str:
		return "openai_compat"

	def chat(self, messages: List[LLMMessage], model: Optional[str] = None, **kwargs: Any) -> LLMResponse:
		headers = {"Authorization": f"Bearer {self.api_key}"}
		payload: Dict[str, Any] = {
			"model": model or self.default_model,
			"messages": messages,
		}
		payload.update(kwargs or {})
		url = f"{self.api_base.rstrip('/')}/chat/completions"
		with httpx.Client(timeout=self.timeout_s) as client:
			resp = client.post(url, headers=headers, json=payload)
			resp.raise_for_status()
			data = resp.json()
		content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
		usage = data.get("usage", {})
		return {"content": content, "usage": usage, "raw": data}

