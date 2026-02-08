from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import httpx
from .base import LLMProvider, LLMMessage, LLMResponse


class LlamaStackProvider(LLMProvider):
	def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, default_model: Optional[str] = None, timeout_s: int = 60):
		self.base_url = base_url or os.environ.get("LLAMA_STACK_BASE_URL", "http://localhost:8080")
		# Be forgiving if user passed without scheme
		if isinstance(self.base_url, str) and "://" not in self.base_url:
			self.base_url = f"http://{self.base_url}"
		self.api_key = api_key or os.environ.get("LLAMA_STACK_API_KEY", "")
		self.default_model = default_model or os.environ.get("LLAMA_STACK_MODEL", "llama3.1")
		self.timeout_s = timeout_s

	def name(self) -> str:
		return "llamastack"

	def chat(self, messages: List[LLMMessage], model: Optional[str] = None, **kwargs: Any) -> LLMResponse:
		headers = {}
		if self.api_key:
			headers["Authorization"] = f"Bearer {self.api_key}"

		payload: Dict[str, Any] = {
			"model": model or self.default_model,
			"messages": messages,
		}
		payload.update(kwargs or {})

		# Assuming LlamaStack exposes an OpenAI-compatible /chat/completions
		url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
		with httpx.Client(timeout=self.timeout_s) as client:
			resp = client.post(url, headers=headers, json=payload)
			resp.raise_for_status()
			data = resp.json()

		# Normalize to LLMResponse
		content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
		usage = data.get("usage", {})
		return {"content": content, "usage": usage, "raw": data}

