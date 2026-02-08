from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import httpx
from .base import LLMProvider, LLMMessage, LLMResponse


class OllamaProvider(LLMProvider):
    """
    Minimal Ollama chat client. Defaults to http://127.0.0.1:11434
    API: POST /api/chat
    Docs: https://github.com/ollama/ollama/blob/main/docs/api.md#chat
    """

    def __init__(self, base_url: Optional[str] = None, default_model: Optional[str] = None, timeout_s: int = 120):
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        if "://" not in self.base_url:
            self.base_url = f"http://{self.base_url}"
        self.default_model = default_model or os.environ.get("OLLAMA_MODEL", "llama3.1")
        self.timeout_s = timeout_s

    def name(self) -> str:
        return "ollama"

    def chat(self, messages: List[LLMMessage], model: Optional[str] = None, **kwargs: Any) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": False,
        }
        # Map common kwargs if present
        if "temperature" in kwargs:
            payload["options"] = payload.get("options", {})
            payload["options"]["temperature"] = kwargs["temperature"]

        url = f"{self.base_url.rstrip('/')}/api/chat"
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = (data.get("message") or {}).get("content", "")
        # Ollama returns eval_count/eval_duration; map lightly to usage
        usage = {
            "eval_count": data.get("eval_count"),
            "eval_duration": data.get("eval_duration"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "prompt_eval_duration": data.get("prompt_eval_duration"),
        }
        return {"content": content, "usage": usage, "raw": data}

