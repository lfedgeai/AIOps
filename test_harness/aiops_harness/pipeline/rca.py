from __future__ import annotations

from typing import Any, Dict, List
from ..llm.base import LLMProvider, LLMMessage


def root_cause_analysis(provider: LLMProvider, prompt_template: str, correlated_context: str, model: str | None = None) -> Dict[str, Any]:
	messages: List[LLMMessage] = [
		{"role": "system", "content": "You are an SRE expert performing root cause analysis."},
		{"role": "user", "content": f"{prompt_template}\n\nContext:\n{correlated_context}\n\nReturn JSON with keys: root_cause (string), confidence (0-1)."},
	]
	resp = provider.chat(messages, model=model, temperature=0.2)
	return {"raw": resp, "content": resp.get("content", "")}

