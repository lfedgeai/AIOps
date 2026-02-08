from __future__ import annotations

from typing import Any, Dict, List
from ..llm.base import LLMProvider, LLMMessage


def suggest_remediation(provider: LLMProvider, prompt_template: str, rca_summary: str, max_steps: int = 5, model: str | None = None) -> Dict[str, Any]:
	messages: List[LLMMessage] = [
		{"role": "system", "content": "You are an SRE expert suggesting safe remediation steps."},
		{"role": "user", "content": f"{prompt_template}\n\nRCA:\n{rca_summary}\n\nReturn JSON with keys: steps (list of strings, up to {max_steps}), rollback (list of strings)."},
	]
	resp = provider.chat(messages, model=model, temperature=0.2)
	return {"raw": resp, "content": resp.get("content", "")}

