from __future__ import annotations

from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod


class LLMMessage(Dict[str, str]):
	"""
	Convenience alias for messages, e.g. {"role": "user", "content": "..."}.
	"""


class LLMResponse(Dict[str, Any]):
	"""
	Standardized response with at least:
	- "content": str
	- "usage": Dict[str, int] (optional)
	"""


class LLMProvider(ABC):
	@abstractmethod
	def name(self) -> str:
		raise NotImplementedError

	@abstractmethod
	def chat(self, messages: List[LLMMessage], model: Optional[str] = None, **kwargs: Any) -> LLMResponse:
		raise NotImplementedError

