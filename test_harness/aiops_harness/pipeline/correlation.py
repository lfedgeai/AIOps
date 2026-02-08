from __future__ import annotations

from typing import Any, Dict, List, Tuple
from ..llm.base import LLMProvider, LLMMessage
from ..data.schema import LogRecord, MetricRecord, TraceRecord


def summarize_events_for_prompt(logs: List[LogRecord], metrics: List[MetricRecord], traces: List[TraceRecord], limit: int = 100) -> str:
	def fmt_log(x: LogRecord) -> str:
		return f"[log ts={x.get('timestamp'):.3f} svc={x.get('service')} level={x.get('level')}] {x.get('message')}"
	def fmt_metric(x: MetricRecord) -> str:
		return f"[metric ts={x.get('timestamp'):.3f} svc={x.get('service')} {x.get('metric')}={x.get('value')}]"
	def fmt_trace(x: TraceRecord) -> str:
		return f"[trace ts={x.get('timestamp'):.3f} svc={x.get('service')} name={x.get('name')} dur_ms={x.get('duration_ms')}]"
	parts: List[str] = []
	for rec in logs[: limit]:
		parts.append(fmt_log(rec))
	for rec in metrics[: limit]:
		parts.append(fmt_metric(rec))
	for rec in traces[: limit]:
		parts.append(fmt_trace(rec))
	return "\n".join(parts)


def correlate_signals(provider: LLMProvider, prompt_template: str, logs: List[LogRecord], metrics: List[MetricRecord], traces: List[TraceRecord], limit: int = 100, model: str | None = None) -> Dict[str, Any]:
	context = summarize_events_for_prompt(logs, metrics, traces, limit=limit)
	user_prompt = f"{prompt_template}\n\nEvents:\n{context}\n\nReturn JSON with keys: correlated_groups (list of lists of string ids or descriptions), incident_summary (string)."
	messages: List[LLMMessage] = [
		{"role": "system", "content": "You are an SRE expert performing signal correlation."},
		{"role": "user", "content": user_prompt},
	]
	resp = provider.chat(messages, model=model, temperature=0.2)
	return {"raw": resp, "content": resp.get("content", ""), "prompt": user_prompt}

