from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class LogRecord(TypedDict, total=False):
	timestamp: float
	host: str
	service: str
	level: str
	message: str
	tags: Dict[str, str]
	incident_id: Optional[str]


class MetricRecord(TypedDict, total=False):
	timestamp: float
	host: str
	service: str
	metric: str
	value: float
	tags: Dict[str, str]
	incident_id: Optional[str]


class TraceRecord(TypedDict, total=False):
	timestamp: float
	trace_id: str
	span_id: str
	parent_span_id: Optional[str]
	service: str
	name: str
	duration_ms: float
	tags: Dict[str, str]
	incident_id: Optional[str]


class IncidentRecord(TypedDict, total=False):
	incident_id: str
	start_ts: float
	detected_ts: Optional[float]
	resolved_ts: Optional[float]
	severity: str
	root_cause_summary: Optional[str]
	remediation_summary: Optional[str]

