from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
from .data.schema import IncidentRecord


def _safe_delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
	if a is None or b is None:
		return None
	return float(a - b)


def compute_mttd(incidents: List[IncidentRecord]) -> Optional[float]:
	deltas = []
	for inc in incidents:
		delta = _safe_delta(inc.get("detected_ts"), inc.get("start_ts"))
		if delta is not None and delta >= 0:
			deltas.append(delta)
	return float(np.mean(deltas)) if deltas else None


def compute_mttr(incidents: List[IncidentRecord]) -> Optional[float]:
	deltas = []
	for inc in incidents:
		delta = _safe_delta(inc.get("resolved_ts"), inc.get("start_ts"))
		if delta is not None and delta >= 0:
			deltas.append(delta)
	return float(np.mean(deltas)) if deltas else None


def compute_mmtd(model_detection_ts: List[Optional[float]], earliest_related_signal_ts: List[Optional[float]]) -> Optional[float]:
	deltas = []
	for md, es in zip(model_detection_ts, earliest_related_signal_ts):
		delta = _safe_delta(md, es)
		if delta is not None and delta >= 0:
			deltas.append(delta)
	return float(np.mean(deltas)) if deltas else None

