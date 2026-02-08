from __future__ import annotations

import json
import os
from typing import Iterable, List, Tuple
from .schema import LogRecord, MetricRecord, TraceRecord, IncidentRecord


def read_jsonl(path: str) -> Iterable[dict]:
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			yield json.loads(line)


def load_dataset(root_dir: str, logs_file: str, metrics_file: str, traces_file: str, incidents_file: str) -> Tuple[List[LogRecord], List[MetricRecord], List[TraceRecord], List[IncidentRecord]]:
	logs_path = os.path.join(root_dir, logs_file)
	metrics_path = os.path.join(root_dir, metrics_file)
	traces_path = os.path.join(root_dir, traces_file)
	incidents_path = os.path.join(root_dir, incidents_file)

	logs: List[LogRecord] = [LogRecord(**x) for x in read_jsonl(logs_path)]
	metrics: List[MetricRecord] = [MetricRecord(**x) for x in read_jsonl(metrics_path)]
	traces: List[TraceRecord] = [TraceRecord(**x) for x in read_jsonl(traces_path)]
	incidents: List[IncidentRecord] = [IncidentRecord(**x) for x in read_jsonl(incidents_path)]
	return logs, metrics, traces, incidents

