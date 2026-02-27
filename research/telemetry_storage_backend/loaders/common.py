#!/usr/bin/env python3
"""
Shared data extraction logic for Doris and ClickHouse loaders.
Ensures identical processing of the same static dataset.
"""
from __future__ import annotations
import json
import time
from pathlib import Path


def iso_to_sql_datetime(iso_str: str | None) -> str:
    """
    Convert ISO 8601 timestamp (e.g. 2026-02-07T05:02:29.326473093Z) to SQL-friendly format.
    ClickHouse and Doris accept: YYYY-MM-DD HH:MM:SS or YYYY-MM-DDTHH:MM:SS (without Z).
    """
    if not iso_str or not isinstance(iso_str, str):
        return time.strftime("%Y-%m-%d %H:%M:%S")
    # Strip Z and replace T with space for maximum compatibility
    s = iso_str.rstrip("Z").replace("T", " ")
    # Truncate fractional seconds to 6 digits if needed
    if "." in s:
        base, frac = s.split(".", 1)
        frac = frac[:6].ljust(6, "0")[:6]
        return f"{base}.{frac}"
    return s


def extract_log_rows(data_dir: Path, batch: int):
    """Yield log rows in batches. Same logic for both backends."""
    log_files = sorted(data_dir.rglob("logs_*.txt")) or sorted(data_dir.glob("logs_*.txt"))
    rows = []
    for lf in log_files:
        try:
            raw = lf.read_text(errors="ignore")
            rows.append({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "service": "unknown",
                "level": "info",
                "message": raw[:800000],
                "trace_id": "",
                "span_id": "",
                "attrs": {},
            })
            if len(rows) >= batch:
                yield rows
                rows = []
        except Exception:
            continue
    if rows:
        yield rows


def extract_span_rows(data_dir: Path, batch: int, max_per_file: int = 200):
    """Yield span rows in batches. Same logic for both backends."""
    trace_files = sorted(data_dir.rglob("traces_*.json")) or sorted(data_dir.glob("traces_*.json"))
    rows = []
    for tf in trace_files:
        try:
            arr = json.loads(tf.read_text())
        except Exception:
            continue
        for hit in arr[:max_per_file]:
            src = hit.get("_source", {})
            dur = src.get("duration", 0)
            duration_ms = int(dur) // 1_000_000 if isinstance(dur, (int, float)) else 0
            rows.append({
                "ts_start": iso_to_sql_datetime(src.get("startTime")),
                "ts_end": iso_to_sql_datetime(src.get("endTime")),
                "trace_id": src.get("traceId", ""),
                "span_id": src.get("spanId", ""),
                "parent_span_id": src.get("parentSpanId", ""),
                "service": src.get("resource", {}).get("service.name", ""),
                "name": src.get("name", ""),
                "duration_ms": duration_ms,
                "attributes": src.get("attributes", {}),
            })
            if len(rows) >= batch:
                yield rows
                rows = []
    if rows:
        yield rows


def extract_metric_rows(data_dir: Path, batch: int):
    """Yield metric rows in batches. Same logic for both backends."""
    metric_files = sorted(data_dir.rglob("metrics_*.json")) or sorted(data_dir.glob("metrics_*.json"))
    rows = []
    for mf in metric_files:
        try:
            doc = json.loads(mf.read_text())
        except Exception:
            continue
        rows.append({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metric_name": "req_rate",
            "value": float(len(doc.get("req_rate", []))) if isinstance(doc.get("req_rate"), list) else 0.0,
            "labels": {},
        })
        if len(rows) >= batch:
            yield rows
            rows = []
    if rows:
        yield rows
