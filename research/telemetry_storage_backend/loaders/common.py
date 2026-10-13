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


def _trace_file_for_log(log_path: Path, data_dir: Path) -> Path | None:
    """Return traces file path for a logs file, e.g. logs/test/logs_X.txt -> traces/test/traces_X.json."""
    try:
        rel = log_path.relative_to(data_dir)
    except ValueError:
        return None
    if len(rel.parts) < 2 or rel.parts[0] != "logs":
        return None
    suffix = log_path.stem[5:] if log_path.stem.startswith("logs_") else log_path.stem
    trace_path = data_dir / "traces" / rel.parts[1] / f"traces_{suffix}.json"
    return trace_path if trace_path.exists() else None


def _trace_file_for_metric(metric_path: Path, data_dir: Path) -> Path | None:
    """Return traces file path for a metrics file, e.g. metrics/eval/metrics_X.json -> traces/eval/traces_X.json."""
    try:
        rel = metric_path.relative_to(data_dir)
    except ValueError:
        return None
    if len(rel.parts) < 2:
        return None
    # Support both metrics/ and metadata/ folders (metadata often mirrors metrics)
    if rel.parts[0] not in ("metrics", "metadata"):
        return None
    suffix = metric_path.stem[8:] if metric_path.stem.startswith("metrics_") else metric_path.stem
    trace_path = data_dir / "traces" / rel.parts[1] / f"traces_{suffix}.json"
    return trace_path if trace_path.exists() else None


def extract_log_rows(data_dir: Path, batch: int, target_rows: int | None = None):
    """Yield log rows in batches. Same logic for both backends.
    Correlates with traces: logs_X ↔ traces_X get same trace_id, span_id, service.
    If target_rows is set, cycle through files until at least target_rows are emitted."""
    log_files = sorted(data_dir.rglob("logs_*.txt")) or sorted(data_dir.glob("logs_*.txt"))
    emitted = 0
    while True:
        rows = []
        for lf in log_files:
            try:
                raw = lf.read_text(errors="ignore")
                trace_id, span_id, service = "", "", "unknown"
                tf = _trace_file_for_log(lf, data_dir)
                if tf:
                    try:
                        arr = json.loads(tf.read_text())
                        if arr:
                            src = arr[0].get("_source", {})
                            trace_id = src.get("traceId", "")
                            span_id = src.get("spanId", "")
                            service = src.get("resource", {}).get("service.name", "")
                    except Exception:
                        pass
                rows.append({
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "service": service,
                    "level": "info",
                    "message": raw[:200000],  # 200KB max for benchmark (avoid huge Druid files)
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "attrs": {},
                })
                if len(rows) >= batch:
                    yield rows
                    emitted += len(rows)
                    rows = []
                    if target_rows and emitted >= target_rows:
                        return
            except Exception:
                continue
        if rows:
            yield rows
            emitted += len(rows)
        if not target_rows or emitted >= target_rows:
            break


def extract_span_rows(data_dir: Path, batch: int, max_per_file: int = 200, target_rows: int | None = None):
    """Yield span rows in batches. Same logic for both backends.
    If target_rows is set, cycle through files until at least target_rows are emitted."""
    trace_files = sorted(data_dir.rglob("traces_*.json")) or sorted(data_dir.glob("traces_*.json"))
    emitted = 0
    while True:
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
                    emitted += len(rows)
                    rows = []
                    if target_rows and emitted >= target_rows:
                        return
        if rows:
            yield rows
            emitted += len(rows)
        if not target_rows or emitted >= target_rows:
            break


def extract_metric_rows(data_dir: Path, batch: int, target_rows: int | None = None):
    """Yield metric rows in batches. Same logic for both backends.
    Correlates with traces: explodes req_rate/cart_add_p95 into rows with service in labels.
    Adds trace_id to labels when a matching traces file exists (metrics_X <-> traces_X).
    If target_rows is set, cycle through files until at least target_rows are emitted."""
    metric_files = sorted(data_dir.rglob("metrics_*.json")) or sorted(data_dir.glob("metrics_*.json"))
    emitted = 0
    while True:
        rows = []
        for mf in metric_files:
            try:
                doc = json.loads(mf.read_text())
            except Exception:
                continue
            base_ts = time.strftime("%Y-%m-%d %H:%M:%S")
            trace_id = ""
            tf = _trace_file_for_metric(mf, data_dir)
            if tf:
                try:
                    arr = json.loads(tf.read_text())
                    if arr:
                        trace_id = arr[0].get("_source", {}).get("traceId", "")
                except Exception:
                    pass
            added = False
            for series_name, entries in [("req_rate", doc.get("req_rate", [])), ("cart_add_p95", doc.get("cart_add_p95", []))]:
                if not isinstance(entries, list):
                    continue
                for ent in entries:
                    if not isinstance(ent, dict):
                        continue
                    metric = ent.get("metric", {})
                    value_arr = ent.get("value", [0, 0])
                    val = float(value_arr[1]) if len(value_arr) > 1 else 0.0
                    service = metric.get("service_name", "")
                    labels = {"service": service} if service else {}
                    if trace_id:
                        labels["trace_id"] = trace_id
                    row = {"ts": base_ts, "metric_name": series_name, "value": val, "labels": labels, "trace_id": trace_id}
                    rows.append(row)
                    added = True
                    if len(rows) >= batch:
                        yield rows
                        emitted += len(rows)
                        rows = []
                        if target_rows and emitted >= target_rows:
                            return
            if not added:
                labels = {"trace_id": trace_id} if trace_id else {}
                rows.append({
                    "ts": base_ts,
                    "metric_name": "req_rate",
                    "value": float(len(doc.get("req_rate", []))) if isinstance(doc.get("req_rate"), list) else 0.0,
                    "labels": labels,
                    "trace_id": trace_id,
                })
        if rows:
            yield rows
            emitted += len(rows)
        if not target_rows or emitted >= target_rows:
            break
