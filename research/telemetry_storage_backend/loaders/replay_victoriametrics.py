#!/usr/bin/env python3
"""
Replay telemetry files into VictoriaMetrics (metrics), VictoriaLogs (logs),
and VictoriaTraces (traces via OTLP HTTP).
Uses shared extraction from loaders.common.
Environment:
  VM_HTTP  (default: http://localhost:8428) — VictoriaMetrics
  VL_HTTP  (default: http://localhost:9428) — VictoriaLogs
  VT_HTTP  (default: http://localhost:10428) — VictoriaTraces (OTLP via main HTTP)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import remote_write

VM_HTTP = os.environ.get("VM_HTTP", "http://localhost:8428")
VL_HTTP = os.environ.get("VL_HTTP", "http://localhost:9428")
VL_PUSH_URL = f"{VL_HTTP}/insert/loki/api/v1/push"
VT_HTTP = os.environ.get("VT_HTTP", "http://localhost:10428")
PUSH_CHUNK_SIZE = 5


def push_metrics(rows: list[dict]) -> bool:
    """Push metric rows to VictoriaMetrics via /api/v1/write (Prometheus remote write)."""
    url = f"{VM_HTTP}/api/v1/write"
    timeseries = []
    for row in rows:
        ts_epoch_ms = _ts_to_epoch_ms(row.get("ts", ""))
        labels = {"__name__": row.get("metric_name", "unknown")}
        row_labels = row.get("labels", {})
        if isinstance(row_labels, str):
            try:
                row_labels = json.loads(row_labels)
            except Exception:
                row_labels = {}
        for k, v in row_labels.items():
            if k and v:
                labels[k] = str(v)
        timeseries.append({
            "labels": labels,
            "value": float(row.get("value", 0)),
            "timestamp_ms": ts_epoch_ms,
        })
    body = remote_write.encode_write_request(timeseries)
    try:
        r = requests.post(url, data=body,
                          headers={"Content-Type": "application/x-protobuf",
                                   "Content-Encoding": "snappy",
                                   "X-Prometheus-Remote-Write-Version": "0.1.0"},
                          timeout=60)
        if r.status_code in (200, 204):
            return True
        print(f"[vm] metrics push HTTP {r.status_code}: {r.text[:300]}")
    except Exception as e:
        print(f"[vm] metrics push error: {e}")
    return False


def _ts_to_ns(ts: str) -> str:
    """Convert SQL-style timestamp to nanoseconds since epoch (same as Loki loader)."""
    try:
        if "." in ts:
            dt = datetime.strptime(ts[:26], "%Y-%m-%d %H:%M:%S.%f")
        else:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        return str(int(dt.timestamp() * 1_000_000_000))
    except Exception:
        return str(int(time.time() * 1_000_000_000))


def _row_to_stream(row: dict, run_id: str = "") -> dict:
    """Convert one log row to a Loki stream (same format as replay_loki.py)."""
    service = (row.get("service") or "unknown").replace('"', '\\"')[:64]
    level = (row.get("level") or "info").replace('"', '\\"')[:32]
    stream = {"job": "telemetry", "service": service or "unknown", "level": level or "info"}
    if run_id:
        stream["run_id"] = run_id
    if row.get("trace_id"):
        stream["trace_id"] = str(row["trace_id"])[:64]
    if row.get("span_id"):
        stream["span_id"] = str(row["span_id"])[:64]
    ts_ns = _ts_to_ns(row.get("ts", ""))
    line = (row.get("message") or "").replace("\n", " ").replace("\r", "")
    if len(line) > 200_000:
        line = line[:200_000]
    return {"stream": stream, "values": [[ts_ns, line]]}


def push_logs(rows: list[dict], run_id: str = "") -> bool:
    """Push log rows to VictoriaLogs via Loki-compatible push API."""
    streams = [_row_to_stream(r, run_id) for r in rows]
    for i in range(0, len(streams), PUSH_CHUNK_SIZE):
        chunk = streams[i : i + PUSH_CHUNK_SIZE]
        payload = {"streams": chunk}
        try:
            r = requests.post(VL_PUSH_URL, json=payload,
                              headers={"Content-Type": "application/json"}, timeout=60)
            if r.status_code not in (200, 204):
                print(f"[vl] logs push HTTP {r.status_code}: {r.text[:300]}")
                return False
        except Exception as e:
            print(f"[vl] logs push error: {e}")
            return False
    return True


def push_traces(rows: list[dict]) -> bool:
    """Push span rows to VictoriaTraces via OTLP HTTP JSON (/v1/traces)."""
    url = f"{VT_HTTP}/insert/opentelemetry/v1/traces"
    spans_by_service: dict[str, list] = {}
    for row in rows:
        svc = row.get("service", "unknown")
        spans_by_service.setdefault(svc, []).append(row)

    resource_spans = []
    for svc, svc_rows in spans_by_service.items():
        otlp_spans = []
        for row in svc_rows:
            start_ns = _ts_to_epoch_ns(row.get("ts_start", ""))
            end_ns = _ts_to_epoch_ns(row.get("ts_end", ""))
            attrs = row.get("attributes", {})
            if isinstance(attrs, str):
                try:
                    attrs = json.loads(attrs)
                except Exception:
                    attrs = {}
            otlp_attrs = [{"key": k, "value": {"stringValue": str(v)}} for k, v in attrs.items()]
            otlp_spans.append({
                "traceId": _hex_pad(row.get("trace_id", ""), 32),
                "spanId": _hex_pad(row.get("span_id", ""), 16),
                "parentSpanId": _hex_pad(row.get("parent_span_id", ""), 16) if row.get("parent_span_id") else "",
                "name": row.get("name", ""),
                "kind": 1,
                "startTimeUnixNano": str(start_ns),
                "endTimeUnixNano": str(end_ns),
                "attributes": otlp_attrs,
                "status": {},
            })
        resource_spans.append({
            "resource": {
                "attributes": [{"key": "service.name", "value": {"stringValue": svc}}],
            },
            "scopeSpans": [{"scope": {"name": "telemetry-bench"}, "spans": otlp_spans}],
        })

    payload = {"resourceSpans": resource_spans}
    for attempt in range(5):
        try:
            r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
            if r.status_code in (200, 202):
                return True
            print(f"[vt] traces push HTTP {r.status_code}: {r.text[:300]}")
        except (requests.exceptions.ConnectionError, ConnectionResetError) as e:
            if attempt < 4:
                print(f"[vt] traces push retry {attempt + 1}/5 ({e})")
                time.sleep(3)
                continue
            print(f"[vt] traces push error after retries: {e}")
        except Exception as e:
            print(f"[vt] traces push error: {e}")
            break
    return False


def _ts_to_epoch_ms(ts: str) -> int:
    try:
        if "." in ts:
            from datetime import datetime
            dt = datetime.strptime(ts[:26], "%Y-%m-%d %H:%M:%S.%f")
            return int(dt.timestamp() * 1000)
        elif ts:
            from datetime import datetime
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            return int(dt.timestamp() * 1000)
    except Exception:
        pass
    return int(time.time() * 1000)


def _ts_to_epoch_ns(ts: str) -> int:
    return _ts_to_epoch_ms(ts) * 1_000_000


def _hex_pad(val: str, length: int) -> str:
    """Ensure hex string is exactly `length` chars, zero-padded or truncated."""
    if not val:
        return "0" * length
    cleaned = val.replace("-", "")
    if len(cleaned) < length:
        cleaned = cleaned.zfill(length)
    return cleaned[:length]


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay telemetry into VictoriaMetrics/Logs/Traces")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--scale-to", type=int, default=None, help="Target row count per type")
    ap.add_argument("--stats", type=Path, default=None, help="Write ingestion stats JSON")
    ap.add_argument("--run-id", type=str, default="", help="Label to isolate this run in VictoriaLogs")
    args = ap.parse_args()
    data_dir = args.data_dir
    assert data_dir.exists(), f"DATA_DIR not found: {data_dir}"
    target = args.scale_to
    stats = {"logs": 0, "spans": 0, "metrics": 0,
             "logs_duration_s": 0, "spans_duration_s": 0, "metrics_duration_s": 0}

    t0 = time.time()
    for log_rows in common.extract_log_rows(data_dir, args.batch, target_rows=target):
        if not push_logs(log_rows, run_id=args.run_id):
            return 1
        stats["logs"] += len(log_rows)
    stats["logs_duration_s"] = round(time.time() - t0, 3)

    t0 = time.time()
    for span_rows in common.extract_span_rows(data_dir, args.batch, target_rows=target):
        if not push_traces(span_rows):
            return 1
        stats["spans"] += len(span_rows)
    stats["spans_duration_s"] = round(time.time() - t0, 3)

    t0 = time.time()
    for met_rows in common.extract_metric_rows(data_dir, args.batch, target_rows=target):
        if not push_metrics(met_rows):
            return 1
        stats["metrics"] += len(met_rows)
    stats["metrics_duration_s"] = round(time.time() - t0, 3)

    if args.stats:
        args.stats.write_text(json.dumps(stats))
    print("[replay] victoriametrics done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
