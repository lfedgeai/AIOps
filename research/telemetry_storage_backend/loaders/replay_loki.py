#!/usr/bin/env python3
"""
Replay log files into Grafana Loki via HTTP push API (logs only).
Uses shared extraction from loaders.common.
Environment: LOKI_HTTP (default: http://localhost:3100)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import time
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

LOKI_HTTP = os.environ.get("LOKI_HTTP", "http://localhost:3100")
PUSH_URL = f"{LOKI_HTTP}/loki/api/v1/push"
# Loki gRPC max message size is 4MB; keep each push under ~3MB to avoid "message larger than max"
PUSH_CHUNK_SIZE = 5


def _ts_to_ns(ts: str) -> str:
    """Convert SQL-style timestamp to nanoseconds since epoch."""
    try:
        if "." in ts:
            dt = datetime.strptime(ts[:26], "%Y-%m-%d %H:%M:%S.%f")
        else:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        return str(int(dt.timestamp() * 1_000_000_000))
    except Exception:
        import time
        return str(int(time.time() * 1_000_000_000))


def _row_to_stream(row: dict, run_id: str = "") -> dict:
    """Convert one log row to a Loki stream (one stream per row for label uniqueness)."""
    service = (row.get("service") or "unknown").replace('"', '\\"')[:64]
    level = (row.get("level") or "info").replace('"', '\\"')[:32]
    # Loki labels must be valid; avoid empty
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


def push_batch(streams: list[dict]) -> bool:
    """Push a batch of streams to Loki."""
    payload = {"streams": streams}
    try:
        r = requests.post(PUSH_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
        if r.status_code in (200, 204):
            return True
        print(f"[loki] push HTTP {r.status_code}: {r.text[:300]}")
    except Exception as e:
        print(f"[loki] push error: {e}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay logs into Loki (logs only)")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--scale-to", type=int, default=None, help="Target row count")
    ap.add_argument("--stats", type=Path, default=None)
    ap.add_argument("--run-id", type=str, default="", help="Label to isolate this run's data in Loki")
    args = ap.parse_args()
    data_dir = args.data_dir
    assert data_dir.exists(), f"DATA_DIR not found: {data_dir}"
    target = args.scale_to
    stats = {"logs": 0}

    emitted = 0
    run_id = getattr(args, "run_id", "") or ""
    for log_rows in common.extract_log_rows(data_dir, args.batch, target_rows=target):
        streams = [_row_to_stream(r, run_id) for r in log_rows]
        # Push in small chunks to stay under Loki gRPC 4MB limit (log lines can be 200KB each)
        for i in range(0, len(streams), PUSH_CHUNK_SIZE):
            chunk = streams[i : i + PUSH_CHUNK_SIZE]
            if not push_batch(chunk):
                return 1
            time.sleep(0.2)
        stats["logs"] += len(log_rows)
        emitted += len(log_rows)
        if target and emitted >= target:
            break

    if args.stats:
        args.stats.write_text(json.dumps(stats))
    print("[replay] loki done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
