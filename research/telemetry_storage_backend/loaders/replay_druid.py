#!/usr/bin/env python3
"""
Replay ground-truth telemetry files into Apache Druid via native batch ingestion.
Uses shared extraction from loaders.common for identical data.
Writes JSONL to .druid_load/ (mounted in middlemanager), submits ingestion tasks.
Environment: DRUID_OVERLORD_HTTP (default: http://localhost:8081)
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

OVERLORD_HTTP = os.environ.get("DRUID_OVERLORD_HTTP", "http://localhost:8081")
LOAD_DIR = Path(".druid_load")
INTERVAL = "2020-01-01/2030-01-01"  # Wide interval for our data


def _row_for_druid(row: dict) -> dict:
    """Convert row to Druid format (attrs/attributes/labels as JSON strings)."""
    out = dict(row)
    if isinstance(out.get("attrs"), dict):
        out["attrs"] = json.dumps(out["attrs"])
    if isinstance(out.get("attributes"), dict):
        out["attributes"] = json.dumps(out["attributes"])
    if isinstance(out.get("labels"), dict):
        out["labels"] = json.dumps(out["labels"])
    return out


def _submit_task(spec: dict) -> str | None:
    """Submit ingestion task, return task_id or None."""
    try:
        r = requests.post(
            f"{OVERLORD_HTTP}/druid/indexer/v1/task",
            json=spec,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        if r.status_code in (200, 201):
            tid = r.json().get("task")
            return tid
        print(f"[druid] task submit {r.status_code}: {r.text[:300]}")
    except Exception as e:
        print(f"[druid] task submit error: {e}")
    return None


def _wait_task(task_id: str, timeout_s: int = 600) -> bool:
    """Wait for task to complete. Returns True if success."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(
                f"{OVERLORD_HTTP}/druid/indexer/v1/task/{task_id}/status",
                timeout=10,
            )
            if r.status_code != 200:
                time.sleep(5)
                continue
            st = r.json().get("status", {}).get("status")
            if st == "SUCCESS":
                return True
            if st in ("FAILED", "CANCELED"):
                print(f"[druid] task {task_id} {st}: {r.text[:500]}")
                return False
        except Exception as e:
            print(f"[druid] task status error: {e}")
        time.sleep(5)
    print(f"[druid] task {task_id} timeout")
    return False


def _ingest_datasource(
    data_source: str,
    timestamp_col: str,
    dimensions: list[str],
    metrics_spec: list[dict],
    filter_glob: str,
) -> bool:
    """Run ingestion for one datasource."""
    spec = {
        "type": "index_parallel",
        "spec": {
            "dataSchema": {
                "dataSource": data_source,
                "timestampSpec": {
                    "column": timestamp_col,
                    "format": "iso" if "T" in timestamp_col else "yyyy-MM-dd HH:mm:ss",
                },
                "dimensionsSpec": {
                    "dimensions": dimensions,
                },
                "metricsSpec": metrics_spec,
                "granularitySpec": {
                    "segmentGranularity": "DAY",
                    "queryGranularity": "SECOND",
                    "intervals": [INTERVAL],
                },
            },
            "ioConfig": {
                "type": "index_parallel",
                "inputSource": {
                    "type": "local",
                    "baseDir": f"/mnt/bench/{LOAD_DIR}",
                    "filter": filter_glob,
                },
                "inputFormat": {"type": "json"},
                "dropExisting": False,
            },
            "tuningConfig": {
                "type": "index_parallel",
                "maxNumConcurrentSubTasks": 2,
            },
        },
    }
    # Support fractional seconds (e.g. 2026-02-07 06:15:17.395775); format without .SSSSSS drops most rows
    spec["spec"]["dataSchema"]["timestampSpec"]["format"] = "yyyy-MM-dd HH:mm:ss.SSSSSS"

    task_id = _submit_task(spec)
    if not task_id:
        return False
    return _wait_task(task_id)


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay telemetry into Druid")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=5000)
    ap.add_argument("--scale-to", type=int, default=None, help="Target row count per type (cycle data to reach it)")
    ap.add_argument("--stats", type=Path, default=None, help="Write ingestion stats JSON to path")
    args = ap.parse_args()
    data_dir = args.data_dir
    assert data_dir.exists(), f"DATA_DIR not found: {data_dir}"
    target = args.scale_to
    stats = {"logs": 0, "spans": 0, "metrics": 0}

    LOAD_DIR.mkdir(exist_ok=True)
    for f in LOAD_DIR.glob("*.jsonl"):
        f.unlink()

    # Logs
    logs_path = LOAD_DIR / "logs_batch.jsonl"
    with open(logs_path, "w") as out:
        for rows in common.extract_log_rows(data_dir, args.batch, target_rows=target):
            stats["logs"] += len(rows)
            for row in rows:
                out.write(json.dumps(_row_for_druid(row)) + "\n")
    if logs_path.stat().st_size > 0:
        if not _ingest_datasource(
            "telemetry_logs",
            "ts",
            ["service", "level", "message", "trace_id", "span_id", "attrs"],
            [{"type": "count", "name": "count"}],
            "logs_*.jsonl",
        ):
            return 1

    # Spans
    spans_path = LOAD_DIR / "spans_batch.jsonl"
    with open(spans_path, "w") as out:
        for rows in common.extract_span_rows(data_dir, args.batch, target_rows=target):
            stats["spans"] += len(rows)
            for row in rows:
                out.write(json.dumps(_row_for_druid(row)) + "\n")
    if spans_path.stat().st_size > 0:
        if not _ingest_datasource(
            "telemetry_spans",
            "ts_start",
            ["trace_id", "span_id", "parent_span_id", "service", "name", "attributes"],
            [{"type": "count", "name": "count"}, {"type": "longSum", "name": "duration_ms", "fieldName": "duration_ms"}],
            "spans_*.jsonl",
        ):
            return 1

    # Metrics
    metrics_path = LOAD_DIR / "metrics_batch.jsonl"
    with open(metrics_path, "w") as out:
        for rows in common.extract_metric_rows(data_dir, args.batch, target_rows=target):
            stats["metrics"] += len(rows)
            for row in rows:
                out.write(json.dumps(_row_for_druid(row)) + "\n")
    if metrics_path.stat().st_size > 0:
        if not _ingest_datasource(
            "telemetry_metrics",
            "ts",
            ["metric_name", "labels", "trace_id"],
            [{"type": "count", "name": "count"}, {"type": "doubleSum", "name": "value", "fieldName": "value"}],
            "metrics_*.jsonl",
        ):
            return 1

    # Cleanup
    for f in LOAD_DIR.glob("*.jsonl"):
        f.unlink()

    if args.stats:
        args.stats.write_text(json.dumps(stats))

    # Ensure segments are readable by Historical (fixes permission issues in shared volume)
    subprocess.run(
        ["docker", "run", "--rm", "-v", "telemetry_storage_backend_druid_shared:/opt/shared",
         "alpine", "chmod", "-R", "a+rX", "/opt/shared/segments"],
        capture_output=True,
        timeout=30,
    )

    print("[replay] druid done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
