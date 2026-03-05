#!/usr/bin/env python3
"""
Replay ground-truth telemetry files into OceanBase via MySQL protocol.
Uses shared extraction from loaders.common for identical data.
Environment: OCEANBASE_HOST (default: localhost), OCEANBASE_PORT (2881), OCEANBASE_USER (root)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

try:
    import pymysql
except ImportError:
    print("[replay] Install pymysql: pip install pymysql")
    sys.exit(1)

OB_HOST = os.environ.get("OCEANBASE_HOST", "127.0.0.1")
OB_PORT = int(os.environ.get("OCEANBASE_PORT", "2881"))
OB_USER = os.environ.get("OCEANBASE_USER", "root")
OB_PASS = os.environ.get("OCEANBASE_PASS", "")
OB_DB = "telemetry"


def _row_for_oceanbase(row: dict) -> tuple:
    """Convert row to tuple for INSERT. Serialize JSON dicts."""
    def _j(v):
        if isinstance(v, dict):
            return json.dumps(v)
        return v if v is not None else None

    return (
        row.get("ts"), row.get("service"), row.get("level"), row.get("message"),
        row.get("trace_id") or "", row.get("span_id") or "", _j(row.get("attrs")),
    )


def _span_row(row: dict) -> tuple:
    def _j(v):
        if isinstance(v, dict):
            return json.dumps(v)
        return v if v is not None else None
    return (
        row.get("ts_start"), row.get("trace_id") or "", row.get("ts_end"),
        row.get("span_id") or "", row.get("parent_span_id") or "", row.get("service") or "",
        row.get("name") or "", row.get("duration_ms"), _j(row.get("attributes")),
    )


def _metric_row(row: dict) -> tuple:
    def _j(v):
        if isinstance(v, dict):
            return json.dumps(v)
        return v if v is not None else None
    return (
        row.get("ts"), row.get("metric_name") or "", row.get("value"), _j(row.get("labels")),
    )


def insert_batch(conn, table: str, rows: list, row_fn) -> bool:
    if not rows:
        return True
    cols = {
        "logs": ("ts", "service", "level", "message", "trace_id", "span_id", "attrs"),
        "spans": ("ts_start", "trace_id", "ts_end", "span_id", "parent_span_id", "service", "name", "duration_ms", "attributes"),
        "metrics": ("ts", "metric_name", "value", "labels"),
    }[table]
    placeholders = ", ".join(["%s"] * len(cols))
    cols_str = ", ".join(cols)
    sql = f"INSERT INTO {OB_DB}.{table} ({cols_str}) VALUES ({placeholders})"
    try:
        with conn.cursor() as cur:
            cur.executemany(sql, [row_fn(r) for r in rows])
        conn.commit()
        return True
    except Exception as e:
        print(f"[oceanbase] {table}: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay telemetry into OceanBase")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=5000)
    ap.add_argument("--scale-to", type=int, default=None, help="Target row count per type (cycle data to reach it)")
    ap.add_argument("--stats", type=Path, default=None, help="Write ingestion stats JSON to path")
    args = ap.parse_args()
    data_dir = args.data_dir
    assert data_dir.exists(), f"DATA_DIR not found: {data_dir}"
    target = args.scale_to
    stats = {"logs": 0, "spans": 0, "metrics": 0}

    conn = pymysql.connect(
        host=OB_HOST, port=OB_PORT, user=OB_USER, password=OB_PASS or None,
        database=OB_DB, charset="utf8mb4",
    )
    try:
        for log_rows in common.extract_log_rows(data_dir, args.batch, target_rows=target):
            stats["logs"] += len(log_rows)
            if not insert_batch(conn, "logs", log_rows, _row_for_oceanbase):
                return 1

        for span_rows in common.extract_span_rows(data_dir, args.batch, target_rows=target):
            stats["spans"] += len(span_rows)
            if not insert_batch(conn, "spans", span_rows, _span_row):
                return 1

        for met_rows in common.extract_metric_rows(data_dir, args.batch, target_rows=target):
            stats["metrics"] += len(met_rows)
            if not insert_batch(conn, "metrics", met_rows, _metric_row):
                return 1
    finally:
        conn.close()

    if args.stats:
        args.stats.write_text(json.dumps(stats))
    print("[replay] oceanbase done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
