#!/usr/bin/env python3
"""
Replay ground-truth telemetry files into ClickHouse via HTTP.
Uses shared extraction from loaders.common for identical data.
Environment: CLICKHOUSE_HTTP (default: http://localhost:8123), CLICKHOUSE_PASSWORD (optional)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

CH_HTTP = os.environ.get("CLICKHOUSE_HTTP", "http://localhost:8123")
CH_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD", "")
DB = "telemetry"

def _ch_params(extra: dict | None = None) -> dict:
    params = dict(extra) if extra else {}
    if CH_PASSWORD:
        params["user"] = "default"
        params["password"] = CH_PASSWORD
    return params

def _row_for_clickhouse(row: dict) -> dict:
    """Convert row to ClickHouse format (attrs/attributes as JSON strings)."""
    out = dict(row)
    if isinstance(out.get("attrs"), dict):
        out["attrs"] = json.dumps(out["attrs"])
    if isinstance(out.get("attributes"), dict):
        out["attributes"] = json.dumps(out["attributes"])
    if isinstance(out.get("labels"), dict):
        out["labels"] = json.dumps(out["labels"])
    return out

def insert_jsonl(table: str, rows: list) -> bool:
    url = f"{CH_HTTP}/"
    ch_rows = [_row_for_clickhouse(r) for r in rows]
    body = "\n".join(json.dumps(r) for r in ch_rows)
    query = f"INSERT INTO {DB}.{table} FORMAT JSONEachRow"
    params = _ch_params({"query": query})
    try:
        r = requests.post(url, params=params, data=body.encode(), timeout=120)
        if r.status_code == 200:
            return True
        print(f"[clickhouse] {table} HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[clickhouse] {table}: {e}")
    return False

def main() -> int:
    ap = argparse.ArgumentParser(description="Replay telemetry into ClickHouse")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=5000)
    args = ap.parse_args()
    data_dir = args.data_dir
    assert data_dir.exists(), f"DATA_DIR not found: {data_dir}"

    for log_rows in common.extract_log_rows(data_dir, args.batch):
        if not insert_jsonl("logs", log_rows):
            return 1

    for span_rows in common.extract_span_rows(data_dir, args.batch):
        if not insert_jsonl("spans", span_rows):
            return 1

    for met_rows in common.extract_metric_rows(data_dir, args.batch):
        if not insert_jsonl("metrics", met_rows):
            return 1

    print("[replay] clickhouse done")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
