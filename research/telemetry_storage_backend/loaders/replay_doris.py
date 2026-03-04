#!/usr/bin/env python3
"""
Replay ground-truth telemetry files into Apache Doris using Stream Load.

Inputs:
- A directory with files named like:
  - logs_*.txt        (raw aggregated lines per window)
  - traces_*.json     (array of trace/span docs as exported from OpenSearch)
  - metrics_*.json    (structured metric snapshots per window)

Behavior:
- Creates JSONL batches and loads them into Doris tables (logs, spans, metrics).
- Keeps implementation intentionally simple and conservative to serve as
  a portable, backend-agnostic loader skeleton for the benchmark harness.

Environment:
- DORIS_FE_HTTP (default: http://localhost:8030)
- DORIS_BE_HTTP (default: http://localhost:8040) - use BE for stream load to avoid redirect stripping auth
- DORIS_DB      (default: telemetry)
- DORIS_USER    (default: root)
- DORIS_PASS    (default: empty)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import time
import requests
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

FE_HTTP = os.getenv("DORIS_FE_HTTP", "http://localhost:8030")
FE_INTERNAL_HTTP = os.getenv("DORIS_FE_INTERNAL_HTTP", "http://127.0.0.1:8030")
# Use BE for stream load: FE redirects to BE and strips Authorization on redirect
BE_HTTP = os.getenv("DORIS_BE_HTTP") or FE_HTTP.replace(":8030", ":8040").replace("8030", "8040")
DB = os.getenv("DORIS_DB", "telemetry")
USER = os.getenv("DORIS_USER", "root")
PASS = os.getenv("DORIS_PASS", "")
DORIS_CONTAINER = os.getenv("DORIS_CONTAINER", "tsb-doris")

def stream_load(table: str, file_path: Path, fmt: str = "json", columns: str | None = None, jsonpaths: str | None = "$", read_json_by_line: bool = True) -> bool:
    """
    Stream-load a file into Doris.

    Args:
      table: Target table name (e.g., 'logs', 'spans', 'metrics')
      file_path: Path to JSON/JSONL payload to upload
      fmt: 'json' (default) or other Doris-supported format
      columns: Optional explicit columns mapping
      jsonpaths: JSON path mapping (default '$')
      read_json_by_line: 'true' for JSONL (one JSON object per line)

    Returns:
      True if the FE accepted the load; False otherwise.
    """
    url = f"{BE_HTTP}/api/{DB}/{table}/_stream_load"
    headers = {
        "label": f"tsb_{table}_{int(time.time()*1000)}",
        "format": fmt,
        "Expect": "100-continue",
        "max_filter_ratio": "1",
    }
    if fmt == "json":
        headers["jsonpaths"] = jsonpaths or "$"
        headers["read_json_by_line"] = "true" if read_json_by_line else "false"
        headers["Content-Type"] = "application/json; charset=UTF-8"
    if columns:
        headers["columns"] = columns
    auth = (USER, PASS)
    try:
        with open(file_path, "rb") as f:
            r = requests.put(url, headers=headers, data=f, auth=auth, timeout=120)
        if r.status_code in (200, 201):
            try:
                body = r.json()
                if body.get("Status") == "Success":
                    print(f"[stream_load] {table} {file_path.name}: OK ({body.get('NumberLoadedRows', 0)} rows)")
                    return True
                if body.get("Status") == "Fail":
                    print(f"[stream_load] {table} {file_path.name}: {r.text.strip()[:200]}")
                    return False
            except Exception:
                pass
            print(f"[stream_load] {table} {file_path.name}: {r.text.strip()[:200]}")
        else:
            print(f"[stream_load] {table} {file_path.name} HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[stream_load] requests error for {table}: {e}")
    # Fallback to curl (handles Expect: 100-continue reliably)
    curl_cmd = [
        "curl", "-sSL", "--location-trusted", "-u", f"{USER}:{PASS}",
        "-H", f"label: tsb_{table}_{int(time.time()*1000)}",
        "-H", f"format: {fmt}",
        "-H", f"jsonpaths: {jsonpaths or '$'}",
        "-H", f"read_json_by_line: {'true' if read_json_by_line else 'false'}",
        "-H", "max_filter_ratio: 1",
        "-H", "Expect: 100-continue",
        "-T", str(file_path),
        url,
    ]
    try:
        out = subprocess.run(curl_cmd, check=True, capture_output=True, text=True)
        print(f"[stream_load][curl] {table} {file_path.name}: {out.stdout.strip()[:200]}")
        return True
    except subprocess.CalledProcessError:
        # Try running curl inside the Doris container to follow internal redirects to BE
        try:
            with open(file_path, "rb") as f:
                in_bytes = f.read()
            in_url = f"http://127.0.0.1:8040/api/{DB}/{table}/_stream_load"
            jp = jsonpaths or "$"
            exec_cmd = [
                "docker", "exec", "-i",
                "-e", f"USER={USER}", "-e", f"PASS={PASS}",
                "-e", f"LABEL=tsb_{table}_{int(time.time()*1000)}",
                "-e", f"URL={in_url}", "-e", f"JSONPATHS={jp}",
                DORIS_CONTAINER, "bash", "-lc",
                'curl -sSL --location-trusted -u "$USER:$PASS" -H "label: $LABEL" -H "format: json" '
                '-H "jsonpaths: $JSONPATHS" -H "read_json_by_line: true" -H "max_filter_ratio: 1" '
                '-H "Expect: 100-continue" -T - "$URL"'
            ]
            out2 = subprocess.run(exec_cmd, input=in_bytes, check=True, capture_output=True)
            print(f"[stream_load][exec-curl] {table} {file_path.name}: {out2.stdout.strip()[:200]}")
            return True
        except subprocess.CalledProcessError as ce2:
            print(f"[stream_load][exec-curl] FAILED {table}: {ce2.stdout or ce2.stderr}")
            return False

def rows_to_jsonl(rows):
    """
    Convert a list of dict rows into JSONL (bytes).
    """
    return ("\n".join(json.dumps(r) for r in rows)).encode()

def main() -> int:
    """
    CLI entrypoint: bulk-replay logs, traces, metrics into Doris using Stream Load.

    Flags:
      --data-dir: path to 'otel_ground_truth_data'
      --table-prefix: (unused placeholder for future)
      --batch: max rows per load request
    """
    ap = argparse.ArgumentParser(description="Replay otel_ground_truth_data into Doris via Stream Load")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--table-prefix", type=str, default="")
    ap.add_argument("--batch", type=int, default=5000)
    ap.add_argument("--scale-to", type=int, default=None, help="Target row count per type (cycle data to reach it)")
    ap.add_argument("--stats", type=Path, default=None, help="Write ingestion stats JSON to path")
    args = ap.parse_args()
    data_dir: Path = args.data_dir
    assert data_dir.exists(), f"DATA_DIR not found: {data_dir}"
    target = args.scale_to
    stats = {"logs": 0, "spans": 0, "metrics": 0}

    # Logs (shared extraction from common)
    logs_tmp = Path(".replay_logs.jsonl")
    for log_rows in common.extract_log_rows(data_dir, args.batch, target_rows=target):
        stats["logs"] += len(log_rows)
        logs_tmp.write_bytes(rows_to_jsonl(log_rows))
        jp = json.dumps(["$.ts","$.service","$.level","$.message","$.trace_id","$.span_id","$.attrs"])
        if not stream_load("logs", logs_tmp, jsonpaths=jp, read_json_by_line=True):
            return 1
    if logs_tmp.exists():
        logs_tmp.unlink(missing_ok=True)

    # Traces (shared extraction with ISO datetime conversion)
    traces_tmp = Path(".replay_spans.jsonl")
    for span_rows in common.extract_span_rows(data_dir, args.batch, target_rows=target):
        stats["spans"] += len(span_rows)
        traces_tmp.write_bytes(rows_to_jsonl(span_rows))
        jp = json.dumps(["$.ts_start","$.trace_id","$.ts_end","$.span_id","$.parent_span_id","$.service","$.name","$.duration_ms","$.attributes"])
        if not stream_load("spans", traces_tmp, jsonpaths=jp, read_json_by_line=True):
            return 1
    if traces_tmp.exists():
        traces_tmp.unlink(missing_ok=True)

    # Metrics (shared extraction)
    metrics_tmp = Path(".replay_metrics.jsonl")
    for met_rows in common.extract_metric_rows(data_dir, args.batch, target_rows=target):
        stats["metrics"] += len(met_rows)
        metrics_tmp.write_bytes(rows_to_jsonl(met_rows))
        jp = json.dumps(["$.ts","$.metric_name","$.value","$.labels"])
        if not stream_load("metrics", metrics_tmp, jsonpaths=jp, read_json_by_line=True):
            return 1
    if metrics_tmp.exists():
        metrics_tmp.unlink(missing_ok=True)

    if args.stats:
        args.stats.write_text(json.dumps(stats))
    print("[replay] done")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
