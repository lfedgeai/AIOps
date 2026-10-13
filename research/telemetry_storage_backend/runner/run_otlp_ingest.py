#!/usr/bin/env python3
"""
Run OTLP ingestion: telemetrygen -> collector -> Doris, ClickHouse.
Sends N spans, N logs, N metrics to the collector (default 1000 each).
Returns JSON with duration_s, rows, rows_per_sec per backend.
"""
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
OTEL_ENDPOINT = "tsb-otel-collector:4317"
CH_HTTP = "http://localhost:8123"
CH_PASSWORD = __import__("os").environ.get("CLICKHOUSE_PASSWORD", "changeme")

def _ch_params():
    p = {"query": ""}
    if CH_PASSWORD:
        p["user"] = "default"
        p["password"] = CH_PASSWORD
    return p

def count_clickhouse_otel():
    """Count rows in otel.otel_logs, otel.otel_traces, otel_metrics_*."""
    total = 0
    for table in ["otel_logs", "otel_traces", "otel_metrics_gauge", "otel_metrics_sum", "otel_metrics_histogram"]:
        try:
            r = requests.post(CH_HTTP, params=_ch_params(), data=f"SELECT count() FROM otel.{table}", timeout=10)
            if r.status_code == 200:
                total += int(r.text.strip() or 0)
        except Exception:
            pass
    return total

def count_doris_otel():
    """Count rows in otel.otel_logs, otel.otel_traces, otel.otel_metrics."""
    total = 0
    for table in ["otel_logs", "otel_traces", "otel_metrics"]:
        try:
            out = subprocess.run(
                ["docker", "run", "--rm", "--network", "tsb-net", "mysql:8", "sh", "-lc",
                 f"mysql -h tsb-doris -P 9030 -uroot -N -B -D otel -e 'SELECT COUNT(*) FROM {table}'"],
                capture_output=True, text=True, timeout=15,
            )
            if out.returncode == 0 and out.stdout.strip().isdigit():
                total += int(out.stdout.strip())
        except Exception:
            pass
    return total

def run_telemetrygen(count: int, rate: int = 1000):
    """Run telemetrygen to send count traces, count logs, count metrics."""
    # --rate N = N/sec; 50k at 1000/sec = ~50s
    for sig, flag in [("traces", "traces"), ("logs", "logs"), ("metrics", "metrics")]:
        cmd = [
            "docker", "run", "--rm", "--network", "tsb-net",
            "ghcr.io/open-telemetry/opentelemetry-collector-contrib/telemetrygen:latest",
            sig, f"--{flag}", str(count), "--rate", str(rate),
            "--otlp-endpoint", OTEL_ENDPOINT, "--otlp-insecure",
        ]
        subprocess.run(cmd, check=True, timeout=max(120, count // rate + 60))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, help="Write stats JSON here")
    ap.add_argument("--count", type=int, default=1000, help="Spans, logs, metrics each (default 1000)")
    ap.add_argument("--rate", type=int, default=1000, help="Items per second (default 1000)")
    args = ap.parse_args()

    t0 = time.time()
    run_telemetrygen(args.count, args.rate)
    elapsed = time.time() - t0
    # Allow collector a moment to flush
    time.sleep(3)

    result = {
        "mechanism": f"OTLP ({args.count} spans, {args.count} logs, {args.count} metrics)",
        "duration_s": round(elapsed, 2),
        "doris": {"rows": count_doris_otel()},
        "clickhouse": {"rows": count_clickhouse_otel()},
    }
    for k in ["doris", "clickhouse"]:
        r = result[k]["rows"]
        result[k]["rows_per_sec"] = round(r / result["duration_s"], 0) if result["duration_s"] > 0 else 0

    if args.stats:
        args.stats.write_text(json.dumps(result))
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
