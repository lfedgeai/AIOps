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
CH_HTTP = __import__("os").environ.get("CLICKHOUSE_HTTP", "http://localhost:8123")
CH_PASSWORD = __import__("os").environ.get("CLICKHOUSE_PASSWORD", "changeme")
VM_HTTP = __import__("os").environ.get("VM_HTTP", "http://localhost:8428")
VL_HTTP = __import__("os").environ.get("VL_HTTP", "http://localhost:9428")
VT_HTTP = __import__("os").environ.get("VT_HTTP", "http://localhost:10428")

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

def count_victorialogs_otel():
    try:
        r = requests.get(f"{VL_HTTP}/select/logsql/query",
                         params={"query": "* | stats count() as total", "limit": 1}, timeout=10)
        if r.status_code == 200:
            lines = [ln for ln in r.text.strip().splitlines() if ln.strip()]
            if lines:
                return int(json.loads(lines[0]).get("total", 0))
    except Exception:
        pass
    return 0


def count_victoriametrics_rows_inserted():
    """Get total rows inserted from /metrics (vm_rows_inserted_total summed across all types)."""
    try:
        r = requests.get(f"{VM_HTTP}/metrics", timeout=10)
        if r.status_code == 200:
            total = 0
            for line in r.text.splitlines():
                if line.startswith("vm_rows_inserted_total{"):
                    parts = line.rsplit(" ", 1)
                    if len(parts) == 2:
                        total += int(float(parts[1]))
            return total
    except Exception:
        pass
    return 0


def count_victoriatraces_otel():
    try:
        r = requests.get(f"{VT_HTTP}/select/logsql/query",
                         params={"query": "* | stats count() as total", "limit": 1}, timeout=10)
        if r.status_code == 200:
            lines = [ln for ln in r.text.strip().splitlines() if ln.strip()]
            if lines:
                return int(json.loads(lines[0]).get("total", 0))
    except Exception:
        pass
    return 0


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

    # Snapshot VM counts before OTLP ingestion (VM has no otel.* table isolation)
    before_vl = count_victorialogs_otel()
    before_vm = count_victoriametrics_rows_inserted()
    before_vt = count_victoriatraces_otel()
    print(f"[otlp] before: VL={before_vl}, VM={before_vm}, VT={before_vt}")

    t0 = time.time()
    run_telemetrygen(args.count, args.rate)
    elapsed = time.time() - t0

    # Poll until VictoriaMetrics rows_inserted increases (up to 30s)
    for i in range(15):
        time.sleep(2)
        if count_victoriametrics_rows_inserted() > before_vm:
            print(f"[otlp] VictoriaMetrics data ready after {(i + 1) * 2}s")
            break
    else:
        print("[otlp] VictoriaMetrics data not ready after 30s, proceeding anyway")

    # VM backends: report delta (after - before) to isolate OTLP-only rows
    after_vl = count_victorialogs_otel()
    after_vm = count_victoriametrics_rows_inserted()
    after_vt = count_victoriatraces_otel()
    print(f"[otlp] after:  VL={after_vl}, VM={after_vm}, VT={after_vt}")

    result = {
        "mechanism": f"OTLP ({args.count} spans, {args.count} logs, {args.count} metrics)",
        "count": args.count,
        "duration_s": round(elapsed, 2),
        "doris": {"rows": count_doris_otel()},
        "clickhouse": {"rows": count_clickhouse_otel()},
        "victorialogs": {"rows": after_vl - before_vl},
        "victoriametrics": {"rows": after_vm - before_vm},
        "victoriatraces": {"rows": after_vt - before_vt},
    }
    for k in ["doris", "clickhouse", "victorialogs", "victoriametrics", "victoriatraces"]:
        r = result[k]["rows"]
        result[k]["rows_per_sec"] = round(r / result["duration_s"], 0) if result["duration_s"] > 0 else 0

    if args.stats:
        args.stats.write_text(json.dumps(result))
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
