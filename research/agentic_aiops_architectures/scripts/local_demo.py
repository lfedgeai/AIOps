#!/usr/bin/env python3
"""
Local demo: start ClickHouse, create OTEL schema, seed synthetic fault data.

Use when cluster (oc, flagd) is unavailable. Run harness with --skip-fault-injection.

Usage:
  python scripts/local_demo.py                    # seed only (ClickHouse must be at localhost:8123)
  python scripts/local_demo.py --start-ch        # start Docker ClickHouse + seed
  python scripts/local_demo.py --start-ch --run  # then run harness with --skip-fault-injection
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Use 28123 to avoid conflict with oc port-forward (8123) and other services
CLICKHOUSE_HTTP = "http://localhost:28123"


def _run_sql(sql: str, ch: str = CLICKHOUSE_HTTP) -> str:
    import requests
    try:
        # ClickHouse HTTP: POST body = SQL query
        r = requests.post(ch, data=sql.encode("utf-8"), timeout=10)
        if r.status_code != 200:
            return f"Error: {r.status_code} {r.reason}: {r.text[:500]}"
        return (r.text or "").strip()
    except Exception as e:
        return f"Error: {e}"


def start_clickhouse() -> bool:
    """Start ClickHouse in Docker if not already running."""
    # Check if already up
    out = _run_sql("SELECT 1")
    if "1" in out and "Error" not in out:
        print(f"[local_demo] ClickHouse already running at {CLICKHOUSE_HTTP}")
        return True
    print("[local_demo] Starting ClickHouse container...")
    r = subprocess.run(
        [
            "docker", "run", "-d", "--name", "agentic-aiops-clickhouse",
            "-p", "28123:8123",
            "-e", "CLICKHOUSE_DB=otel",
            "-e", "CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1",
            "clickhouse/clickhouse-server:24.3",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    if r.returncode != 0:
        if "already in use" in (r.stderr or ""):
            subprocess.run(["docker", "start", "agentic-aiops-clickhouse"], capture_output=True)
        else:
            print(f"[local_demo] Failed to start ClickHouse: {r.stderr}")
            return False
    for _ in range(30):
        time.sleep(1)
        out = _run_sql("SELECT 1")
        if "1" in out and "Error" not in out:
            print("[local_demo] ClickHouse ready")
            return True
    print("[local_demo] ClickHouse failed to become ready")
    return False


def create_schema(ch: str) -> bool:
    """Create otel.otel_logs and otel.otel_traces matching agent queries."""
    # Schema: agents expect SeverityText, Timestamp, ServiceName, Body (logs)
    # and Timestamp, Duration (ns), ServiceName (traces). Run each statement separately.
    statements = [
        "CREATE DATABASE IF NOT EXISTS otel",
        """CREATE TABLE IF NOT EXISTS otel.otel_logs (
        Timestamp DateTime64(3),
        SeverityText String,
        ServiceName String,
        Body String
    ) ENGINE = MergeTree()
    ORDER BY (Timestamp, ServiceName)""",
        """CREATE TABLE IF NOT EXISTS otel.otel_traces (
        Timestamp DateTime64(3),
        Duration Int64,
        ServiceName String
    ) ENGINE = MergeTree()
    ORDER BY (Timestamp, ServiceName)""",
    ]
    for sql in statements:
        out = _run_sql(sql.strip(), ch)
        if "Error" in out:
            print(f"[local_demo] Schema error: {out}")
            return False
    print("[local_demo] Schema created")
    return True


def seed_data(ch: str, since_ts: str | None = None) -> bool:
    """Insert synthetic error logs and high-latency spans to trigger agent detection."""
    # Use timestamps within [fault_injection_time, now]; fault_injection_time = now - 2 min
    # So seed with now - 1 min to ensure data is in window
    if since_ts:
        # Parse ISO8601 to datetime
        try:
            dt = datetime.fromisoformat(since_ts.replace("Z", "+00:00"))
        except ValueError:
            dt = datetime.now(timezone.utc)
    else:
        dt = datetime.now(timezone.utc)

    base_ts = dt.strftime("%Y-%m-%d %H:%M:%S")
    # 6 error logs from cart-service + 4 high-latency spans (>5s) for agent detection
    ns_5s = 5_000_000_000

    logs_sql = f"""
    INSERT INTO otel.otel_logs (Timestamp, SeverityText, ServiceName, Body) VALUES
    ('{base_ts}', 'error', 'cart-service', 'Cart failed to add item'),
    ('{base_ts}', 'error', 'cart-service', 'Cart failed to add item'),
    ('{base_ts}', 'error', 'cart-service', 'Cart failed to add item'),
    ('{base_ts}', 'error', 'cart-service', 'Cart failed to add item'),
    ('{base_ts}', 'error', 'cart-service', 'Cart failed to add item'),
    ('{base_ts}', 'error', 'cart-service', 'Cart failed to add item');
    """
    traces_sql = f"""
    INSERT INTO otel.otel_traces (Timestamp, Duration, ServiceName) VALUES
    ('{base_ts}', {ns_5s + 1000000000}, 'cart-service'),
    ('{base_ts}', {ns_5s + 2000000000}, 'cart-service'),
    ('{base_ts}', {ns_5s + 3000000000}, 'cart-service'),
    ('{base_ts}', {ns_5s + 4000000000}, 'cart-service');
    """
    for sql in [logs_sql.strip(), traces_sql.strip()]:
        out = _run_sql(sql, ch)
        if "Error" in out:
            print(f"[local_demo] Seed error: {out}")
            return False
    print(f"[local_demo] Seeded 6 error logs + 4 high-latency spans (since ~{base_ts})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Local demo: ClickHouse + synthetic data")
    ap.add_argument("--start-ch", action="store_true", help="Start ClickHouse via Docker")
    ap.add_argument("--run", action="store_true", help="Run harness with --skip-fault-injection")
    ap.add_argument("--since", help="ISO8601 for seed window start (default: now - 2 min)")
    args = ap.parse_args()

    if args.start_ch and not start_clickhouse():
        return 1
    if not create_schema(CLICKHOUSE_HTTP):
        return 1

    # Seed with timestamps at "now - 1 min" so data falls in harness window [now-2min, now]
    from datetime import timedelta
    seed_ts = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    if not seed_data(CLICKHOUSE_HTTP, seed_ts):
        return 1

    if args.run:
        harness = ROOT / "code" / "harness" / "run_harness.py"
        env = os.environ.copy()
        env["CLICKHOUSE_HTTP"] = CLICKHOUSE_HTTP
        cmd = [
            "python3", str(harness),
            "--skip-fault-injection",
            "--detection-timeout", "30",
            "--poll-interval", "5",
        ]
        # MLflow is on by default (same as CLI). Opt out: LOCAL_DEMO_NO_MLFLOW=1
        if os.environ.get("LOCAL_DEMO_NO_MLFLOW", "").lower() in ("1", "true", "yes"):
            cmd.append("--no-mlflow")
        print(f"[local_demo] Running: CLICKHOUSE_HTTP={CLICKHOUSE_HTTP} {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=str(ROOT), env=env).returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
