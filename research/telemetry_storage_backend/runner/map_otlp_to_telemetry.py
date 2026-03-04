#!/usr/bin/env python3
"""
Map OTLP data (otel.*) into telemetry schema (telemetry.*) so canonical queries
run against the combined batch + OTLP data.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
CH_HTTP = os.environ.get("CLICKHOUSE_HTTP", "http://localhost:8123")
CH_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD", "changeme")
DORIS_PASS = os.environ.get("DORIS_PASS", "")


def _ch_params(extra: dict | None = None) -> dict:
    p = dict(extra) if extra else {}
    if CH_PASSWORD:
        p["user"] = "default"
        p["password"] = CH_PASSWORD
    return p


def _run_doris_sql(sql: str, db: str = "telemetry") -> None:
    pass_arg = f"-p{DORIS_PASS}" if DORIS_PASS else ""
    subprocess.run(
        ["docker", "run", "--rm", "--network", "tsb-net", "-i",
         "mysql:8", "sh", "-lc", f"mysql -h tsb-doris -P 9030 -uroot {pass_arg} -D {db}"],
        input=sql.encode(),
        check=True,
        timeout=120,
    )


def map_doris() -> dict:
    """Copy otel.* -> telemetry.* for Doris. Returns row counts."""
    counts = {}
    # Logs: otel_logs -> telemetry.logs
    sql = """
INSERT INTO telemetry.logs (ts, service, level, message, trace_id, span_id, attrs)
SELECT timestamp, service_name, COALESCE(severity_text, 'INFO'), COALESCE(body, ''), trace_id, span_id,
       COALESCE(CAST(log_attributes AS STRING), CAST(resource_attributes AS STRING), '{}')
FROM otel.otel_logs
"""
    try:
        _run_doris_sql(sql, db="otel")
    except Exception as e:
        print(f"[map_doris] logs: {e}")

    # Spans: otel_traces -> telemetry.spans (duration is microseconds in Doris)
    sql = """
INSERT INTO telemetry.spans (ts_start, trace_id, ts_end, span_id, parent_span_id, service, name, duration_ms, attributes)
SELECT timestamp, trace_id, end_time, span_id, parent_span_id, service_name, span_name,
       duration / 1000, COALESCE(CAST(span_attributes AS STRING), '{}')
FROM otel.otel_traces
"""
    try:
        _run_doris_sql(sql, db="otel")
    except Exception as e:
        print(f"[map_doris] spans: {e}")

    # Metrics: otel_metrics_gauge, otel_metrics_sum -> telemetry.metrics
    for tbl in ["otel_metrics_gauge", "otel_metrics_sum"]:
        sql = f"""
INSERT INTO telemetry.metrics (ts, metric_name, value, labels)
SELECT timestamp, metric_name, value, COALESCE(CAST(attributes AS STRING), '{{}}')
FROM otel.{tbl}
"""
        try:
            _run_doris_sql(sql, db="otel")
        except Exception as e:
            print(f"[map_doris] {tbl}: {e}")

    return counts


def map_clickhouse() -> dict:
    """Copy otel.* -> telemetry.* for ClickHouse. Returns row counts."""
    # Logs: otel_logs -> telemetry.logs
    # ClickHouse: Timestamp, TraceId, SpanId, ServiceName, SeverityText, Body, ResourceAttributes, LogAttributes
    sql = """
    INSERT INTO telemetry.logs (ts, service, level, message, trace_id, span_id, attrs)
    SELECT Timestamp, ServiceName, coalesce(SeverityText, 'INFO'), coalesce(Body, ''), TraceId, SpanId,
           coalesce(toJSONString(LogAttributes), toJSONString(ResourceAttributes), '{}')
    FROM otel.otel_logs
    """
    try:
        r = requests.post(CH_HTTP, params=_ch_params(), data=sql, timeout=120)
        if r.status_code != 200:
            print(f"[map_clickhouse] logs: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[map_clickhouse] logs: {e}")

    # Spans: otel_traces -> telemetry.spans (Duration is nanoseconds in OTLP/CH)
    # ts_end = Timestamp + Duration/1e9
    sql = """
    INSERT INTO telemetry.spans (ts_start, trace_id, ts_end, span_id, parent_span_id, service, name, duration_ms, attributes)
    SELECT Timestamp, TraceId, addNanoseconds(Timestamp, Duration), SpanId, ParentSpanId, ServiceName, SpanName,
           toInt64(Duration / 1000000), coalesce(toJSONString(SpanAttributes), '{}')
    FROM otel.otel_traces
    """
    try:
        r = requests.post(CH_HTTP, params=_ch_params(), data=sql, timeout=120)
        if r.status_code != 200:
            print(f"[map_clickhouse] spans: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[map_clickhouse] spans: {e}")

    # Metrics: gauge and sum
    for tbl in ["otel_metrics_gauge", "otel_metrics_sum"]:
        sql = f"""
        INSERT INTO telemetry.metrics (ts, metric_name, value, labels)
        SELECT TimeUnix, MetricName, Value, coalesce(toJSONString(Attributes), '{{}}')
        FROM otel.{tbl}
        """
        try:
            r = requests.post(CH_HTTP, params=_ch_params(), data=sql, timeout=120)
            if r.status_code != 200:
                print(f"[map_clickhouse] {tbl}: {r.status_code} {r.text[:200]}")
        except Exception as e:
            print(f"[map_clickhouse] {tbl}: {e}")

    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Map otel.* data into telemetry.* for query benchmarking")
    ap.add_argument("--doris", action="store_true", help="Map Doris otel -> telemetry")
    ap.add_argument("--clickhouse", action="store_true", help="Map ClickHouse otel -> telemetry")
    ap.add_argument("--both", action="store_true", help="Map both (default when neither specified)")
    args = ap.parse_args()

    do_both = args.both or (not args.doris and not args.clickhouse)
    if do_both:
        args.doris = args.clickhouse = True

    if args.doris:
        print("[map] Doris otel -> telemetry")
        map_doris()
    if args.clickhouse:
        print("[map] ClickHouse otel -> telemetry")
        map_clickhouse()

    print("[map] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
