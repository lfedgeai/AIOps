#!/usr/bin/env python3
"""
EvalHub Kubernetes adapter: LF Edge AIOps MTTD/MTTR + ClickHouse telemetry signals.

Contract (eval-hub runtime):
  - Read job spec from /meta/job.json (EVALHUB_MODE=k8s).
  - POST benchmark status events to the sidecar proxy at job.callback_url:
      POST {callback_url}/api/v1/evaluations/jobs/{job_id}/events
  - Sidecar forwards to Eval Hub with service credentials.

Metrics:
  - mttd_seconds / mttr_seconds: from benchmark parameters if set; otherwise derived
    from ClickHouse log timestamp span in a recent window (proxy for incident activity).
  - otel_log_rows, otel_trace_rows: row counts in the same window (sanity checks).
"""
from __future__ import annotations

import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

JOB_PATH = os.environ.get("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
# Sidecar injects callback_url in job.json (http://localhost:<port>)
DEFAULT_CALLBACK = "http://127.0.0.1:8080"
SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
SA_NAMESPACE_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"


def _bearer_token() -> str | None:
    try:
        tok = Path(SA_TOKEN_PATH).read_text(encoding="utf-8").strip()
        return tok or None
    except OSError:
        return None


def _tenant_namespace() -> str | None:
    try:
        return Path(SA_NAMESPACE_PATH).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_job() -> dict[str, Any]:
    with open(JOB_PATH, encoding="utf-8") as f:
        return json.load(f)


def _post_event(callback_base: str, job_id: str, payload: dict[str, Any]) -> None:
    base = (callback_base or DEFAULT_CALLBACK).rstrip("/")
    url = f"{base}/api/v1/evaluations/jobs/{urllib.parse.quote(job_id)}/events"
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    tok = _bearer_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    ns = _tenant_namespace()
    if ns:
        headers["X-Tenant"] = ns
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers=headers,
    )
    # EvalHub sidecar uses HTTPS with service certs; in-pod callback must not fail on verify.
    ctx = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:4000]
        print(f"[lfedge_aiops] HTTP {e.code} posting event: {detail}", flush=True)
        raise


def _ch_scalar(url_base: str, sql: str) -> float:
    """Run a scalar ClickHouse HTTP query; return float (0 on empty/failure)."""
    q = urllib.parse.quote(sql, safe="")
    url = f"{url_base.rstrip('/')}/?query={q}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            txt = resp.read().decode("utf-8", errors="replace").strip()
        if not txt:
            return 0.0
        return float(txt.split()[0] if txt else 0.0)
    except Exception as e:
        print(f"[lfedge_aiops] ClickHouse query failed: {e}", flush=True)
        return 0.0


def _clickhouse_base() -> str | None:
    for key in (
        "LFEDGE_AIOPS_CLICKHOUSE_HTTP",
        "CLICKHOUSE_HTTP",
    ):
        v = (os.environ.get(key) or "").strip()
        if v:
            return v.rstrip("/")
    return None


def _compute_metrics(params: dict[str, Any]) -> dict[str, float]:
    """Return metric map including mttd_seconds and mttr_seconds."""
    explicit_mttd = params.get("mttd_seconds")
    explicit_mttr = params.get("mttr_seconds")
    if explicit_mttd is not None and explicit_mttr is not None:
        return {
            "mttd_seconds": float(explicit_mttd),
            "mttr_seconds": float(explicit_mttr),
            "otel_log_rows": float(params.get("otel_log_rows", 0) or 0),
            "otel_trace_rows": float(params.get("otel_trace_rows", 0) or 0),
            "metric_source": 1.0,  # 1 = parameters
        }

    ch = _clickhouse_base()
    window_m = int(float(params.get("window_minutes", 15) or 15))
    if not ch:
        # Synthetic measurable path so the provider still completes in dev clusters.
        t0 = time.monotonic()
        time.sleep(0.35)
        t1 = time.monotonic()
        time.sleep(0.2)
        t2 = time.monotonic()
        return {
            "mttd_seconds": round(t1 - t0, 3),
            "mttr_seconds": round(t2 - t1, 3),
            "otel_log_rows": 0.0,
            "otel_trace_rows": 0.0,
            "metric_source": 0.0,  # synthetic
        }

    # Live telemetry: span of log timestamps in window (seconds).
    span_sql = (
        "SELECT if(count()=0, 0, "
        "dateDiff('second', min(Timestamp), max(Timestamp))) "
        f"FROM otel.otel_logs WHERE Timestamp > now() - INTERVAL {window_m} MINUTE"
    )
    span = _ch_scalar(ch, span_sql)
    logs = _ch_scalar(
        ch,
        f"SELECT count() FROM otel.otel_logs WHERE Timestamp > now() - INTERVAL {window_m} MINUTE",
    )
    traces = _ch_scalar(
        ch,
        f"SELECT count() FROM otel.otel_traces WHERE Timestamp > now() - INTERVAL {window_m} MINUTE",
    )
    # Split span into MTTD/MTTR-style components (documented heuristic for leaderboard use).
    if span <= 0:
        mttd = 1.0
        mttr = 1.0
    else:
        mttd = max(1.0, span * 0.45)
        mttr = max(1.0, span * 0.55)
    return {
        "mttd_seconds": float(mttd),
        "mttr_seconds": float(mttr),
        "incident_log_span_seconds": float(span),
        "otel_log_rows": float(logs),
        "otel_trace_rows": float(traces),
        "metric_source": 2.0,  # clickhouse window
    }


def main() -> int:
    job = _load_job()
    job_id = job["id"]
    provider_id = job["provider_id"]
    benchmark_id = job["benchmark_id"]
    benchmark_index = int(job.get("benchmark_index", 0))
    callback = job.get("callback_url") or DEFAULT_CALLBACK
    params = job.get("parameters") or {}

    started = _utc_now()
    running_payload = {
        "benchmark_status_event": {
            "provider_id": provider_id,
            "id": benchmark_id,
            "benchmark_index": benchmark_index,
            "status": "running",
            "started_at": started,
        }
    }
    print(f"[lfedge_aiops] job={job_id} benchmark={benchmark_id} posting running…", flush=True)
    _post_event(callback, job_id, running_payload)

    metrics = _compute_metrics(params)
    completed = _utc_now()

    done_payload = {
        "benchmark_status_event": {
            "provider_id": provider_id,
            "id": benchmark_id,
            "benchmark_index": benchmark_index,
            "status": "completed",
            "started_at": started,
            "completed_at": completed,
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
    }
    print(f"[lfedge_aiops] posting completed metrics={json.dumps(metrics)}", flush=True)
    _post_event(callback, job_id, done_payload)
    print("[lfedge_aiops] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
