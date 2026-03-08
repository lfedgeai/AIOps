#!/usr/bin/env python3
"""
Baseline single-shot classifier for AIOps failure detection.

Non-agentic: fetches a fixed telemetry summary from ClickHouse,
applies threshold-based detection, returns structured output.
No tool calls, no iteration. Used as comparison baseline for agentic approaches.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import requests

CLICKHOUSE_HTTP = os.environ.get("CLICKHOUSE_HTTP", "http://localhost:8123")

# Detection thresholds
DEFAULT_ERROR_THRESHOLD = 5
DEFAULT_LATENCY_SPAN_THRESHOLD = 3


@dataclass
class DetectionResult:
    """Result of one classifier invocation."""
    detected: bool
    first_alert_time: str | None
    confidence: float
    suggested_root_cause: str | None
    suggested_remediations: list[str]
    signals: dict[str, Any]


REMEDIATION_MAP = {
    "cartFailure": [
        "Check cart-service health and restart if unhealthy",
        "Verify Redis/cache connectivity for cart state",
        "Scale cart-service replicas if load is high",
    ],
    "paymentFailure": [
        "Check payment service logs for upstream errors",
        "Verify payment gateway connectivity and credentials",
        "Consider retry with exponential backoff",
    ],
    "paymentUnreachable": [
        "Verify network connectivity to payment service",
        "Check service discovery and DNS resolution",
        "Restart payment-service pod if unreachable",
    ],
    "productCatalogFailure": [
        "Check product-catalog service health",
        "Verify product-catalog-products ConfigMap/data source",
        "Restart product-catalog deployment",
    ],
    "recommendationCacheFailure": [
        "Clear or warm recommendation cache",
        "Check recommendation-service connectivity",
        "Restart recommendation-service if cache backend unreachable",
    ],
    "kafkaQueueProblems": [
        "Check Kafka broker health and partition leaders",
        "Verify consumer lag and backpressure",
        "Scale Kafka consumers or increase partition count",
    ],
    "llmRateLimitError": [
        "Check LLM API rate limits and quotas",
        "Implement request queuing or fallback",
        "Consider switching to alternative model endpoint",
    ],
    "adFailure": [
        "Check ad-service health and dependencies",
        "Restart ad-service deployment",
    ],
    "adHighCpu": ["Check ad-service CPU limits", "Scale ad-service horizontally"],
    "emailMemoryLeak": ["Restart email-service", "Profile for memory leaks"],
    "imageSlowLoad": ["Check image service latency", "Add caching for slow responses"],
    "loadGeneratorFloodHomepage": ["Rate limit load generator", "Scale frontend services"],
}

SVC_TO_FLAG = {
    "cart-service": "cartFailure",
    "cart": "cartFailure",
    "payment": "paymentFailure",
    "product-catalog": "productCatalogFailure",
    "recommendation": "recommendationCacheFailure",
    "ad-service": "adFailure",
    "email": "emailMemoryLeak",
    "frontend": "loadGeneratorFloodHomepage",
}


def query_clickhouse(ch_http: str, sql: str) -> str:
    try:
        r = requests.post(ch_http, data=sql, timeout=30)
        r.raise_for_status()
        return (r.text or "").strip()
    except Exception as e:
        return f"Error: {e}"


def map_service_to_remediations(service_name: str) -> list[str]:
    svc_lower = (service_name or "").lower()
    flag = next((v for k, v in SVC_TO_FLAG.items() if k in svc_lower), None)
    return REMEDIATION_MAP.get(flag or "", [
        "Check service health and restart if needed",
        "Review recent deployments and rollback if necessary",
        "Verify upstream dependencies and network connectivity",
    ])


def run_detection(
    ch_http: str,
    since_ts: str,
    error_threshold: int = DEFAULT_ERROR_THRESHOLD,
    latency_spans_threshold: int = DEFAULT_LATENCY_SPAN_THRESHOLD,
) -> DetectionResult:
    """Single-shot: fetch telemetry, apply thresholds, return result."""
    ns_5s = 5_000_000_000

    out = query_clickhouse(
        ch_http,
        f"SELECT count() FROM otel.otel_logs WHERE SeverityText = 'ERROR' AND Timestamp >= '{since_ts}'",
    )
    try:
        err_count = int(out.split("\n")[0] or 0)
    except (ValueError, IndexError):
        err_count = 0

    out = query_clickhouse(
        ch_http,
        f"SELECT count() FROM otel.otel_traces WHERE Timestamp >= '{since_ts}' AND Duration > {ns_5s}",
    )
    try:
        high_lat_count = int(out.split("\n")[0] or 0)
    except (ValueError, IndexError):
        high_lat_count = 0

    sql = f"""
    SELECT ServiceName, count() as cnt
    FROM otel.otel_logs
    WHERE SeverityText = 'ERROR' AND Timestamp >= '{since_ts}'
    GROUP BY ServiceName
    ORDER BY cnt DESC
    LIMIT 5
    FORMAT TabSeparated
    """
    out = query_clickhouse(ch_http, sql)
    top_services = []
    for ln in out.split("\n"):
        if ln and not ln.startswith("Error:"):
            parts = ln.split("\t")
            if parts:
                top_services.append(parts[0].strip())

    signals = {
        "log_error_count": err_count,
        "high_latency_span_count": high_lat_count,
        "top_error_services": top_services,
    }

    detected = err_count >= error_threshold or high_lat_count >= latency_spans_threshold
    suggested_root_cause = top_services[0] if top_services else None
    remediations = map_service_to_remediations(suggested_root_cause) if suggested_root_cause else map_service_to_remediations("")

    confidence = 0.0
    if detected:
        confidence = min(0.95, 0.3 + (err_count / 50) + (high_lat_count / 20))

    first_alert_time = datetime.now(timezone.utc).isoformat() if detected else None

    return DetectionResult(
        detected=detected,
        first_alert_time=first_alert_time,
        confidence=confidence,
        suggested_root_cause=suggested_root_cause,
        suggested_remediations=remediations,
        signals=signals,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline single-shot classifier")
    ap.add_argument("--since", required=True, help="ISO8601 timestamp for detection window start")
    ap.add_argument("--error-threshold", type=int, default=DEFAULT_ERROR_THRESHOLD)
    ap.add_argument("--latency-threshold", type=int, default=DEFAULT_LATENCY_SPAN_THRESHOLD)
    ap.add_argument("--json", action="store_true", help="Output JSON to stdout")
    args = ap.parse_args()

    ch_http = os.environ.get("CLICKHOUSE_HTTP", CLICKHOUSE_HTTP)
    result = run_detection(
        ch_http,
        args.since,
        error_threshold=args.error_threshold,
        latency_spans_threshold=args.latency_threshold,
    )

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print(f"detected={result.detected} confidence={result.confidence:.2f}")
        if result.suggested_root_cause:
            print(f"root_cause={result.suggested_root_cause}")
        for r in result.suggested_remediations:
            print(f"  - {r}")

    return 0 if result.detected else 1


if __name__ == "__main__":
    sys.exit(main())
