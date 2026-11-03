"""Synthetic application logs that match the scenario.

Each scenario profile lists log "kinds" with a count range. Lines are drawn
from fixed templates, so logs are realistic but fully determined by the seed.
Templates with service "*" are attributed to an affected (non-healthy)
service, so logs point at the same service the metrics do.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

# Fixed reference time keeps timestamps reproducible from the seed alone.
REFERENCE_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
WINDOW_S = 15 * 60
LEVELS = ("INFO", "WARN", "ERROR", "FATAL")

TEMPLATES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "info_ok": (
        ("INFO", "api", "GET /api/orders 200 in {ms}ms"),
        ("INFO", "api", "GET /api/health 200 in {small}ms"),
        ("INFO", "worker", "job {job} completed in {s}s"),
        ("INFO", "database", "checkpoint complete: wrote {n} buffers"),
        ("INFO", "cache", "eviction run freed {n} keys"),
    ),
    "info_api": (
        ("INFO", "api", "GET /api/orders 200 in {ms}ms"),
        ("INFO", "api", "GET /api/health 200 in {small}ms"),
        ("INFO", "database", "checkpoint complete: wrote {n} buffers"),
    ),
    "warn_benign": (
        ("WARN", "api", "deprecated config key 'cache.ttl' is used; use 'cache.ttl_seconds'"),
        ("WARN", "cache", "request to cache retried (attempt 2), succeeded"),
        ("WARN", "worker", "job {job} took {s}s, above soft limit 30s"),
    ),
    "warn_slow": (
        ("WARN", "database", "slow query {big}ms: SELECT * FROM orders WHERE customer_id = ?"),
        ("WARN", "api", "request queue depth {n} above threshold 50"),
        ("WARN", "api", "GET /api/orders 200 in {big}ms"),
    ),
    "error_timeout": (
        ("ERROR", "*", "upstream timeout after 5000ms calling database"),
        ("ERROR", "api", "connection pool exhausted (max=20, waiting={n})"),
        ("ERROR", "api", "GET /api/orders 504 in 5003ms"),
    ),
    "error_crash": (
        ("FATAL", "*", "process killed: out of memory (rss={mb}MB)"),
        ("ERROR", "*", "process exited with code 137"),
        ("ERROR", "api", "connection refused: {svc}:8080"),
    ),
    "error_hung": (
        ("ERROR", "worker", "watchdog: worker unresponsive, no jobs processed for {s}s"),
        ("ERROR", "worker", "deadlock detected: lock 'jobs_queue' held for {s}s by thread job-runner-3"),
    ),
    "warn_queue": (
        ("WARN", "worker", "{n} jobs waiting in queue, 0 in progress"),
    ),
    "error_app": (
        ("ERROR", "api", "unhandled exception in POST /api/checkout: KeyError: 'currency'"),
        ("ERROR", "api", "POST /api/checkout 500 in {ms}ms"),
    ),
    "health_ok": (
        ("INFO", "*", "health check ok"),
    ),
}


def _values(rng: random.Random, svc: str) -> dict:
    return {
        "ms": rng.randint(20, 180),
        "small": rng.randint(2, 9),
        "big": rng.randint(1500, 4000),
        "s": rng.randint(60, 600),
        "n": rng.randint(20, 400),
        "mb": rng.randint(3800, 4096),
        "job": rng.randint(10000, 99999),
        "svc": svc,
    }


def generate_logs(
    rng: random.Random,
    spec: tuple[tuple[str, int, int], ...],
    affected: list[str],
    service_names: tuple[str, ...],
) -> list[dict]:
    """Return log lines, oldest first, as {ts, level, service, message} dicts."""
    lines = []
    for kind, lo, hi in spec:
        for _ in range(rng.randint(lo, hi)):
            level, service, template = rng.choice(TEMPLATES[kind])
            if service == "*":
                service = rng.choice(affected) if affected else rng.choice(service_names)
            target = rng.choice(affected) if affected else rng.choice(service_names)
            message = template.format(**_values(rng, target))
            age = rng.randint(0, WINDOW_S)
            lines.append((age, level, service, message))
    lines.sort(key=lambda x: -x[0])  # oldest first
    return [
        {
            "ts": (REFERENCE_TIME - timedelta(seconds=age)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": level,
            "service": service,
            "message": message,
        }
        for age, level, service, message in lines
    ]
