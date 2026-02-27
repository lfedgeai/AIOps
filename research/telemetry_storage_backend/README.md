# Telemetry Storage Backend Benchmark

This harness replays pre-collected OpenTelemetry-like ground-truth (`telemetry_data/`) into Apache Doris and/or ClickHouse, runs canonical queries for logs/traces/metrics, and produces reports.

## Layout

- `docker-compose.yml` — Doris + ClickHouse for comparison trials
- `schemas/doris.sql` — Doris database and tables (logs, spans, metrics)
- `schemas/clickhouse.sql` — ClickHouse database and tables
- `loaders/replay.py` — Doris replayer using Stream Load HTTP API
- `loaders/replay_clickhouse.py` — ClickHouse replayer via HTTP
- `queries/doris/*.sql` — canonical Doris queries
- `queries/clickhouse/*.sql` — canonical ClickHouse queries
- `runner/bench.py` — Doris-only: schema → load → queries → report
- `runner/bench_compare.py` — Doris vs ClickHouse: same flow on both, combined report
- `out/` — run outputs (`storage_bench_doris_<ts>/`, `storage_bench_compare_<ts>/`, `rolling_index.html`)

## Quickstart

**Doris-only:**
```bash
cd research/telemetry_storage_backend
make up                  # start Doris
make bench               # run benchmark (uses telemetry_data/)
make down                # stop
```

**Doris vs ClickHouse comparison:**
```bash
make up-compare          # start Doris + ClickHouse
make bench-compare       # run comparison benchmark
make down
```

Notes:
- Requires Docker + docker compose.
- Data source: `telemetry_data/` (logs_*.txt, traces_*.json, metrics_*.json).

## Canonical queries
- Logs:
  - Error counts by service over [t0,t1]
  - Top-K services by error spike (1m bucket)
- Traces:
  - Slow traces per service (duration > threshold)
  - Fan-out/fan-in aggregations per service
- Metrics:
  - p95 latency by service over [t0,t1]
  - Request rate sum/avg

## Outputs
- `out/storage_bench_doris_<ts>/` — Doris-only: `summary.html`, `ingest.json`, `queries.json`
- `out/storage_bench_compare_<ts>/` — Doris vs ClickHouse: `compare.html`, `doris_queries.json`, `clickhouse_queries.json`
- `out/rolling_index.html` — unified index of all runs (newest first)
