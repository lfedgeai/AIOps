# 🔬 Unified Telemetry storage backend comparison
This initiative is comparing different open source storage backend project to evaluate support for AIOps approaches towards autonomous operations.

## 📌 Research Overview
Status: 🏗️ In-Progress, moving towards ✅ Complete & Peer-Reviewed
Domain: Telemetry storage backedn architectures and solutions
Core Question: What unified telemetry storage solutions provide best approaches for AIOps?

## 🎯 Desired Outcomes
Primary Objective: Evaluate storage solutions for performance across
- data ingestion
- model inference
- genAI and predictive model requirements
- logs, traces and metrics
- long term (cold) and hot short term storage requirements.

## Key Deliverables:
Visual Dashboard: Comparison metrics and a UI for results exploration.

## 🛠 Methodology & Framework
Python based test logic, data ingestion
telemetry gen OTLP and static file based data ingestion
SQL based queries against the backends under evaluation

## Approach: Empiric, qualitative analysis
Core Logic: tbd.
Tech Stack: Python (Pandas/NumPy), k8s

## 📊 Data Management & Transparency
1. **Source:** OTEL demo app and telemetrygen tool
2. **Processing:** The data used comes from the OTEL demo app running on k8s applying chaos engineering principles as well as the telemetrygen tool.


## 📂 Repository Structure
```text
├── loeaders/           # logic to load the data
├── out/                # benchmark run test results
├── queries/            # Queries to produce the performance benchmark
├── runner/             # benchmark run logic
├── docs/               # In-depth documentation and literature review
├── schemas/            # backend storage chemas
├── telemetry_data/     # static logs, metrics, traces and metadata
└── README.md           # This file
```


# Implementation Details: Telemetry Storage Backend Benchmark

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
