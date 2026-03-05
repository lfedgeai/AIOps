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
- `docker-compose.druid.yml` — Druid (extends main)
- `docker-compose.oceanbase.yml` — OceanBase CE (extends main)
- `schemas/doris.sql`, `schemas/clickhouse.sql`, `schemas/oceanbase.sql` — database and tables (logs, spans, metrics)
- `loaders/replay_doris.py` — Doris replayer using Stream Load HTTP API
- `loaders/replay_clickhouse.py` — ClickHouse replayer via HTTP
- `loaders/replay_druid.py` — Druid native batch ingestion
- `loaders/replay_oceanbase.py` — OceanBase via MySQL protocol (pymysql)
- `queries/{doris,clickhouse,druid,oceanbase}/*.sql` — canonical queries per backend
- `runner/bench.py` — Doris-only: schema → load → queries → report
- `runner/bench_compare.py` — Doris vs ClickHouse vs Druid vs OceanBase: same flow on all, combined report
- `out/` — run outputs (`storage_bench_doris_<ts>/`, `storage_bench_compare_<ts>/`, `rolling_index.html`)
- `otel-collector-config.yaml` — OTLP receiver → Doris + ClickHouse exporters
- `docker-compose.otel.yml` — OTLP collector service (extends main compose)
- `runner/run_otlp_ingest.py` — sends 1000 spans/logs/metrics via telemetrygen → collector
- `runner/map_otlp_to_telemetry.py` — maps `otel.*` → `telemetry.*` so queries use batch + OTLP data

## Quickstart

**Doris-only:**
```bash
cd research/telemetry_storage_backend
make up                  # start Doris
make bench               # run benchmark (uses telemetry_data/)
make down                # stop
```

**Doris vs ClickHouse vs Druid vs OceanBase comparison:**
```bash
make up-compare          # start Doris + ClickHouse + Druid + OceanBase
make bench-compare       # run comparison benchmark (file load only, no telemetrygen)
make bench-compare SCALE_TO=5000   # scale to 5k rows per type
make down
```

**OTLP ingestion (telemetrygen → collector):**
```bash
make up-otel             # start stack + OTLP collector
make bench-otlp          # file load + telemetrygen (1000 spans, 1000 logs, 1000 metrics) via OTLP → Doris + ClickHouse
make down
```

## Data sources

See `docs/DATA_SOURCES.md` for 50k correlated benchmark options.

| Run | Data source | Notes |
|-----|-------------|-------|
| `bench-compare` | `telemetry_data/` | Pre-collected files (logs_*.txt, traces_*.json, metrics_*.json). No telemetrygen. |
| `bench-otlp` | `telemetry_data/` + telemetrygen | Same file load; additionally sends 1000 spans/logs/metrics via telemetrygen → OTLP collector → Doris + ClickHouse. |

Notes:
- Requires Docker + docker compose.

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
- `out/storage_bench_compare_<ts>/` — Doris vs ClickHouse vs Druid vs OceanBase: `compare.html`, `*_queries.json` per backend
- `out/rolling_index.html` — unified index of all runs (newest first)

The ingestion comparison table shows: **Backend | Mechanism | Duration (s) | Rows | Rows/sec**. Mechanism describes the ingest method (e.g. `Batch file load (5000 rows)` or `OTLP (1000 spans, 1000 logs, 1000 metrics)`). With `--otlp`, OTLP rows are appended to the table.

## Ingestion benchmark (how it works)

**Batch file load** — Same JSON/JSONL files from `telemetry_data/` are replayed into Doris, ClickHouse, Druid, and OceanBase via each loader:
- Doris: Stream Load HTTP API (`loaders/replay_doris.py`)
- ClickHouse: HTTP INSERT (`loaders/replay_clickhouse.py`)
- Druid: Native batch ingestion via Overlord (`loaders/replay_druid.py`)
- OceanBase: MySQL protocol via pymysql (`loaders/replay_oceanbase.py`)

All four backends get identical data. Ingest duration and rows/sec are measured per backend. **Message size** is capped at 200KB in `loaders/common.py` to avoid huge Druid/OceanBase files when scaling.

**OTLP ingestion** — When `--otlp` is used, telemetrygen sends N spans, N logs, and N metrics (gRPC) to the OpenTelemetry Collector (`otel-collector-config.yaml`). The collector batches and exports to Doris and ClickHouse only. Rows are counted in `otel.*` tables after a short flush delay; duration is end-to-end (telemetrygen start → last batch exported). The OTLP data is then **mapped into `telemetry.*`** via `runner/map_otlp_to_telemetry.py` (INSERT … SELECT from `otel.*` with column mapping), so canonical queries run against batch + OTLP data combined.

**Why Druid is not in OTLP** — The OpenTelemetry Collector has no Druid exporter. Doris and ClickHouse both have official OTLP/contrib exporters; Druid typically ingests OTLP data via Kafka (collector → Kafka → Druid). Adding Druid to the OTLP path would require a Kafka-based pipeline, which this harness does not implement.

## Query benchmark (how it works)

After ingestion (and OTLP mapping if `--otlp`), the runner executes the same set of SQL queries on each backend. Each query file in `queries/{doris,clickhouse,druid,oceanbase}/*.sql` is run once per backend via its native API:

- **Doris** — `mysql` client over Docker (`telemetry.logs`, `telemetry.spans`, `telemetry.metrics`)
- **ClickHouse** — HTTP POST to `:8123` with `?query=...`
- **Druid** — HTTP POST to `:8888/druid/v2/sql` with JSON body
- **OceanBase** — `mysql` client over Docker (port 2881, MySQL-compatible)

**What is measured** — For each query, the runner records:
- **Latency (s)** — Wall-clock time from query start to completion (includes network, parsing, execution)
- **Rows** — Number of rows in the **result set** returned by the query (not rows scanned). Queries use `LIMIT 20` or `LIMIT 100`, so row counts are capped.

**Report** — `compare.html` includes:
- **Data volume (full scan)** — Total rows in `telemetry.*` at query time (logs + spans + metrics) and the latency of a full COUNT(*) scan per table. Confirms the combined dataset size and measures full-scan performance.
- **Query comparison** — Bar chart and table with latency, result row count, and error per backend. Includes a `data_volume` query that runs full COUNT on all three tables. The fastest backend and % difference vs. slowest are shown.
- All backends query the same data: batch load + (when `--otlp`) mapped OTLP data in `telemetry.*`.

### Query result differences (row count)

Some queries return different row counts across backends:

- **spans_error_by_service** — Doris and ClickHouse both return 2 rows (Doris uses `$."http.status_code"` for JSON keys with dots). Druid omits or handles `http.status_code` differently (Druid SQL JSON support varies).
- **metrics_p95_latency, metrics_by_service_hourly** — Doris and ClickHouse include OTLP-mapped metrics (e.g. `gen` from telemetrygen); Druid does not, so it has fewer metric names. Hour bucketing also differs (date_trunc vs TIME_FLOOR).
- **traces_slow_by_service** — Returns 0 when no spans have `duration_ms > 500`. Batch and telemetrygen spans are usually short, so empty results are expected.
- **OceanBase on log-search queries** — `logs_errors_by_service` and `logs_search_error` scan the `message` column with `LIKE '%error%'`. OceanBase is row-oriented; these queries can be 10–50× slower than ClickHouse and may time out at scale. OceanBase is suited for OLTP and mixed workloads rather than analytical log search.

---

See [SCALING.md](SCALING.md) for scaling and tuning (50k rows, batch size, streaming simulation, infrastructure).
