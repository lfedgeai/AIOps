# Telemetry Storage Backend Benchmark — Findings Report

**Generated:** 2026-11-11 · 
**Author:** @aspanner 
**research periods** `out/` (2026-02-27 → 2026-06-17)
**Scope:** unified telemetry storage evaluation for AIOps — ingestion and canonical query performance across **8 backends** storing logs, traces and metrics.

---

## 1. Executive summary

The benchmark answers: *which open-source storage backend best serves AIOps workloads that must ingest OTEL telemetry and run correlated log/trace/metric queries?*

**Headline findings (final run, 2026-06-17, 8 backends):**

1. **ClickHouse is the fastest unified SQL backend** — every canonical query completes in **4–61 ms**, including the 3-way log↔trace↔metric correlation joins (8–14 ms). It is also the only SQL backend whose latency is stable across all four recorded benchmark rounds.
2. **The Victoria per-signal stack (VictoriaLogs / VictoriaMetrics / VictoriaTraces) is the fastest overall per tier** (2–5 ms on metrics/traces queries), and its **federated correlation** (Python merge across the three stores) is competitive with ClickHouse (11–16 ms vs 8–14 ms) — but it requires running three stores plus merge logic instead of one.
3. **Druid is a solid second among SQL backends** (14–90 ms per query) but has the **slowest ingestion** (43 rows/s vs ClickHouse's 170).
4. **Doris is consistently 30–60× slower than ClickHouse** on this dataset (331–563 ms per query) with no compensating ingest advantage.
5. **OceanBase is unsuited to analytical log search** — `logs_search_error` at 1.8–3.7 s (30–50× ClickHouse); it is an OLTP engine and behaves like one here.
6. **Production consequence:** ClickHouse was selected and is now the telemetry store for the whole AIOps stack — the evaluation harness, the runtime harness (own `clickhouse-rt` instance), and the OTEL collector export path.

## 2. What was measured

### Backends

| Backend | Class | Ingest mechanism | Signal coverage |
|---|---|---|---|
| ClickHouse | columnar SQL | HTTP INSERT | logs + spans + metrics (unified) |
| Apache Doris | MPP SQL | Stream Load HTTP API | logs + spans + metrics (unified) |
| Apache Druid | columnar store | native batch (Overlord) | logs + spans + metrics (unified) |
| OceanBase CE | distributed MySQL | MySQL protocol (pymysql) | logs + spans + metrics (unified) |
| Loki | log store | push API | logs only |
| VictoriaLogs | log store | push API | logs only |
| VictoriaMetrics | TSDB | push API | metrics only |
| VictoriaTraces | trace store | push API | traces only |
| Victoria (federated) | VL + VM + VT + Python merge | push per store | all three, merged in the runner |

### Data and queries

- **Data:** the repo's `telemetry_data/` ground truth (556 logs, 600 spans, 600 metrics — 1,756 rows loaded per backend in the recorded runs), optionally plus OTLP telemetrygen traffic. The `SCALE_TO` option exists for 50k-row runs, but **no scaled run is recorded in `out/`** — all findings below are at base scale.
- **Queries:** 12 canonical query names executed identically per backend (`queries/<backend>/*`), grouped in tiers (see `queries/COMPARABILITY.md`): `logs_*`, `metrics_*`, `spans_*`/`trace_*`/`traces_*`/`sla_*`, `correlation_*` (the AIOps differentiator), and `data_volume`.
- **Measured:** wall-clock query latency and result row count; end-to-end ingest duration and rows/sec.

### Run timeline

| Date | Backends | Note |
|---|---|---|
| 2026-02-27 | 2 (Doris, ClickHouse) | original comparison |
| 2026-03-04 | 3 (+ Druid) | |
| 2026-03-18 | 5 (+ Loki, OceanBase) | |
| 2026-06-16 → 06-17 | 8 (+ VictoriaLogs/Metrics/Traces, later + federated) | final rounds |

## 3. Findings

### 3.1 Ingestion (final run, 1,756 rows / per-signal splits)

| Backend | Mechanism | Duration (s) | Rows/sec |
|---|---|---|---|
| VictoriaTraces | push (traces) | **0.07** | **8,571** |
| VictoriaMetrics | push (metrics) | 1.36 | 442 |
| ClickHouse | batch HTTP INSERT | 10.34 | 170 |
| Doris | Stream Load | 10.99 | 160 |
| OceanBase | MySQL batch | 13.87 | 127 |
| VictoriaLogs | push (logs) | 10.93 | 51 |
| Druid | native batch | 40.45 | 43 |

Victoria's push APIs dominate per-signal ingest. Among unified SQL backends ClickHouse and Doris are equivalent (~160–170 rows/s), OceanBase trails modestly, and **Druid's batch ingest is ~4× slower than the field**.

### 3.2 Query latency (final run, 2026-06-17; seconds)

| Query | Doris | ClickHouse | Druid | OceanBase | VictoriaLogs | VictoriaMetrics | VictoriaTraces | Victoria (fed.) |
|---|---|---|---|---|---|---|---|---|
| correlation_by_timestamp | 0.537 | **0.014** | 0.090 | 0.435 | — | — | — | 0.011 |
| correlation_by_trace_id | 0.462 | **0.008** | 0.031 | 0.328 | — | — | — | 0.016 |
| data_volume | 0.446 | 0.005 | 0.021 | 0.302 | 0.002 | 0.005 | 0.002 | — |
| logs_errors_by_service | 0.563 | 0.051 | 0.064 | 0.808 | **0.004** | — | — | — |
| logs_recent | 0.414 | 0.035 | **0.020** | 0.367 | 0.159 | — | — | — |
| logs_search_error | 0.507 | 0.061 | 0.067 | 1.830 | **0.046** | — | — | — |
| metrics_by_service_hourly | 0.389 | 0.015 | 0.030 | 0.333 | — | **0.003** | — | — |
| metrics_p95_latency | 0.398 | **0.004** | 0.016 | 0.308 | — | ERR | — | — |
| sla_latency_compliance | 0.367 | 0.013 | 0.017 | 0.317 | — | — | **0.002** | — |
| spans_error_by_service | 0.392 | 0.004 | 0.021 | 0.365 | — | — | **0.002** | — |
| trace_by_id | 0.378 | 0.004 | 0.022 | 0.369 | — | — | **0.003** | — |
| traces_slow_by_service | 0.331 | 0.004 | 0.014 | 0.348 | — | — | **0.002** | — |

(Bold = fastest backend for that query. ERR = VictoriaMetrics `metrics_p95_latency` failed in the final run; it had succeeded in the previous round.)

### 3.3 Stability across rounds

ClickHouse's average query latency was **0.043 s (Feb) → 0.048 s (Mar) → ~0.02 s (Jun)** — stable to improving. Doris degraded to ~1.5 s in the 5-backend March round before returning to ~0.4 s in June; OceanBase was consistently the slowest on log-search queries in every round (1.8–3.7 s). No backend other than ClickHouse was stable-fast across all rounds.

### 3.4 Result-shape comparability (caveats)

Per `queries/COMPARABILITY.md`, rankings are only comparable where row counts agree:

- **Agree everywhere:** `logs_recent` (100), `logs_search_error` (100), `logs_errors_by_service` (13), `data_volume` (3), `metrics_*` (2), `sla_*` (1).
- **Diverge:** `trace_by_id` (ClickHouse/Druid 4 rows, Doris 3, OceanBase/VictoriaTraces 1 — trace-id sampling differs); `correlation_by_trace_id` (**0 rows on all SQL backends vs 20 on Victoria federated** — the SQL 3-way LEFT JOIN returns nothing when trace_ids don't match across tables in the window, while the federated merge reports per-trace span counts). The correlation queries are therefore *latency-comparable but not result-comparable* across the SQL/Victoria divide.
- **Empty by design:** `spans_error_by_service`, `traces_slow_by_service` (0 everywhere — the dataset contains no slow/error spans).

## 4. Conclusions

1. **For a unified AIOps telemetry store, ClickHouse is the winner** — fastest SQL queries including correlation joins, stable across rounds, adequate ingest, one engine for all three signals, native OTEL collector exporter. This is now the production choice across the AIOps program (evaluation harness, runtime harness with its own `clickhouse-rt`, collector fork).
2. **The Victoria stack is the fast specialized alternative** — per-tier query and ingest speed is best-in-class, and federated correlation is latency-competitive with ClickHouse; the cost is operating three stores plus merge logic, and losing single-SQL ad-hoc correlation.
3. **Druid** is a reasonable unified runner-up for query latency but pays ~4× on ingestion and lacks an OTEL collector exporter (Kafka path required).
4. **Doris** offered no advantage in any dimension on this dataset.
5. **OceanBase** should not be used for analytical log search (OLTP-shaped; 30–50× slower on `LIKE '%error%'` scans).
6. **Loki** was superseded within this benchmark by VictoriaLogs (same tier, faster on the same queries in the June rounds).

## 5. Limitations

- All recorded runs are at the **base dataset scale** (~1.7k rows); the 50k-row `SCALE_TO` mode exists but no scaled run is in `out/`. Latency ratios at 50k may differ (especially OceanBase's log scans, which the README expects to degrade further).
- Single-run latencies (no `--query-runs` medians recorded), Docker-resident backends on shared hardware.
- The correlation tier is result-shape-incomparable between SQL and Victoria-federated paths (see §3.4).

## 6. Reproduce

```bash
cd research/telemetry_storage_backend
make up-compare                                   # Doris + ClickHouse + Druid + OceanBase
make bench-compare                                # base-scale comparison
make bench-compare SCALE_TO=50000                 # scaled run (not yet recorded in out/)
make down
```

Per-run artifacts: `out/storage_bench_compare_<timestamp>/` — `compare.html` (full report), `<backend>_queries.json` (latency + row counts), `ingest_stats/`. Run index: `out/rolling_index.html`.

---

*Sources: `out/storage_bench_compare_20260617_124105` (final 8-backend run) and prior runs listed in §2; methodology from `README.md`, `SCALING.md`, `docs/DATA_SOURCES.md`; comparability rules from `queries/COMPARABILITY.md`.*
