# Doris: why the earlier numbers were wrong, and what we changed

## TL;DR

In the runs of 2026‑02‑27 and 2026‑03‑04 every Doris query took **0.42–0.81 s**, including a plain
`COUNT(*)` that ClickHouse answered in 5 ms. A flat floor that does not track query cost is a
measurement artifact, not engine speed.

1. **Measurement (the main problem).** Each Doris (and OceanBase) query was timed around
   `docker run --rm mysql:8 sh -lc "mysql ..."`, so every latency included a container start and a
   new MySQL connection. ClickHouse and Druid were timed around a direct HTTP call from the host.
2. **Schema.** The Doris tables had no partitions and no indexes. The comment said "partition by
   day", but there was no `PARTITION BY`. Log search used three `LIKE '%error%'` scans, and JSON
   attributes were parsed per row. Doris's main log-analytics features (inverted and full-text
   indexes, partition pruning) were not in use.

Results before and after are in [Results](#results).

## 1. Measurement fix

### What was wrong

```python
# runner/bench_compare.py (before)
cmd = ["docker", "run", "--rm", "--network", "tsb-net", "mysql:8", "sh", "-lc", sh_script]
t0 = time.time()
out = subprocess.run(cmd, ...)          # container create + start + mysql connect + query + teardown
dt = time.time() - t0
```

Compare that with ClickHouse:

```python
t0 = time.time()
r = requests.post(CH_HTTP, params={"query": sql})   # HTTP call only
dt = time.time() - t0
```

So each Doris number was roughly *container start‑up (~0.4–0.5 s) + query*. OceanBase had the same
problem, as did `get_data_volume` and the Doris-only `runner/bench.py`. Each query also ran exactly
once, cold, so the result mixed in first-run plan and metadata cache effects.

### What changed

| Change | Where |
|---|---|
| Doris and OceanBase queries run on a **persistent `pymysql` connection** to `:9030` / `:2881`, as a real application would | `runner/bench_compare.py` (`_mysql_conn`, `_run_mysql_query`), `runner/bench.py` |
| ClickHouse and Druid use one `requests.Session` (keep-alive), so every backend reuses its connection | `runner/bench_compare.py` |
| Timing uses `time.perf_counter()` and covers statement execution plus fetching the result set | same |
| **1 warm-up + N timed runs (default 5), median reported**; min and max are stored too | `bench_backend(..., runs, warmup)`, `--query-runs`, `--warmup-runs` |
| Doris schema and truncate go through pymysql instead of `echo '<sql>' \| mysql`, which broke on any single quote in the DDL | `_run_doris_sql`, `runner/bench.py:apply_schema` |
| `--backends doris,clickhouse,...` to run a subset. Previously Druid and OceanBase were mandatory. | `main()`, `Makefile BACKENDS=` |
| The ClickHouse readiness check used the port from `CLICKHOUSE_HTTP` (compose maps `28223`); it had been hard-coded to `8123` | `main()` |
| On `--all`, the Doris tables are dropped and recreated, because `CREATE TABLE IF NOT EXISTS` would silently keep an old layout | `apply_doris_schema(recreate=True)` |

## 2. Schema: every query shape gets an access path

Rule: each canonical query must be served by one of these:

- the **sort key prefix plus partition pruning** (time-range scans, `ORDER BY ts DESC LIMIT n`),
- an **inverted index** (equality or range on a column), or
- a **full-text inverted index** (token search in log text).

### Table changes (`schemas/doris.sql`)

| Change | Why |
|---|---|
| `AUTO PARTITION BY RANGE (date_trunc(ts, 'day'))` on logs, spans, metrics (partition column is `NOT NULL`) | Every query filters `ts >= NOW() - INTERVAL …`. Without partitions each one scanned all tablets. Daily partitions also make TTL and retention a partition drop. |
| `logs`: `DISTRIBUTED BY RANDOM` instead of `HASH(service)` | `service` is heavily skewed: in this dataset `frontend` is 50.5% of rows and `load-generator` 18.7%, so one of 8 hash buckets holds at least half the data and a single scanner thread dominates. No log query does a point lookup by service. |
| `logs.message`: `INVERTED` index with `parser=unicode`, `lower_case=true`, `support_phrase=true` | Full-text search. `MATCH_ANY 'error'` replaces `LIKE '%error%' OR LIKE '%Error%' OR LIKE '%ERROR%'`, which was three substring scans over messages of up to 200 KB. |
| `INVERTED` indexes on `logs.level`, `logs.service`, `logs.trace_id`, `spans.trace_id`, `spans.service`, `metrics.metric_name` | Equality filters and trace lookups. |
| `spans.duration_ms`: `INVERTED` (numeric, BKD) | Range filters `duration_ms > 500` and `> 5000`. |
| New `spans.http_status_code INT`, promoted from `attributes["http.status_code"]` at load time, with an `INVERTED` index | The error-span query parsed JSON on every row. A typed, indexed column turns it into an index range lookup. |
| New `metrics.trace_id VARCHAR`, promoted from `labels.trace_id` at load time, with an `INVERTED` index | `correlation_by_trace_id` joined on `json_extract_string(labels,'$.trace_id')`, computed per row. It now joins on a plain column. |
| `compaction_policy=time_series` (logs, spans) | Recommended for append-only observability tables: less write amplification during compaction. |
| Compression left at the default `lz4`, **not** `zstd` | Tested. With `zstd` on logs, `logs_recent` went from 32 to 52 ms and `logs_search_error` from 99 to 157 ms, because both decompress the wide `message` column. Spans showed no measurable difference. Use `"compression" = "zstd"` when storage cost matters more than latency. |

The promoted columns are filled by `loaders/replay_doris.py` for batch loads and by
`runner/map_otlp_to_telemetry.py` for OTLP data.

### Query → access path

| Query | Filter / shape | Access path |
|---|---|---|
| `logs_recent` | `ORDER BY ts DESC LIMIT 100` | Sort key `(ts, service)`: Top‑N on the key column |
| `logs_errors_by_service` | `ts` range, `level = 'error' OR message MATCH_ANY 'error'` | Partition pruning; `idx_level` ∪ `idx_message` (full text) |
| `logs_search_error` | `ts` range, `message MATCH_ANY 'error'`, `ORDER BY ts DESC LIMIT 100` | Partition pruning; `idx_message` (full text); sort-key Top‑N |
| `trace_by_id` | `trace_id = ?` | `idx_trace_id` plus hash-bucket pruning on `trace_id` |
| `traces_slow_by_service` | `ts_start` range, `duration_ms > 500` | Partition pruning; `idx_duration_ms` (BKD range) |
| `spans_error_by_service` | `ts_start` range, `http_status_code >= 500 OR duration_ms > 5000` | Partition pruning; `idx_http_status_code` ∪ `idx_duration_ms` |
| `sla_latency_compliance` | `ts_start` range, aggregate over all rows in range | Partition pruning plus sort key (a full aggregate needs every row in range; no index helps) |
| `correlation_by_trace_id` | join spans ↔ logs ↔ metrics on `trace_id` | Plain-column equi-joins instead of `json_extract` per row. A join is bounded by its hash-join cost, not by an access path. |
| `correlation_by_timestamp` | per-minute buckets over a `ts` range | Partition pruning plus sort key |
| `metrics_p95_latency`, `metrics_by_service_hourly` | `ts` range, group by `metric_name` | Partition pruning plus sort key `(ts, metric_name)` |
| `data_volume` | `COUNT(*)` | Full scan by definition |

### Semantics to be aware of

- **Token vs substring match.** `MATCH_ANY 'error'` matches the *token* `error` in any case. It
  does not match `errors` or `ServerError`. ClickHouse's `positionCaseInsensitive(message,'error')`
  is a substring match, so counts can differ. Use `MATCH_PHRASE_PREFIX` or add more tokens if
  substring behaviour is needed.
- **ClickHouse bug fixed on the way.** `http.status_code` is a JSON *number* in the dataset (about
  35k spans, 968 of them 5xx). `JSONExtractString` returns `''` for numbers, so the old ClickHouse
  query never counted a 5xx span. It now uses `JSONExtractInt`, so both engines count the same rows.
- **Fairness.** The ClickHouse schema is unchanged here: no skip indexes, no partitioning, no
  `LowCardinality`. A fully fair comparison should give ClickHouse the equivalents too
  (`PARTITION BY toDate(ts)`, `tokenbf_v1` or `ngrambf_v1` on `message`, `bloom_filter` on
  `trace_id`, typed `http_status_code`).

## Also fixed: span queries were scanning nothing

Span timestamps come from the recorded dataset (2026‑02‑06/07), but every span query filters on
`NOW() - INTERVAL 1 DAY` or `30 DAY`. Once the data is more than 30 days old, `trace`, `SLA`,
`error-span` and correlation queries match 0 rows on **every** backend and time an empty scan.
Logs and metrics are already stamped at load time. `loaders/common.py`, which all loaders share,
now shifts span times so the newest span lands at load time, keeping relative spacing. Set
`TSB_REBASE_SPAN_TIME=0` to load the original timestamps.

## Also fixed: measure at merged steady state

Each Stream Load batch creates a rowset. With `compaction_policy=time_series`, Doris merges them
only after size or time thresholds, so right after a bulk load each span tablet held 18 rowsets.
Every query then merged them on the fly, which added 5–80 ms per query and at first made the new
schema look slower than the old one. ClickHouse has the same issue with unmerged parts. The runner
now runs a Doris full compaction and ClickHouse `OPTIMIZE TABLE … FINAL` before timing (skip with
`--no-compact`), so both engines are measured at steady state.

| Query (new schema, 2.5M spans) | before compaction | after compaction |
|---|---|---|
| `spans_error_by_service` | 24.2 ms | 12.3 ms |
| `logs_search_error` | 54.3 ms | 30.3 ms |
| `sla_latency_compliance` | 37.0 ms | 23.2 ms |
| `correlation_by_trace_id` | 260.6 ms | 181.3 ms |

## Results

**Setup.** Apple M3 Max, with Podman 5.8 running a VM with 8 vCPU and 14 GB. Doris 3.0.8
(`apache/doris:3.0.8-all`, a single FE+BE) and ClickHouse 24.3.18, both with default settings.
Values are median client latency in ms over 5 timed runs after 1 warm-up (10 after 2 for the
scaled set). Every configuration returned **identical row counts** for every query. Both engines
were compacted before timing.

- **A** = original harness (`docker run mysql:8` per query), old schema and queries
- **B** = new harness (persistent pymysql), old schema and queries
- **C** = new harness, new schema and queries

### Default dataset (278 logs, 77,014 spans, 1,108 metrics, the same data as the published runs)

| Query | ClickHouse | Doris A | Doris B | Doris C |
|---|---:|---:|---:|---:|
| correlation_by_timestamp | 11.4 | 235.5 | 23.3 | 24.0 |
| correlation_by_trace_id | 20.3 | 234.8 | 25.3 | 24.6 |
| data_volume | 4.8 | 232.0 | 14.0 | 16.3 |
| logs_errors_by_service | 34.8 | 246.8 | 22.5 | **11.8** |
| logs_recent | 16.6 | 254.6 | 31.2 | 49.0 ¹ |
| logs_search_error | 34.0 | 250.7 | 35.8 | **29.5** |
| metrics_by_service_hourly | 4.1 | 221.8 | 9.6 | 9.9 |
| metrics_p95_latency | 4.0 | 216.9 | 9.3 | 10.3 |
| sla_latency_compliance | 3.6 | 222.3 | 9.4 | 9.8 |
| spans_error_by_service | 20.5 | 221.3 | 12.7 | **11.7** |
| trace_by_id | 4.1 | 219.9 | 12.1 | 11.9 |
| traces_slow_by_service | 4.0 | 221.3 | 9.2 | 11.2 |

¹ Noisy on this run: the range was 29–64 ms. With 278 rows of up to 200 KB each, the query is
dominated by transferring `LEFT(message,150)` for 100 rows.

**Going from A to B removes about 210 ms from every Doris query.** That was container start-up in
the harness, not Doris. On the original machine the overhead was about 450 ms, which is why every
published Doris number sat between 0.42 and 0.81 s.

### Scaled in-database (×32 spans = 2,464,448; ×16 logs = 4,448, about 0.9 GB of text; ×32 metrics = 35,456)

The data was replicated with `INSERT … SELECT` identically in each engine. Copies get a suffixed
`trace_id`, so a trace lookup still returns one trace.

| Query | ClickHouse | Doris B (old schema) | Doris C (new schema) | C vs B |
|---|---:|---:|---:|---:|
| logs_errors_by_service | 94.1 | 97.4 | **13.3** | 7.3× faster |
| logs_search_error | 98.6 | 225.1 | **34.1** | 6.6× faster |
| spans_error_by_service | 66.5 | 42.5 | **13.4** | 3.2× faster |
| correlation_by_timestamp | 22.9 | 63.3 | 61.1 | ≈ |
| correlation_by_trace_id | 89.2 | 174.8 | 186.4 | ≈ (within noise; range 169–248) |
| logs_recent | 20.1 | 27.5 | 31.3 | ≈ |
| trace_by_id | 10.7 | 12.9 | 13.7 | ≈ |
| sla_latency_compliance | 6.1 | 21.9 | 23.5 | ≈ |
| traces_slow_by_service | 5.8 | 9.5 | 9.7 | ≈ |
| metrics_p95_latency | 4.8 | 11.1 | 10.5 | ≈ |
| metrics_by_service_hourly | 6.0 | 10.0 | 10.5 | ≈ |
| data_volume | 4.7 | 15.1 | 18.4 | ≈ |

### Reading the numbers

- The **index-served shapes** (full-text log search, error-span filters) are where the schema
  change pays off. They are 3–7× faster than the old schema and 3–7× faster than the untuned
  ClickHouse schema.
- **Scan and aggregate shapes** (SLA, per-minute correlation, metrics rollups) and short lookups
  are unchanged, within noise. At this size they are bound by a fixed per-query floor of about
  10 ms in Doris (see `metrics_p95_latency` and `data_volume`), compared with about 4–5 ms for ClickHouse.
  Partitioning and sort keys matter once the data covers many days. Here it covers one.
- `correlation_by_trace_id` is a three-way hash join over 2.4M spans. An index doesn't change
  its cost.
- `traces_slow_by_service` returns 0 rows on every engine: no span in the dataset has
  `duration_ms > 500` within the 1‑day window.

### Reproduce

```bash
docker compose up -d doris clickhouse
make bench-compare BACKENDS=doris,clickhouse QUERY_RUNS=5
```

On Podman, start Doris with `--pids-limit=-1`. The default 2048-PID cap kills the BE (`Cannot
fork`) under query load. Docker Desktop has no such cap.
