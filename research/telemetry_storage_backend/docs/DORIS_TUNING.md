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
3. **Query and layout.** `correlation_by_trace_id` de-duplicated a raw three-way join with
   `COUNT(DISTINCT)`. It now aggregates per trace first, and spans and metrics are colocated on
   `trace_id`. JSON attributes are now `VARIANT`, so ad-hoc attribute filters are columnar.

On 2.46M spans, the new schema makes log search 8–20× faster, error-span filtering 3× faster and
trace correlation 1.6× faster. The other queries are unchanged: they're bound by a fixed Doris
per-query floor of about 15–20 ms, which is planning and scheduling rather than scanning. Results
are in [Results](#results).

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
| `logs`: `DISTRIBUTED BY RANDOM` instead of `HASH(service)` | `service` is heavily skewed (`frontend` is 50.5% of rows), and no log query looks up by service. We also tried `HASH(trace_id)`, to join the colocation group below: it spread logs evenly, but made `logs_search_error` 2.5× slower (42→115 ms), so logs stay RANDOM. Note that RANDOM writes each load job to one tablet, so this small dataset (a few loads) ends up in a single tablet; with continuous ingestion the loads spread across all 8. |
| `spans` and `metrics`: `DISTRIBUTED BY HASH(trace_id)` with `"colocate_with" = "telemetry_trace"` | Colocated tables have matching buckets, so a join on `trace_id` runs bucket-local with no data exchange. `correlation_by_trace_id` went from 248 to 180 ms. Metrics previously hashed on `metric_name`, and no metric query depends on that. **Caveat:** production OTLP metrics usually have no `trace_id`; if so, they'd all hash into one bucket, and metrics should stay on `HASH(metric_name)` outside the group. |
| 8 buckets per partition (unchanged) | Tested 8, 4 and 2 on the scaled data: the suite totalled 303, 482 and 571 ms. With 8 vCPUs, one tablet per core gives the most scan parallelism. Fewer tablets don't lower the per-query floor (see below). |
| `logs.message`: `INVERTED` index with `parser=unicode`, `lower_case=true`, `support_phrase=true` | Full-text search. `MATCH_ANY 'error'` replaces `LIKE '%error%' OR LIKE '%Error%' OR LIKE '%ERROR%'`, which was three substring scans over messages of up to 200 KB. |
| `INVERTED` indexes on `logs.level`, `logs.service`, `logs.trace_id`, `spans.trace_id`, `spans.service`, `metrics.metric_name` | Equality filters and trace lookups. |
| `spans.duration_ms`: `INVERTED` (numeric, BKD) | Range filters `duration_ms > 500` and `> 5000`. |
| New `spans.http_status_code INT`, promoted from `attributes["http.status_code"]` at load time, with an `INVERTED` index | The error-span query parsed JSON on every row. A typed, indexed column turns it into an index range lookup. |
| New `metrics.trace_id VARCHAR`, promoted from `labels.trace_id` at load time, with an `INVERTED` index | `correlation_by_trace_id` joined on `json_extract_string(labels,'$.trace_id')`, computed per row. It now joins on a plain column, which is also the colocation key. |
| `attributes`, `labels`, `attrs`: `VARIANT` instead of `JSON`, each with an `INVERTED` index | VARIANT stores every JSON key as its own typed sub-column, so filters on keys that were never promoted become columnar and indexed with no schema change. On 2.4M spans, `attributes['rpc.grpc.status_code'] != 0` went from 53 to 29 ms and `attributes['http.method'] = 'POST'` from 51 to 21 ms (vs `json_extract_string`), with identical results. Dotted keys stay literal: `attributes['http.status_code']`. We kept the promoted columns because querying the hot key via VARIANT was slower (22→37 ms) than the indexed `INT` column. |
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
| `correlation_by_trace_id` | join spans ↔ logs ↔ metrics on `trace_id` | Rewritten: each side is aggregated per trace before joining; spans ⋈ metrics is colocated (bucket-local); plain-column join keys instead of `json_extract` |
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

### Query rewrite: `correlation_by_trace_id`

The original query joined raw span, log and metric rows and then removed duplicates with
`COUNT(DISTINCT …)`. The rewrite aggregates each table per `trace_id` first and joins the small
results. The span side uses `GROUP BY` over a `SELECT DISTINCT`, which Doris runs as two plain
aggregations instead of a multi-phase distinct aggregation. Plain `COUNT(*)` over the same groups
took 42 ms against 154 ms for `COUNT(DISTINCT span_id)`. On 2.4M spans the rewrite alone went from
178 to 102 ms, with identical results.

### The per-query floor, and what doesn't lower it

Doris answers `SELECT 1` in 3.6 ms (ClickHouse: 2.7 ms), so the MySQL protocol and parser are not
the problem. Any query that touches a table, even one with 3 rows, costs about 16 ms. A query
profile shows the BE executes it in under 1 ms; the rest is FE planning (including about 5 ms of
table locking) and scheduling fragments onto the BE. That's why Doris's simplest queries sit at
about 10–20 ms, while ClickHouse's are at about 5 ms.

Things we measured that don't help:

- **Fewer buckets:** slower, as shown above.
- **`parallel_pipeline_task_num = 8`** (the default is half the cores): no change once there are
  8 tablets.
- **`zstd`:** slower, as shown above.

What does cut the floor is caching results or plans: the SQL cache, or server-side prepared
statements, which only apply to primary-key point lookups on unique-key tables. Those measure
cache hits rather than query execution, so this benchmark leaves them off, the same reason it
doesn't use pre-aggregated materialized views.

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

**Setup.** Apple M3 Max, with Podman 5.8 running a VM with 8 vCPU and 20 GB. Doris 3.0.8
(`apache/doris:3.0.8-all`, a single FE+BE) and ClickHouse 24.3.18, both with default settings.
Values are median client latency in ms over 5 timed runs after 1 warm-up (10 after 2 for the
scaled set). Every configuration returned **identical row counts** for every query, except where
noted. Both engines were compacted before timing. All numbers in this section come from a single
session.

- **A** = original harness (`docker run mysql:8` per query), old schema and queries
- **B** = new harness (persistent pymysql), old schema and queries
- **C** = new harness, new schema and queries

### Default dataset (278 logs, 77,014 spans, 1,108 metrics, the same data as the published runs)

| Query | ClickHouse | Doris A | Doris B | Doris C |
|---|---:|---:|---:|---:|
| correlation_by_timestamp | 17.1 | 356.0 | 38.0 | 48.3 |
| correlation_by_trace_id | 29.4 | 359.6 | 38.8 | 44.5 |
| data_volume | 7.3 | 308.7 | 22.5 | 25.2 |
| logs_errors_by_service | 53.5 | 342.9 | 27.9 | **19.4** |
| logs_recent | 10.7 | 389.9 | 84.6 | 68.4 |
| logs_search_error | 54.0 | 337.6 | 60.7 | **37.0** |
| metrics_by_service_hourly | 6.8 | 313.3 | 16.8 | 20.6 |
| metrics_p95_latency | 5.2 | 294.4 | 15.7 | 22.5 |
| sla_latency_compliance | 6.0 | 285.8 | 15.1 | 23.1 |
| spans_error_by_service | 28.9 | 298.8 | 21.8 | 26.7 |
| trace_by_id | 7.6 | 298.8 | 18.4 | 23.4 |
| traces_slow_by_service | 6.0 | 291.1 | 15.2 | 18.6 |

**Going from A to B removes 271–321 ms from every Doris query.** That was container start-up in
the harness, not Doris. On the original machine the overhead was about 450 ms, which is why every
published Doris number sat between 0.42 and 0.81 s.

At this size the new schema only pays off on the log-search queries. The simple span and metric
queries are a few ms slower in C, from the extra per-query cost of VARIANT sub-columns, more
indexes and the colocation group, which shows up only when a query does almost no work.

### Scaled in-database (×32 spans = 2,464,448; ×16 logs = 4,448, about 0.9 GB of text; ×32 metrics = 35,456)

The data was replicated with `INSERT … SELECT` identically in each engine. Copies get a suffixed
`trace_id`, so a trace lookup still returns one trace.

| Query | ClickHouse | Doris B (old schema) | Doris C (new schema) | C vs B |
|---|---:|---:|---:|---:|
| logs_errors_by_service | 146.8 | 341.7 | **16.8** | 20.4× faster |
| logs_search_error | 134.4 | 433.0 | **51.3** | 8.4× faster |
| spans_error_by_service | 95.9 | 58.3 | **19.6** | 3.0× faster |
| correlation_by_trace_id | 111.6 | 233.6 | **147.1** | 1.6× faster |
| correlation_by_timestamp | 28.1 | 89.1 | 79.3 | ≈ |
| data_volume | 6.3 | 23.5 | 23.1 | ≈ |
| logs_recent | 37.5 | 35.5 | 36.4 | ≈ |
| metrics_by_service_hourly | 8.8 | 18.3 | 17.6 | ≈ |
| metrics_p95_latency | 6.4 | 22.0 | 17.5 | 1.3× faster |
| sla_latency_compliance | 7.5 | 33.5 | 30.7 | ≈ |
| trace_by_id | 11.6 | 22.3 | 18.8 | ≈ |
| traces_slow_by_service | 7.3 | 22.8 | 14.7 | 1.6× faster |

`trace_by_id` returned 16 spans in C and 4 in B. The query picks "any trace" with `LIMIT 1`, and
colocation changed the physical row order, so it fetched a larger trace.

### Reading the numbers

- The **index-served shapes** (full-text log search, error-span filters) are where the schema
  change pays off most: 3–20× faster than the old schema. B's log-search numbers vary widely
  from run to run (see min/max in `out/doris_tuning_20260929/ab_results.json`), so read those
  ratios as "an order of magnitude", not as exact values.
- **`correlation_by_trace_id`** is 1.6× faster, from the query rewrite plus colocating spans and
  metrics on `trace_id`.
- **Scan and aggregate shapes** (SLA, per-minute correlation, metrics rollups) and short lookups
  are unchanged or slightly faster. They're bound by the per-query floor described above.
  Partitioning and sort keys matter once the data covers many days. Here it covers one.
- `traces_slow_by_service` returns 0 rows on every engine: no span in the dataset has
  `duration_ms > 500` within the 1‑day window.

### Reproduce

```bash
docker compose up -d doris clickhouse
make bench-compare BACKENDS=doris,clickhouse QUERY_RUNS=5
```

On Podman, start Doris with `--pids-limit=-1`. The default 2048-PID cap kills the BE (`Cannot
fork`) under query load. Docker Desktop has no such cap.

Also pin the container IP, e.g. `--ip 10.89.0.50` on the `tsb-net` network. The all-in-one image
identifies its FE by IP, so if a restart assigns a new address the FE waits forever in `UNKNOWN`
state and never comes back.
