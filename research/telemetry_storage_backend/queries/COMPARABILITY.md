# Query comparability across backends

The benchmark reuses **query names** across backends, but not every backend can run every query, and not every query returns the same result shape. The report column **same result shape?** flags when row counts diverge; the **faster (comparable)** column only ranks backends that agree on row count.

## Tiers

| Tier | Queries | SQL backends | Loki | VictoriaLogs | VictoriaMetrics | VictoriaTraces |
|------|---------|--------------|------|--------------|-----------------|----------------|
| **logs** | `logs_*` | yes | yes | yes | — | — |
| **metrics** | `metrics_*` | yes | — | — | yes | — |
| **traces** | `spans_*`, `trace_*`, `traces_*`, `sla_*` | yes | — | — | — | yes |
| **correlation** | `correlation_*` | yes (3-way join) | — | — | — | **Victoria (federated)**: VL + VT + VM queries + Python merge |
| **meta** | `data_volume` | yes (3 counts) | — | 1 count each | 1 count each | 1 count each |

## Making queries more comparable

### 1. Match result row counts

- Use the same `LIMIT` everywhere (20 for aggregates, 100 for scans).
- Loki `query_range` limit was raised to **100** to match SQL `logs_recent` / `logs_search_error`.
- VictoriaMetrics `metrics_p95_latency` uses `quantile by (__name__)` so row count tracks ClickHouse `GROUP BY metric_name`.
- VictoriaTraces `trace_by_id` uses a **sample trace_id** from the same telemetry files SQL uses (`TRACE_ID` placeholder substituted at bench time).

### 2. Match filters and time windows

- Align lookback windows: 30d logs/traces, 7d hourly metrics, 365d p95, 1d slow spans.
- Use equivalent predicates: `duration_ms > 500` ↔ `"span_attr:duration_ms":>500`, HTTP 500 ↔ `"span_attr:http.status_code":>=500`.
- OTLP ingest now adds **`duration_ms`** as a span attribute so VictoriaTraces can filter like SQL.

### 3. Separate signal stores vs unified SQL

Victoria **correlation** uses split query files under `queries/victoriametrics/`:

- `correlation_by_trace_id.{spans.traceql,logs.logql,metrics.metricsql}`
- `correlation_by_timestamp.{spans.traceql,logs.logql,metrics.metricsql}`

The harness merges on `trace_id` or `(service, time_bucket)` and reports as **Victoria (federated)** (wall-clock for 3 API calls + merge).

### 4. Scale and warmup

At ~500–2k rows, latencies are noise. For meaningful comparisons:

```bash
make up-compare
SCALE_TO=50000 BATCH=5000 make bench-compare
```

Run twice and discard the first run if you need stable numbers.

### 5. Future improvements

- **Correlation**: run three tier-local queries and merge in the harness, or accept correlation as SQL-only.
- **metrics_by_service_hourly**: MetricsQL lacks SQL-style `GROUP BY hour`; current query is an approximate rollup — compare SQL backends among themselves.
- **Loki**: large log lines may hit gRPC payload limits; if so, reduce `limit` or truncate message size in the loader.
- **Validation pass**: optional post-query check that row counts match SQL reference before recording latency.

## Reading the report

| same result shape? | Meaning |
|--------------------|---------|
| **yes (N rows)** | All backends that ran this query returned N rows — latencies are comparable. |
| **partial — SQL N rows (...)** | SQL backends agree; VM/Loki diverge — winner is among SQL group only. |
| **no — ...** | Row counts differ — treat latency ranking as informational only. |
| **per-store** | `data_volume` — each backend counts its own store (by design). |
