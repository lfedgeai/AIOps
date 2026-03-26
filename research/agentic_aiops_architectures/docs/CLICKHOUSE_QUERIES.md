# ClickHouse queries for OTEL telemetry (AIOps harness)

Reference for the **`otel`** database used by the OpenTelemetry Collector **ClickHouse exporter** and the agent tools in `code/tools/agent_tools.py`.

**HTTP interface:** `POST ${CLICKHOUSE_HTTP}` with raw SQL body (e.g. `curl -sS "${CLICKHOUSE_HTTP}" --data-binary @query.sql`).

**Official DDL / examples:** [clickhouseexporter README](https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/main/exporter/clickhouseexporter/README.md) and `internal/sqltemplates/` (source of truth for columns).

### Running SQL inside the ClickHouse pod (OpenShift / Kubernetes)

When the server runs in-cluster (e.g. **`otel-demo`** deployment **`clickhouse`** from `manifests/clickhouse.yaml`), you can run queries **from inside the container** with `clickhouse-client` (no port-forward needed).

**Interactive client** (multi-line SQL, `\q` to exit may vary; often Ctrl+D):

```bash
oc exec -it -n otel-demo deploy/clickhouse -- clickhouse-client --database otel
```

**One-shot query** (non-interactive):

```bash
oc exec -n otel-demo deploy/clickhouse -- clickhouse-client --database otel --query "SELECT count() FROM otel_logs"
```

**Multiline from your shell** (heredoc):

```bash
oc exec -i -n otel-demo deploy/clickhouse -- clickhouse-client --database otel --multiquery <<'SQL'
SHOW TABLES;
SELECT count() FROM otel_logs;
SQL
```

**HTTP from inside the same pod** (matches how probes/docs often test):

```bash
oc exec -n otel-demo deploy/clickhouse -- wget -qO- "http://127.0.0.1:8123/?query=SELECT%201"
```

**If you use a different namespace or workload name**, list targets first:

```bash
oc get pods -n otel-demo -l app=clickhouse
oc exec -it -n otel-demo <pod-name> -- clickhouse-client --database otel
```

Default user in the demo manifest is **`default`** with empty password unless you added secrets; the client uses the server’s local socket when run inside the pod.

---

## 1. Database & tables

Default database name from exporter config is usually **`otel`**. Fully qualified names:

| Table | Purpose |
|--------|---------|
| **`otel.otel_logs`** | OTLP logs |
| **`otel.otel_traces`** | OTLP traces (spans) |
| **`otel.otel_traces_trace_id_ts`** | TraceId → time bounds (speeds trace-scoped reads) |
| **`otel.otel_metrics_gauge`** | Gauge metric points |
| **`otel.otel_metrics_sum`** | Sum/counter metric points |
| **`otel.otel_metrics_histogram`** | Histograms (if used) |
| **`otel.otel_metrics_summary`** | Summaries (if used) |
| **`otel.otel_metrics_exp_histogram`** | Exponential histograms (if used) |

### 1.1 List tables and row counts

```sql
SHOW TABLES FROM otel;
```

```sql
SELECT
  name,
  total_rows,
  formatReadableSize(total_bytes) AS size
FROM system.tables
WHERE database = 'otel'
ORDER BY name;
```

### 1.2 Describe schema (run per table)

```sql
DESCRIBE TABLE otel.otel_logs;
DESCRIBE TABLE otel.otel_traces;
DESCRIBE TABLE otel.otel_metrics_gauge;
DESCRIBE TABLE otel.otel_metrics_sum;
```

**Note:** Newer collector builds may use **JSON** types for attributes (`LogAttributes`, `ResourceAttributes`, …) instead of `Map(LowCardinality(String), String)`. If `DESCRIBE` shows JSON, use `LogAttributes.key` / `JSONExtract*` instead of `LogAttributes['key']` in predicates.

---

## 2. Canonical column summary (Map-style schema)

Summarized from upstream `sqltemplates` (your deployment may differ slightly).

### `otel.otel_logs`

| Column | Role |
|--------|------|
| `Timestamp` | `DateTime64(9)` — log time |
| `TraceId`, `SpanId`, `TraceFlags` | **Correlation** to traces |
| `SeverityText`, `SeverityNumber` | Level |
| `ServiceName` | `service.name` |
| `Body` | Message |
| `ResourceAttributes`, `ScopeAttributes`, `LogAttributes` | Maps (or JSON) — extra dimensions |

### `otel.otel_traces`

| Column | Role |
|--------|------|
| `Timestamp` | Span start |
| `TraceId`, `SpanId`, `ParentSpanId` | **Correlation** |
| `SpanName`, `SpanKind`, `ServiceName` | Span identity |
| `Duration` | **Nanoseconds** (divide by `1e6` for ms) |
| `StatusCode`, `StatusMessage` | Error status |
| `ResourceAttributes`, `SpanAttributes` | Maps (or JSON) |
| `Events`, `Links` | Nested structures |

### `otel.otel_metrics_gauge` / `otel.otel_metrics_sum`

| Column | Role |
|--------|------|
| `TimeUnix` | `DateTime64(9)` — sample time (use for time windows) |
| `StartTimeUnix` | Start of cumulative window (sum) |
| `MetricName`, `Value` | Name and value |
| `ServiceName` | Often populated from resource |
| `Attributes` | Labels (may include `service.name`, `service_name`, …) |
| `Exemplars` | Nested — can carry `TraceId` / `SpanId` when exporters attach them |

---

## 3. Health & volume

```sql
SELECT count() FROM otel.otel_logs;
SELECT count() FROM otel.otel_traces;
SELECT count() FROM otel.otel_metrics_gauge;
SELECT count() FROM otel.otel_metrics_sum;
```

```sql
SELECT
  toStartOfHour(Timestamp) AS hour,
  count() AS logs
FROM otel.otel_logs
WHERE Timestamp >= now() - INTERVAL 24 HOUR
GROUP BY hour
ORDER BY hour;
```

```sql
SELECT ServiceName, count() AS spans
FROM otel.otel_traces
WHERE Timestamp >= now() - INTERVAL 1 HOUR
GROUP BY ServiceName
ORDER BY spans DESC
LIMIT 30;
```

---

## 4. Time windows

Harness / agents pass **ISO8601** timestamps; ClickHouse comparisons often use:

```sql
-- Example: window start (replace with your fault injection time)
-- Prefer DateTime64 for consistency with tables:
SELECT toDateTime64('2026-03-23 12:00:00', 3) AS window_start;
```

For **`otel_logs`** / **`otel_traces`** (`Timestamp` is `DateTime64`):

```sql
WHERE Timestamp >= toDateTime64('2026-03-23 12:00:00', 3)
```

For **metrics** (`TimeUnix`):

```sql
WHERE TimeUnix >= toDateTime64('2026-03-23 12:00:00', 3)
```

---

## 5. Logs queries

### 5.1 Recent logs (matches agent-style search)

```sql
SELECT
  formatDateTime(Timestamp, '%Y-%m-%d %H:%M:%S') AS ts,
  ServiceName,
  SeverityText,
  Body
FROM otel.otel_logs
WHERE Timestamp >= toDateTime64('2026-03-23 12:00:00', 3)
ORDER BY Timestamp DESC
LIMIT 50;
```

### 5.2 Case-insensitive substring in `Body` (same idea as `search_logs` in `agent_tools.py`)

```sql
SELECT
  formatDateTime(Timestamp, '%Y-%m-%d %H:%M:%S') AS ts,
  ServiceName,
  Body
FROM otel.otel_logs
WHERE Timestamp >= toDateTime64('2026-03-23 12:00:00', 3)
  AND positionCaseInsensitive(Body, 'error') > 0
ORDER BY Timestamp DESC
LIMIT 20;
```

### 5.3 Logs by service

```sql
SELECT Timestamp, SeverityText, Body
FROM otel.otel_logs
WHERE ServiceName = 'cart-service'
  AND Timestamp >= now() - INTERVAL 1 HOUR
ORDER BY Timestamp DESC
LIMIT 100;
```

### 5.4 Logs with map attribute (Map schema)

```sql
SELECT Timestamp, Body, LogAttributes
FROM otel.otel_logs
WHERE LogAttributes['deployment.environment'] = 'production'
  AND Timestamp >= now() - INTERVAL 1 HOUR
LIMIT 100;
```

### 5.5 Logs that carry a trace id (for correlation)

```sql
SELECT Timestamp, ServiceName, TraceId, SpanId, Body
FROM otel.otel_logs
WHERE TraceId != ''
  AND Timestamp >= toDateTime64('2026-03-23 12:00:00', 3)
ORDER BY Timestamp DESC
LIMIT 100;
```

---

## 6. Traces queries

### 6.1 Slow spans (duration in nanoseconds)

```sql
SELECT
  formatDateTime(Timestamp, '%Y-%m-%d %H:%M:%S') AS ts,
  ServiceName,
  SpanName,
  Duration / 1000000 AS duration_ms,
  StatusCode
FROM otel.otel_traces
WHERE Timestamp >= toDateTime64('2026-03-23 12:00:00', 3)
  AND Duration > 5 * 1e9   -- > 5 seconds
ORDER BY Duration DESC
LIMIT 50;
```

### 6.2 Error spans

```sql
SELECT
  Timestamp,
  TraceId,
  SpanId,
  ServiceName,
  SpanName,
  StatusCode,
  StatusMessage
FROM otel.otel_traces
WHERE StatusCode = 'Error'
  AND Timestamp >= now() - INTERVAL 1 HOUR
ORDER BY Timestamp DESC
LIMIT 100;
```

### 6.3 Traces by service (summary — same shape as `search_traces` tool)

```sql
SELECT
  formatDateTime(Timestamp, '%Y-%m-%d %H:%M:%S') AS ts,
  ServiceName,
  Duration / 1000000 AS duration_ms
FROM otel.otel_traces
WHERE ServiceName = 'cart-service'
  AND Timestamp >= toDateTime64('2026-03-23 12:00:00', 3)
ORDER BY Timestamp DESC
LIMIT 20;
```

### 6.4 Full trace by `TraceId` (optimized with `otel_traces_trace_id_ts` if present)

```sql
WITH
  '391dae938234560b16bb63f51501cb6f' AS trace_id,
  coalesce(
    (SELECT min(Start) FROM otel.otel_traces_trace_id_ts WHERE TraceId = trace_id),
    toDateTime('1970-01-01 00:00:00')
  ) AS start,
  coalesce(
    (SELECT max(End) + 1 FROM otel.otel_traces_trace_id_ts WHERE TraceId = trace_id),
    now() + INTERVAL 1 DAY
  ) AS end
SELECT
  Timestamp,
  TraceId,
  SpanId,
  ParentSpanId,
  SpanName,
  SpanKind,
  ServiceName,
  Duration,
  StatusCode,
  StatusMessage
FROM otel.otel_traces
WHERE TraceId = trace_id
  AND Timestamp >= start
  AND Timestamp <= end
ORDER BY Timestamp
LIMIT 500;
```

If **`otel_traces_trace_id_ts`** does not exist, simplify:

```sql
SELECT *
FROM otel.otel_traces
WHERE TraceId = '391dae938234560b16bb63f51501cb6f'
ORDER BY Timestamp
LIMIT 500;
```

---

## 7. Metrics queries

### 7.1 Recent gauge points

```sql
SELECT TimeUnix, MetricName, ServiceName, Attributes, Value
FROM otel.otel_metrics_gauge
WHERE TimeUnix >= toDateTime64('2026-03-23 12:00:00', 3)
ORDER BY TimeUnix DESC
LIMIT 50;
```

### 7.2 Recent sum/counter points

```sql
SELECT TimeUnix, MetricName, ServiceName, Attributes, Value
FROM otel.otel_metrics_sum
WHERE TimeUnix >= toDateTime64('2026-03-23 12:00:00', 3)
ORDER BY TimeUnix DESC
LIMIT 50;
```

### 7.3 Filter by metric name and attribute (Map schema)

```sql
SELECT TimeUnix, Value, Attributes
FROM otel.otel_metrics_sum
WHERE MetricName = 'calls'
  AND Attributes['service_name'] = 'featureflagservice'
LIMIT 100;
```

### 7.4 List distinct metric names seen recently

```sql
SELECT MetricName, count() AS n
FROM otel.otel_metrics_sum
WHERE TimeUnix >= now() - INTERVAL 6 HOUR
GROUP BY MetricName
ORDER BY n DESC
LIMIT 50;
```

---

## 8. Correlation queries

Correlation in OpenTelemetry is primarily via **`TraceId`** (and optionally **`SpanId`**).  
**Service + time** is a secondary lens when trace ids are missing on logs/metrics.

### 8.1 By **TraceId** — logs ∪ spans (same request path)

```sql
WITH '391dae938234560b16bb63f51501cb6f' AS tid
SELECT
  'span' AS signal,
  formatDateTime(Timestamp, '%Y-%m-%d %H:%M:%S.%f') AS ts,
  ServiceName,
  SpanName AS detail,
  TraceId,
  SpanId
FROM otel.otel_traces
WHERE TraceId = tid
UNION ALL
SELECT
  'log',
  formatDateTime(Timestamp, '%Y-%m-%d %H:%M:%S.%f'),
  ServiceName,
  Body,
  TraceId,
  SpanId
FROM otel.otel_logs
WHERE TraceId = tid
ORDER BY ts;
```

### 8.2 By **TraceId** — metrics (attributes or exemplars)

Many deployments **do not** put `trace_id` on every metric sample. When present on **Attributes** (Map):

```sql
SELECT TimeUnix, MetricName, ServiceName, Value, Attributes
FROM otel.otel_metrics_sum
WHERE TimeUnix >= toDateTime64('2026-03-23 12:00:00', 3)
  AND Attributes['trace_id'] = '391dae938234560b16bb63f51501cb6f'
LIMIT 200;
```

```sql
SELECT TimeUnix, MetricName, ServiceName, Value, Attributes
FROM otel.otel_metrics_gauge
WHERE TimeUnix >= toDateTime64('2026-03-23 12:00:00', 3)
  AND Attributes['trace_id'] = '391dae938234560b16bb63f51501cb6f'
LIMIT 200;
```

If your schema stores OTLP attribute keys with dots, try:

```sql
-- Example alternate key shapes
AND (Attributes['trace_id'] = '...' OR Attributes['trace.id'] = '...')
```

**Exemplars** (when populated) attach trace context to individual points; querying them depends on how the Nested `Exemplars` column is projected (varies by ClickHouse version). Inspect with:

```sql
SELECT MetricName, Exemplars.TraceId, Exemplars.SpanId, Exemplars.Value
FROM otel.otel_metrics_sum
WHERE length(Exemplars.TraceId) > 0
LIMIT 5;
```

### 8.3 By **timestamp ± window** and **ServiceName** (triangulation)

Use one anchor time `t0` (e.g. alert time or span start) and a small window (e.g. 30s):

```sql
WITH
  toDateTime64('2026-03-23 12:00:00', 3) AS t0,
  toIntervalSecond(30) AS half_window,
  'cart-service' AS svc
SELECT 'log' AS kind, Timestamp AS ts, ServiceName, Body AS detail
FROM otel.otel_logs
WHERE ServiceName = svc
  AND Timestamp BETWEEN t0 - half_window AND t0 + half_window
UNION ALL
SELECT 'span', Timestamp, ServiceName, SpanName
FROM otel.otel_traces
WHERE ServiceName = svc
  AND Timestamp BETWEEN t0 - half_window AND t0 + half_window
UNION ALL
SELECT 'metric', TimeUnix, ServiceName, concat(MetricName, '=', toString(Value))
FROM otel.otel_metrics_sum
WHERE ServiceName = svc
  AND TimeUnix BETWEEN t0 - half_window AND t0 + half_window
ORDER BY ts
LIMIT 500;
```

### 8.4 By **timestamp** — join logs and spans on **TraceId** when both exist in window

```sql
SELECT
  l.Timestamp AS log_ts,
  t.Timestamp AS span_ts,
  l.ServiceName,
  l.TraceId,
  l.Body,
  t.SpanName,
  t.Duration / 1e6 AS duration_ms
FROM otel.otel_logs AS l
INNER JOIN otel.otel_traces AS t
  ON l.TraceId = t.TraceId AND l.TraceId != ''
WHERE l.Timestamp >= toDateTime64('2026-03-23 12:00:00', 3)
  AND t.Timestamp >= toDateTime64('2026-03-23 12:00:00', 3)
ORDER BY l.Timestamp DESC
LIMIT 100;
```

### 8.5 Traces that have errors — then pull related logs by **TraceId**

```sql
WITH error_traces AS (
  SELECT DISTINCT TraceId
  FROM otel.otel_traces
  WHERE StatusCode = 'Error'
    AND Timestamp >= toDateTime64('2026-03-23 12:00:00', 3)
  LIMIT 50
)
SELECT l.Timestamp, l.ServiceName, l.TraceId, l.SeverityText, l.Body
FROM otel.otel_logs AS l
WHERE l.TraceId IN (SELECT TraceId FROM error_traces)
ORDER BY l.Timestamp DESC
LIMIT 200;
```

---

## 9. Queries embedded in this repo

| Location | What |
|----------|------|
| `code/tools/agent_tools.py` | `search_logs`, `search_traces`, `search_metrics_data`, arbitrary `query_clickhouse` |
| `scripts/local_demo.py` | Minimal `CREATE TABLE` for **local** demo only (subset of columns) |

---

## 10. Troubleshooting

- **Empty results:** Confirm `SELECT now(), max(Timestamp) FROM otel.otel_logs` vs your window start.
- **Unknown column:** Run `DESCRIBE TABLE` — exporter version may use **JSON** vs **Map** for attributes.
- **Permission / auth:** OpenShift ClickHouse may require credentials in the HTTP URL; local demo often has none.
- **Database name:** If not `otel`, use the `database:` value from the collector ClickHouse exporter config.
