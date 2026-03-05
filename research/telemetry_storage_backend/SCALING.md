# Scaling the Telemetry Storage Benchmark

## Quick reference

| What to scale | How |
|---------------|-----|
| **Scale to 50k rows** | `make bench-compare SCALE_TO=50000` |
| **Ingestion latency** | Reported in compare.html: duration_s, rows, rows/sec per backend |
| **Real-live simulation** | `--streaming-batch 500` = smaller batches (more round-trips, stress backends) |
| **Data volume** | `BATCH=10000 make bench-compare` or `DATA_DIR=/path/to/data make bench-compare` |
| **Batch size** | `make bench-compare BATCH=10000` |
| **Query iteration** | Add `--query-runs N` (run each query N times, report median) |
| **Concurrent load** | Run multiple bench processes in parallel |
| **Infrastructure** | Docker resource limits, multi-BE Doris, Druid Historical count |

---

## 1. Data volume

### Scale to a target row count (e.g. 50,000)

The loaders cycle through the source data until each type reaches at least the target count:

```bash
make bench-compare SCALE_TO=50000
# Or directly:
python3 runner/bench_compare.py --all --data-dir telemetry_data --out out --scale-to 50000
```

This yields ≥50,000 logs, ≥50,000 spans, and ≥50,000 metrics (replicating/cycling the 278 log files, 556 trace files, and 556 metric files as needed).

### Use more input data

Point to a larger dataset (Makefile `DATA_DIR` defaults to `./telemetry_data`):

```bash
DATA_DIR=/path/to/large_telemetry_data make bench-compare
```

### Increase span extraction per file

The extractor caps spans per trace file (default 200). Override in `common.extract_span_rows` or add a loader arg:

```python
# loaders/common.py - extract_span_rows(..., max_per_file=200)
# For 10x more spans: max_per_file=2000
```

### Batch size

Larger batches reduce round-trips but increase memory. Default is 5000:

```bash
python3 loaders/replay_doris.py --data-dir telemetry_data --batch 10000
# bench-compare uses 5000 via replay_doris.py
```

---

## 2. Ingestion latency and real-live simulation

### Ingestion metrics (always reported)

When running `make bench-compare`, the report now includes:
- **Duration (s)**: Wall-clock time to ingest all data
- **Rows**: Total rows ingested (logs + spans + metrics)
- **Rows/sec**: Throughput

### Simulate real-live telemetry with smaller batches

Real telemetry arrives in small bursts. Use `--streaming-batch` to send data in smaller chunks:

```bash
# 50k rows in batches of 500 (100 batches) — more realistic for streaming
python3 runner/bench_compare.py --all --data-dir telemetry_data --out out --scale-to 50000 --streaming-batch 500
```

This stresses the ingestion path with more round-trips per second, closer to a live OTLP stream.

---

## 3. Query repetition (stats)

Run each query multiple times and report median/percentiles for stability:

```bash
# Example: add to bench_compare.py
python3 runner/bench_compare.py --all --data-dir telemetry_data --out out --query-runs 5
```

This would run each of the 11 queries 5 times per backend and report median latency.

---

## 4. Infrastructure scaling

### Docker resources

In `docker-compose.yml` / `docker-compose.druid.yml`:

```yaml
services:
  doris:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
```

### Doris: multiple BEs

Add more BE replicas for ingestion and query parallelism:

```yaml
# docker-compose.yml - add BE2, BE3, etc.
```

### ClickHouse

Single node by default. For scale, run ClickHouse cluster or use `clickhouse-keeper` for replication.

### Druid

`docker-compose.druid.yml` uses micro-quickstart (1 Historical). For scale:

- Add more Historicals
- Add more MiddleManagers for ingestion
- Tune `druid.processing.numThreads` and heap

---

## 5. Running at scale (CI / automation)

```bash
# Full comparison (truncate, load, query)
make bench-compare

# Queries only (reuse existing data, faster)
python3 runner/bench_compare.py --data-dir telemetry_data --out out
```

Set `DATA_DIR` and `OUT` via env:

```bash
export DATA_DIR=/mnt/telemetry_data
export OUT=/mnt/bench_out
make bench-compare
```

---

## 6. Loaders: current knobs

| Loader | Args | Effect |
|--------|------|--------|
| `replay_doris.py` (Doris) | `--batch 5000` | Rows per stream load |
| `replay_clickhouse.py` | `--batch 5000` | Rows per INSERT |
| `replay_druid.py` | `--batch 5000` | Rows per ingestion task |
| `replay_oceanbase.py` (OceanBase) | `--batch 5000` | Rows per INSERT (pymysql) |

`common.py` controls extraction:

- `max_per_file=200` in `extract_span_rows` → increase for more spans per trace file
- `raw[:200000]` (200KB) in `extract_log_rows` → message size cap; prevents huge Druid/OceanBase files at scale
- `rglob("logs_*.txt")` / `traces_*.json` / `metrics_*.json` → add more files to `DATA_DIR` for more data
