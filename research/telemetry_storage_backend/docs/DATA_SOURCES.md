# Data Sources for 50k Correlated Benchmarks

## Option A: Use Existing Data + Correlation ✅ Implemented

The harness uses `telemetry_data/` with **paired files** by scenario:
- `logs/test/logs_test01_adFailure_on.txt` ↔ `traces/test/traces_test01_adFailure_on.json` ↔ `metrics/test/metrics_test01_adFailure_on.json`

**Correlation** (implemented in `loaders/common.py`):
- **Logs**: Pull `trace_id`, `span_id`, `service` from the matching traces file (first span)
- **Metrics**: Explode `req_rate` / `cart_add_p95` into one row per (service, metric) with `labels: {service: "frontend"}`
- **Spans**: Already have trace_id, span_id, service

To run with **50k correlated rows**:
```bash
make bench-compare SCALE_TO=50000
```

**Pros**: No new tools, uses existing data, correlation by file pairing  
**Cons**: Cycling duplicates the same patterns

**Note**: Log messages are capped at 200KB per row (`loaders/common.py`) to avoid huge files when scaling. At 50k rows this keeps Druid/OceanBase ingestion tractable.

---

## Option B: telemetrygen (Official OTel)

[telemetrygen](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/cmd/telemetrygen) generates traces, metrics, and logs and sends **OTLP only** (gRPC/HTTP). It does not write files.

**Harness support**: The benchmark includes telemetrygen via `make bench-otlp`:

```bash
make up-otel
make bench-otlp                    # default: 1000 spans, 1000 logs, 1000 metrics
OTLP_COUNT=50000 make bench-otlp  # scale OTLP volume
```

Telemetrygen sends OTLP to the OpenTelemetry Collector, which exports to **Doris and ClickHouse**. OTLP data is then mapped into `telemetry.*` via `runner/map_otlp_to_telemetry.py`, so canonical queries run against batch + OTLP data combined. Druid is not in the OTLP path (no collector exporter).

**Standalone telemetrygen** (outside the harness):

```bash
go install github.com/open-telemetry/opentelemetry-collector-contrib/cmd/telemetrygen@latest
telemetrygen traces --traces 50000
telemetrygen logs --logs 50000
telemetrygen metrics --metrics 50000
```

**Correlation**: telemetrygen's traces, logs, and metrics are generated **separately** — they do not share trace_id by default. The harness maps OTLP into `telemetry.*` for querying; for custom correlation you would need a collector pipeline that links signals (e.g. trace context propagation).

**Pros**: Official, configurable (service names, span count)  
**Cons**: OTLP pipeline required; correlation needs custom collector config

---

## Option C: otelgen (Community)

[otelgen](https://github.com/krzko/otelgen) — community tool, similar OTLP output. Check if it supports file export or shared trace context across signals.

---

## Recommendation

For **50k correlated data** with your current harness:

1. **Implement correlation in extractors** (Option A) — pair logs/metrics with traces by file name, propagate trace_id/span_id/service.
2. Run `make bench-compare SCALE_TO=50000` to reach 50k rows.

For **net-new correlated data** from a generator, you’d need either:
- A collector pipeline: telemetrygen → Collector (file exporter) → files in your format, or
- A custom Python generator that emits `logs_*.txt`, `traces_*.json`, `metrics_*.json` with shared trace_ids and service names.
