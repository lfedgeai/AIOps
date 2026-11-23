# Storage benchmark: ClickHouse against Thanos + Loki + Tempo

Part 2 of the study in [#45](https://github.com/lfedgeai/AIOps/issues/45): retrieval and correlation, without an agent.

One OTel Collector sends the same OTLP data to both designs. `generate.py` makes the data from a seed, injects a payment fault, and computes the expected answer of each task. `bench.py` compares every answer with that expected answer, and it stops if a store lost data. A latency counts only for a correct answer.

```bash
./run.sh 20000 30     # traces, timed runs per task
```

Needs Docker, Python 3 with `requests`, and about 5 GB of free memory. Go runs in a container.

## Tasks

| Task | Question | Siloed calls |
|---|---|---|
| `trace_context` | All spans and logs of one trace | Tempo, Loki |
| `service_window` | Spans, error spans, error logs and error ratio per service and minute | Tempo, Loki, Thanos |
| `error_drilldown` | Worst service by metric, then its error traces, then their error logs | Thanos, then Tempo, then Loki (each step needs the previous one) |
| `bulk_join` | All error spans in five minutes joined with the error logs of the same trace | Tempo, Loki |

Siloed time is recorded as backend time (HTTP) and correlation time (JSON decode and join). Python and Go use the same algorithm on the same response bodies.

## Preliminary results (2026-11-23)

One session on a laptop, 30 timed runs after 3 warm-up runs, p50 in ms. Each design has 4 GiB of memory in total and the same four cores. The design that is not measured is paused. No data loss and no wrong answers at either scale.

| Task | Traces | ClickHouse total | Siloed backend, sequential | Siloed backend, parallel (derived) | Correlate Python | Correlate Go | Response bytes, ClickHouse | Response bytes, siloed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trace_context | 20k | 9.2 | 59.6 | 55.7 | 0.11 | 0.09 | 1,889 | 12,121 |
| service_window | 20k | 9.9 | 88.5 | 51.1 | 0.30 | 0.12 | 2,848 | 19,397 |
| error_drilldown | 20k | 15.0 | 78.6 | 78.6 | 0.64 | 0.50 | 3,533 | 130,177 |
| bulk_join | 20k | 7.6 | 127.0 | 72.3 | 6.32 | 5.26 | 629 | 945,346 |
| trace_context | 100k | 5.8 | 200.6 | 196.8 | 0.14 | 0.09 | 1,889 | 12,172 |
| service_window | 100k | 11.3 | 316.8 | 164.0 | 0.31 | 0.17 | 2,947 | 18,622 |
| error_drilldown | 100k | 21.5 | 505.4 | 505.4 | 3.79 | 3.10 | 17,197 | 679,251 |
| bulk_join | 100k | 16.2 | 487.0 | 254.9 | 47.67 | 21.10 | 639 | 4,843,662 |

20k traces = 107,552 spans and 180,564 logs. 100k traces = 536,674 spans and 901,693 logs. Both have 17,280 metric points.

What the data shows:

- ClickHouse was faster on every task and stayed under 22 ms. From 20k to 100k traces, its latency grew at most 2.1 times (`bulk_join`); the siloed latency grew 3.4 to 6.4 times.
- The correlation code is a small part of the siloed time: at most 9% (`bulk_join`, 100k). Go against Python matters only for the large join (21 ms against 48 ms).
- The siloed responses are much larger, up to 4.8 MB against 639 bytes. An agent cannot read that into its context, so it must use narrower queries. This is the main point to test in part 3.

## Limits

- The Loki result depends on the label selector. A trace ID lookup over all services took about 170 ms; with the service known it took about 14 ms (100k traces, ad hoc check with both designs running). The tasks use the broad selector, because the caller does not know the services. Loki's bloom filters are not enabled; the ClickHouse exporter creates bloom filter indexes by default.
- Synthetic data, one fault type, one node, data in local storage, and warm caches. Thanos runs Receive and Query only: no object storage, Store Gateway, or Compactor.
- The time ranges are backfilled. Tempo's `ingestion_time_range_slack` is raised to 3h, because the 2m default hides backfilled spans from time-filtered queries.
- The parallel column is computed from the slowest backend. It was not measured with concurrent requests. Each design ran as one block, ClickHouse first, in one session; the host had a load average of 3 to 7.
- Image versions: Tempo 2.9.5 (not 3.x), Loki 3.6.22, Thanos 0.42.5, ClickHouse 26.3 (a newer 26.8 exists).
- The collector writes metric timestamps to ClickHouse with one-second precision (`DateTime`).
- Ingest rate, CPU, memory and disk (part 1), and MLflow logging, are not done.

## Why not `telemetry_storage_backend`

- Its static data does not correlate. In the sampled files with full logs, 2 to 9 of more than 9,300 log trace IDs match a span, and the metrics are one snapshot for each file. Its loader also gives a whole log file one trace ID.
- The runner for the federated Victoria design is not in the repository (only its query files).
- [#40](https://github.com/lfedgeai/AIOps/pull/40) rewrites `bench_compare.py`.
