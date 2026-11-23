# Storage benchmark: ClickHouse against Thanos + Loki + Tempo

Part 2 of the study in [#45](https://github.com/lfedgeai/AIOps/issues/45): retrieval and correlation, without an agent.

One OTel Collector sends the same OTLP data to both designs. `generate.py` makes the data from a seed, injects a payment fault, and computes the expected answer of each task. `bench.py` compares every answer with that expected answer, and it stops if a store lost data. A latency counts only for a correct answer.

```bash
./run.sh 20000 10 4 45     # traces, runs per round, rounds, seed
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

## Arms

- `clickhouse`: one SQL query. `clickhouse_no_skip_index`: the same query with skip indexes (bloom filter, minmax) not used.
- `broad_seq`: Loki selects all services, calls run one after another.
- `broad_par`: the same calls; calls that do not depend on each other run concurrently (measured).
- `scoped`: Tempo first, then Loki only for the services (and, for one trace, the time range) that Tempo returned. `service_window` covers all services, so it has no scoped arm. `error_drilldown` is a chain, so it has no parallel arm. In this corpus error logs exist only for services with error spans, so a scoped Loki query loses nothing; real data can be different.

## Results (2026-11-23)

Four sessions on one laptop: two scales, two seeds, a new stack for each session. Each session has 4 rounds of 10 timed runs for each arm, after 3 warm-up runs, and the design order alternates. The design that is not measured is paused. Each design has 4 GiB in total and the same four cores. No data loss and no wrong answers in any arm.

Total time including correlation in Python, p50 in ms, seed 45 / seed 46:

| Task | Traces | ClickHouse | ClickHouse, no skip index | Siloed broad, sequential | Siloed broad, parallel | Siloed scoped |
|---|---:|---:|---:|---:|---:|---:|
| trace_context | 20k | 5.7 / 5.7 | 7.5 / 6.5 | 60.8 / 54.0 | 58.6 / 51.1 | 37.0 / 32.7 |
| service_window | 20k | 8.1 / 8.0 | 7.8 / 8.0 | 92.4 / 69.5 | 57.1 / 48.4 | – |
| error_drilldown | 20k | 15.1 / 14.8 | 15.6 / 15.5 | 79.5 / 76.5 | – | 57.9 / 53.6 |
| bulk_join | 20k | 8.5 / 7.9 | 8.9 / 7.8 | 135.1 / 125.4 | 92.2 / 79.1 | 101.4 / 95.0 |
| trace_context | 100k | 8.4 / 9.4 | 10.1 / 12.1 | 182.6 / 199.4 | 179.8 / 191.8 | 70.3 / 59.7 |
| service_window | 100k | 14.2 / 12.8 | 12.4 / 13.1 | 173.3 / 173.1 | 159.4 / 166.2 | – |
| error_drilldown | 100k | 20.6 / 23.7 | 21.1 / 23.5 | 544.0 / 549.7 | – | 332.2 / 341.5 |
| bulk_join | 100k | 15.3 / 16.6 | 15.6 / 16.8 | 555.9 / 694.0 | 360.3 / 488.3 | 478.4 / 555.4 |

20k traces = about 108,000 spans and 181,000 logs. 100k traces = about 537,000 spans and 902,000 logs. Full data, p95, and the p50 of each round are in `results/`.

What the data shows:

- ClickHouse was faster on every task. Against the best siloed arm of each task, the siloed design took 3.6 to 10.8 times as long at 20k traces and 6.3 to 29 times as long at 100k.
- The best siloed arm is not always the same: scoped for `trace_context` and `error_drilldown`, parallel for `bulk_join` and `service_window`.
- The skip indexes do not explain the difference. Without them ClickHouse was at most about 1.3 times slower (`trace_context`).
- The correlation code is a small part of the siloed time: below 3% on three tasks, and 4% to 14% on `bulk_join`. On `bulk_join` at 100k traces, Go takes 21 to 26 ms and Python 52 to 66 ms.
- The siloed responses are much larger: about 5 MB against 639 bytes for `bulk_join` at 100k traces. An agent cannot read that into its context. This is the main point to test in part 3.

These are latencies of 6 to 700 ms. They do not show an effect on MTTD or MTTR; an LLM call takes seconds. Part 3 measures that.

## Limits

- Synthetic data, one fault type, one node, local storage, warm caches, hot data (about 30 minutes). Thanos runs Receive and Query only: no object storage, Store Gateway, or Compactor. Metrics are few (17,280 points), so this is not a test of Thanos.
- Loki bloom filters are not measured. They are experimental, need separate containers, and in a test here the queries were answered by the ingester (`usedBloomFilters: false`, no chunks from storage), where blooms do not apply. A store-only query returned no rows, for a reason that was not found.
- The time ranges are backfilled. Tempo's `ingestion_time_range_slack` is raised to 3h, because the 2m default hides backfilled spans from time-filtered queries.
- `service_window` with sequential calls became faster during each session: from about 340 to about 170 ms after round 1 at 100k, and from about 90 to about 55 ms after 2 or 3 rounds at 20k. The parallel arm, which always runs after it, does not show this. The cause was not found. The p50 in the table mixes both states.
- The stores use mostly default settings and were not tuned by an expert. Image versions: Tempo 2.9.5 (not 3.x), Loki 3.6.22, Thanos 0.42.5, ClickHouse 26.3 (a newer 26.8 exists).
- The collector writes metric timestamps to ClickHouse with one-second precision (`DateTime`).
- Background load on the host: load average 3 to 7 on 16 threads.
- Ingest rate, CPU, memory and disk (part 1), and MLflow logging, are not done.

## Why not `telemetry_storage_backend`

- Its static data does not correlate. In the sampled files with full logs, 2 to 9 of more than 9,300 log trace IDs match a span, and the metrics are one snapshot for each file. Its loader also gives a whole log file one trace ID.
- The runner for the federated Victoria design is not in the repository (only its query files).
- [#40](https://github.com/lfedgeai/AIOps/pull/40) rewrites `bench_compare.py`.
