# OpenTelemetry Demo notes (Tier 2)

Verified against the OpenTelemetry Demo 3.1.0 (released 2026-09-18) running under Docker Compose on one host, on 2026-10-02. Everything below was observed, not taken from the demo's documentation. Re-check it if you change the pinned version.

## Versions that run

All demo images are `ghcr.io/open-telemetry/demo:3.1.0-*`. The demo's own `.env` at this tag says `IMAGE_VERSION=3.0.0` and `DEMO_VERSION=latest`, so a plain `make start` pulls `latest-*` images. This repo sets `DEMO_VERSION=3.1.0` as an environment variable on every Compose call instead of editing the checkout. Other components: flagd v0.16.0, OpenTelemetry Collector contrib 0.159.0, Prometheus v3.13.1, Jaeger 2.19.0, Grafana 13.1.0, Postgres 18.4, Valkey 9.0.4.

## Reaching things

| Thing | How |
|---|---|
| Shop, flagd-ui, Locust, Jaeger, Grafana | Envoy on `http://localhost:8080` (`/feature`, `/loadgen/`, `/jaeger/ui/`, `/grafana/`) |
| Prometheus | `http://localhost:9090` (fixed host port) |
| OpenSearch | random host port. Ask Docker: `docker port opensearch 9200` (`TIER2_OPENSEARCH_URL` overrides) |
| Other service ports | not fixed on the host. Use Envoy |

The full stack needs about 8 GB free (container limits sum to about 6.7 GB). Images are several GB to pull. Span metrics appear in Prometheus about 2 to 3 minutes after start.

## Fault injection

- flagd reads `src/flagd/demo.flagd.json` (18 flags, all `off`). It hot-reloads within about a second of the file changing (seen in flagd's log as `filepath event ... WRITE`).
- Toggle path used: flagd-ui API through Envoy. `GET /feature/api/read` returns `{"flags": ...}`. `POST /feature/api/write` takes `{"data": <whole config>}` and replaces the whole config, so every change is a read, a modify and a write. The write is atomic (temp file then rename).
- Flag shapes: most flags are enabled by setting `defaultVariant`. `productCatalogFailure` has `targeting.if: [{"==": [{"var": "product_id"}, "OLJCESPC7Z"]}, "off", "off"]`. Targeting overrides `defaultVariant`, so it is enabled by setting the matched branch `targeting.if[1]` to `"on"` (this is what flagd-ui does). Only product `OLJCESPC7Z` then fails.
- The flag file changes on disk. flagd-ui rewrites it as `{"flags": ...}`, dropping the `$schema` key and reformatting. After a reset every flag value is identical to the pinned file, but `git status` in the checkout still shows it modified. `git checkout src/flagd/demo.flagd.json` there restores it exactly.
- `paymentFailure` variant `"90%"` is actually 0.95.
- `flagd.evaluation.*` spans and `feature_flag` attributes appear in telemetry. They would reveal the fault, so the collectors exclude `flagd` and the scrubber removes them (see the README).

## Telemetry

- Metrics: `traces_span_metrics_calls_total` and `traces_span_metrics_duration_milliseconds_bucket`, labels `service_name`, `span_kind`, `span_name`, `status_code` (`STATUS_CODE_ERROR`, `STATUS_CODE_OK`, `STATUS_CODE_UNSET`). Tier 2 uses server and consumer spans only.
- SDKs export every 60 s, so `rate()` over a short range returns nothing. Use `increase()` over at least 180 s, evaluated at a fixed time.
- Traces: Jaeger API at `/jaeger/ui/api/services` and `/jaeger/ui/api/traces?service=...&tags={"error":"true"}` through Envoy.
- Logs: OpenSearch index `otel-logs*`, fields `resource.service.name`, `body`, `severity.number`, `@timestamp`. `severity.text` is inconsistent (`INFO`, `info`, `Information`, `error`, `Error`, `warn`, `WARN`, `Warning`), so filter on `severity.number` (13 and up warn, 17 and up error).
- Services seen: accounting, ad, cart, checkout, currency, email, flagd, flagd-ui, fraud-detection, frontend, frontend-proxy, frontend-web (browser), image-provider, load-generator, payment, product-catalog, quote, recommendation, shipping, telemetry-docs. The last four (flagd, flagd-ui, telemetry-docs, load-generator) are excluded from observations.

## Load

Locust through Envoy: `GET /loadgen/stats/requests` (`state`, `user_count`), `POST /loadgen/swarm` with `user_count` and `spawn_rate`. Default is 5 users. At 30 users the stack looked CPU-saturated: baseline p95 latencies of several seconds (frontend about 11 s, checkout about 9 s, product-catalog about 4 s). The scenarios use 10 users. After changing the user count, wait several minutes before taking a baseline. A baseline taken right after a change still carried the old load (frontend p95 15 s dropped to 1.9 s).

Baseline noise: a healthy system is not silent. `accounting` ("receive orders", a Kafka consumer span) shows a small error rate, `load-generator` shows errors, and some p95 values sit at the histogram ceiling. This is why the healthy control (F000) matters.

## Flags verified

Each was toggled on a live stack, with 150 s of baseline, 240 s of fault and 180 s `increase()` windows. "Error ratio" is errors over server and consumer spans for the service.

| Flag | Result | Used |
|---|---|---|
| `productCatalogFailure` | product-catalog errors 0 to about 3% (95 failing `GetProduct` calls); frontend about 4%; `/api/recommendations` fails. Only one product fails, so the signal is modest. | **F001** |
| `paymentFailure` 100% | `Charge` errors in payment, checkout `PlaceOrder`, frontend `/api/checkout`; about 2.7 failed orders in 2 minutes at 5 users. | **F003** |
| `paymentUnreachable` | checkout error ratio 0 to about 22%; payment receives **no calls** and shows no errors. | **F004** |
| `imageSlowLoad` 10sec | no errors; frontend-web p95 about 1.0 s to 9.1 s, frontend-proxy about 1.8 s to 4.6 s, image-provider unchanged (the delay is injected by Envoy). Only browser users show it. | **F005** |
| `productCatalogLockContention` | product-catalog error ratio 0 to about 18% (324 failing `astronomy-db` spans), p95 at the histogram ceiling, propagates to frontend, checkout, recommendation. | **F006** |

## Flags tried and dropped

| Flag | Why it is not a scenario |
|---|---|
| `cartFailure` 100% | Fires only inside `EmptyCart`, which only checkout calls, and checkout swallows the error (`PlaceOrder` stays OK). Cart error ratio 0.13%, about 1 error in 20 minutes at default load. In practice it is invisible. The retired ID **F002** is not reused. |
| `intlShippingSlowdown` 10sec | Shipping p95 78 ms to 90 ms; no propagation to checkout. It has no measurable effect under the default Locust traffic. A workload change might trigger it (non-US addresses). We did not try that. |
| `adHighCpu` | ad p95 3 ms to 6 ms; no visible effect. |

Not tried: `adFailure`, `adManualGc`, `emailMemoryLeak`, `recommendationCacheFailure`, `kafkaQueueProblems`, `loadGeneratorFloodHomepage`, `failedReadinessProbe`, the AI-agent flags and `emitRawPii`. `kafkaQueueProblems` and `recommendationCacheFailure` are the next candidates.

## Ground-truth judgement calls

- F004 expects `checkout` as the affected service (the defect is in checkout's connection to payment), diagnosis `dependency_unreachable`. An engineer could reasonably say `payment`. Read results with that in mind.
- F005 expects `frontend-proxy` (where the delay is injected). The browser-facing symptom is in `frontend-web`.
- F001 is a one-product failure, not a whole-service failure.
