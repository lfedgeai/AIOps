## Why

Tier 1 tests Jev on seeded synthetic server statistics. That shows basic reasoning, but not whether Jev can make reliable AIOps decisions from real telemetry produced by a realistic distributed application. The LF Edge audience needs a testbed that anyone can clone and reproduce, built on a well-known application rather than made-up numbers.

## What Changes

- Add **Tier 2**: the official OpenTelemetry Demo (Astronomy Shop), pinned to release 3.1.0 and run with Docker Compose, becomes the system under test. We do not fork or modify it.
- Add a fault injector that toggles the demo's own flagd feature flags, plus load-generator control and a health check.
- Add a declarative scenario format and a catalogue of one healthy control and six fault scenarios, each gated by a check that the fault works in the pinned version.
- Add collectors for Prometheus, Jaeger and OpenSearch, and a normalized, size-budgeted `Observation` model with a scrubber that removes fault metadata (flag names and variants, `feature_flag` attributes).
- Add an `AIOpsEngine` interface with a `JevAdapter`, so another AIOps system can be tested later. Generalize `jev/client.py` so questions are a parameter. The default stays the Tier 1 questions.
- Add an independent evaluator that scores detection, localization, diagnosis and action separately as PASS, FAIL or UNKNOWN. There is no composite score. Add false-positive tracking for the healthy control.
- Add a 15-step experiment lifecycle with a mandatory fault reset and a return-to-baseline check, timing metrics (null when not measurable), and one JSON result per experiment.
- Add the developer workflow (`setup.sh`, `health-check.sh`, `run-experiment`, `run-all-experiments`), docs, and an updated README.
- Relabel the existing synthetic experiment as Tier 1. Its code, tests and behaviour do not change.
- **BREAKING (policy only):** `CLAUDE.md` and the v1 design say "no Docker Compose, Prometheus or Grafana". This change allows Compose for the Tier 2 testbed only. Kubernetes stays out of scope.

## Capabilities

### New Capabilities
- `otel-testbed`: pinned OpenTelemetry Demo checkout, Compose start/stop, health check of the demo and its observability stack, and recorded component versions.
- `fault-injection`: toggling and resetting flagd flags, workload control, and checking that the system returns to baseline.
- `tier2-scenarios`: declarative scenario definitions with hidden ground truth, plus the healthy control and fault catalogue.
- `tier2-observation`: collectors, the normalized `Observation`, the fixed telemetry set and size budget, and the anti-leak scrubber.
- `aiops-engine-adapter`: the `AIOpsEngine` interface, the `JevAdapter` with Tier 2 questions, and normalized decisions that keep the raw response.
- `tier2-evaluation`: independent per-dimension verdicts, false-positive rate, per-dimension aggregates and the timing block.
- `tier2-experiment-lifecycle`: the runner lifecycle, contamination guard, result persistence and batch runs.
- `tier2-reproducibility`: scripts, README, CLAUDE.md update, offline CI and the `docs/otel-demo-notes.md` file.

### Modified Capabilities
<!-- None. openspec/specs/ is empty because jev-ops-experiment-v1 is not yet archived, so the Jev client change (optional `questions`) is described under aiops-engine-adapter. -->

## Impact

- **Code:** new `tier2/` package, new `tests/tier2/`, root wrapper scripts. A small backward-compatible change to `jev/client.py`. Tier 1 files (`run.py`, `generator/`, `scenarios/`, `evaluation/`) are not changed.
- **Docs and policy:** `README.md`, `CLAUDE.md`, `docs/otel-demo-notes.md`, `.gitignore`, `.env.example`.
- **Dependencies:** one pinned YAML library for scenario files. The testbed needs Docker with about 6.7 GB of container memory limits for the full demo.
- **External systems:** the OpenTelemetry Demo 3.1.0 (cloned by `setup.sh`, not vendored) and the Jev API.
- **CI:** stays offline. No Compose runs in CI.
