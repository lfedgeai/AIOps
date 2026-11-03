## Context

Tier 1 (change `jev-ops-experiment-v1`) generates seeded `ServerState` observations and compares Jev's severity, action, human review and probable cause with hand-written expected values. The runner calls `client.decide(obs_dict)`, so a duck-typed seam exists, but `run.py` imports `JevClient` directly, and `evaluation/evaluator.py` assumes three fixed fields and one overall PASS. `jev/client.py` hard-codes `QUESTIONS` for server state. There is no Compose, no real telemetry and no localization scoring.

Tier 2 adds the OpenTelemetry Demo 3.1.0 (released 2026-09-18) as system under test. Facts checked against the 3.1.0 tag:

- Fault injection is flagd. `src/flagd/demo.flagd.json` holds 18 flags, all default `off`. They can be toggled by editing the file or through the flagd-ui API (`/feature/api/read` and `/feature/api/write`, which replaces the whole config).
- Envoy on `:8080` routes `/feature`, `/loadgen/`, `/jaeger/`, `/grafana/`. Prometheus is published on `:9090`. Most other ports are not fixed on the host.
- Metrics reach Prometheus by OTLP push. Span metrics are `traces_span_metrics_calls_total` and `traces_span_metrics_duration_milliseconds_bucket`, labelled `service_name`, `span_name`, `span_kind`, `status_code`. Traces are in Jaeger and logs in OpenSearch (`otel-logs`).
- Load comes from Locust (default 5 users, autostart).
- The full stack has container memory limits of about 6.7 GB.

Some facts were read through a summarizer and are not confirmed. They are gated in task group 1.

## Goals / Non-Goals

**Goals:**
- A reproducible local testbed: clone, `./setup.sh`, run an experiment.
- Real telemetry from the demo, with a known injected fault as hidden ground truth.
- Detection, localization, diagnosis and action scored separately, with explicit UNKNOWN.
- Jev behind an interface so another AIOps system can be tested without changing the harness.
- A healthy control to measure false positives.
- No change to Tier 1 behaviour or tests.

**Non-Goals:**
- Forking or modifying the demo, or writing our own app or telemetry generator.
- Kubernetes, LangGraph or any orchestration framework.
- Acting on Jev's recommendation. The harness only records it.
- A composite score.
- A production AIOps platform.

## Decisions

**D1. Additive `tier2/` package; Tier 1 is not moved.** Tier 1 stays at the repo root and is only relabelled in docs. The two tiers share only `jev/client.py`. Alternative: move Tier 1 under `tier1/`. Rejected: it churns working code, imports and tests for no gain.

**D2. Demo is a gitignored pinned clone, not a submodule or fork.** `tier2/testbed/setup.sh` clones the `3.1.0` tag into `tier2/testbed/opentelemetry-demo` and records the tag and the image versions in use in `tier2/testbed/VERSION`. The demo's `.env` at this tag says `IMAGE_VERSION=3.0.0` and `DEMO_VERSION=latest`, so a plain `make start` pulls `latest-*` images, not 3.1.0. The testbed therefore sets `DEMO_VERSION=3.1.0` as a shell variable on every compose call (it overrides the env file, so the checkout stays unmodified; `3.1.0-*` images exist on ghcr.io) and records the image versions that actually run. Alternatives: submodule (pulls the whole repo into git metadata, awkward for contributors) and fork (explicitly out of scope).

**D3. Faults are injected only through flagd.** `FaultInjector` sets one flag's variant and restores it. Two flag shapes exist in `demo.flagd.json` (confirmed from the file and from `src/flagd-ui/lib/flagd_ui/storage.ex`): plain flags are enabled by setting `defaultVariant`; a flag with a `targeting.if` rule (only `productCatalogFailure` at 3.1.0, matching `product_id == OLJCESPC7Z` with both branches `"off"`) is enabled by setting the matched branch `targeting.if[1]` and leaving `defaultVariant` alone, because targeting overrides the default. The injector must handle both shapes and read the effective state back. The consequence for `F001` is that only product `OLJCESPC7Z` fails, not the whole service, which the scenario rationale must say. Which path is used (file edit or flagd-ui write) is decided in task group 1 from what works reliably. Flag-name knowledge lives only in scenario files and the injector. Alternative: kill containers or add network faults. Rejected: not a verified, reproducible mechanism of the demo, and the plan forbids inventing fault mechanisms.

**D4. Scenarios are YAML files, one per experiment.** Fields: `experiment_id`, `application`, `fault {flag, variant}`, `target_service`, `fault_type`, `duration_s`, `workload`, `observation_window_s`, `baseline_window_s`, `expected {incident, affected_service, diagnosis, acceptable_actions}`. Loaded into frozen dataclasses and validated (enum membership, known service). A single pinned YAML library is added. Alternative: Python dataclasses like Tier 1. Rejected for Tier 2: declarative files make "add a new experiment" a one-file change that needs no code review of logic. Expected outcomes are hand-written, as in Tier 1.

**D5. Two separate types keep ground truth out of the prompt.** `GroundTruth` and `FaultSpec` (harness side) never appear in the `Observation` builder's signature. The builder takes only collector output. The request sent to Jev is built from `Observation` alone. Alternative: one scenario dict passed everywhere and filtered at the end. Rejected: filtering fails open, a type boundary fails closed.

**D6. Scrubber and leak test.** Collector output passes through a scrubber that drops any key or value containing `feature_flag`, replaces every flag name from the pinned flag file plus `flagd` in text, and replaces the run's experiment ID (word-boundary match, so hex trace IDs survive). Variants are not scrubbed as free text: `on` and `off` occur in ordinary words, and variants only appear on the wire as `feature_flag` attributes, which are dropped whole. Live telemetry confirmed the risk: payment's spans include `flagd.evaluation.v2.Service/ResolveFloat`, so flag evaluation traffic is visible as span names and edges. The collectors therefore also exclude `flagd`, `flagd-ui`, `telemetry-docs` and `load-generator` entirely. A test runs every scenario through the whole path with fixture telemetry that deliberately contains these strings and asserts that none appears in the state sent to the engine, and that the question set is identical for every scenario. Honest limit, to be documented: service names and error text are evidence, not leakage, and a model could in principle know the demo's flags from training data.

**D7. Fixed observation shape and budget.** Window start and end; per-service rows (request rate, error rate, p50 and p95 latency, change versus the baseline window); dependency edges derived from traces; deduplicated top-K error log lines per service; top-N failing trace summaries. Services are ranked by change versus baseline and capped, and every cap is a named constant. The same fields appear in every run. Jev has a 32k-token limit (`docs/jev-api-notes.md`), so the builder enforces a byte budget by trimming lowest-ranked items first and records what was trimmed in the result (not in the prompt).

**D8. `AIOpsEngine.decide(Observation) -> Decision`, `JevAdapter` wraps `JevClient`.** `JevClient.build_request` and `decide` gain an optional `questions` argument defaulting to the Tier 1 `QUESTIONS`, so Tier 1 requests are byte-identical and its tests do not change. Tier 2 questions: `incident_detected` (yes/no probability), `affected_service` (choice over the full, fixed list of demo services), `diagnosis` (choice), `recommended_action` (choice). The decision keeps `raw_response` always. Parsing stays strict. An out-of-set answer is `invalid`, a failed call is `error`, and both give UNKNOWN verdicts. `confidence` is the model's probability when present, else `null`.

**D9. Tier 2 vocabularies are separate from Tier 1.** Diagnosis: `application_failure`, `latency_degradation`, `dependency_unreachable`, `database_contention`, `resource_exhaustion`, `unknown`, `none`. Action: `investigate`, `restart_service`, `rollback_change`, `scale_up`, `escalate`, `no_action`. They are defined in `tier2/` and do not touch Tier 1 `CAUSES` or `ACTIONS`. The `unknown` diagnosis and `no_action` are always available so Jev is never forced to guess.

**D10. Independent evaluator, four verdicts, no composite.** `evaluate(decision, ground_truth)` is a pure function returning PASS, FAIL or UNKNOWN per dimension. It runs after the decision is stored and never receives the observation builder. A dimension is UNKNOWN only when the decision is `invalid` or `error`, or that field is missing. Otherwise every dimension is scored on its own: if Jev misses the incident, detection is FAIL and the other fields are still compared, so each dimension is reported honestly. Action is PASS if the answer is in `acceptable_actions`. For the healthy control, detection is PASS only when no incident is reported, and an incident is counted as a false positive. Aggregates are counts per dimension plus the false-positive rate, never one blended number.

**D11. Lifecycle in one runner with `try/finally` reset.** Steps follow the 15-step lifecycle. The reset runs in `finally`, so an engine error cannot leave a fault on. After reset, a baseline check compares error rate and latency with the baseline window. If it fails, the run is marked `contaminated` and the batch stops, because later experiments would be invalid. Manifestation is detected harness-side with a Prometheus threshold against the baseline and is never shown to Jev. If it is not detected, `t_manifest` is `null` and the run continues.

**D12. Timing is recorded, never invented.** `started_at`, `t_inject`, `t_manifest`, `t_observation`, `t_decision`, `detection_seconds`, `diagnosis_seconds`, `total_seconds`. `detection_seconds` is `t_decision - t_inject` only when `t_manifest` was measured, otherwise `null`.

**D13. Results are one JSON file per experiment plus a batch summary.** `results/tier2/<run-id>/<experiment_id>.json` holds fault, ground truth, normalized Jev output, raw response, the four verdicts, timing, trim notes, tool and demo versions, and git commit. Ground truth is stored in the result but was never in the request. `results/*` stays git-ignored.

**D14. Collectors are thin injectable HTTP clients.** `prometheus.py`, `jaeger.py`, `opensearch.py` take a `get` function the way `JevClient` takes `post`, so tests run offline with fixtures. Hosts are reached through Envoy on `:8080` and Prometheus on `:9090`. Anything else is verified in task group 1 first.

**D15. CI stays offline.** `./run-all.sh --check` additionally runs the Tier 2 unit tests. CI never starts Compose. Live runs are manual.

**D16. Policy update.** `CLAUDE.md` and the v1 non-goals are amended: Compose is allowed for the Tier 2 testbed only, Kubernetes remains out, and the new rules (no ground truth in observations, scrubber, separate verdicts) are added next to the existing ones.

## Risks / Trade-offs

- [A planned fault does not work in 3.1.0, for example `productCatalogFailure` targeting looked like it returned "off" in both branches] → verify each flag against the raw flag file and by observing errors before writing the scenario. Drop or replace scenarios that fail the check, and document why.
- [flagd does not hot-reload, or the flagd-ui write behaves differently from the summary] → test the toggle path first. Fall back to the other path or to restarting flagd.
- [Full stack is too heavy for the host] → check RAM first. Use `make start-minimal` and drop the Kafka scenario.
- [Fault manifests slowly or noisily, so the window is wrong] → poll for manifestation with a timeout, make windows scenario parameters, and record `t_manifest` as `null` when unmeasured.
- [Live telemetry is not deterministic, so runs are only reproducible at the level of scenario and procedure] → say so in the README. Record versions, timestamps and the exact observation sent, and support repeat runs.
- [Observation exceeds Jev's token limit] → byte budget with ranked trimming, trim notes recorded.
- [Flag evaluation data leaks the fault] → scrubber plus leak test with deliberately contaminated fixtures.
- [Misleading scenarios depend on our own judgement of the "true" cause] → write the rationale next to each expected value, as Tier 1 does, and treat changing it as a change to the experiment.
- [Jev's non-determinism with small N] → support `--repeats`, report per-dimension counts and not percentages from tiny samples.
- [Leftover fault after a crash] → `setup.sh` and `health-check.sh` check that every flag is `off`, and a `reset-faults` command restores them.

## Migration Plan

1. Add `tier2/` and tests alongside Tier 1. No Tier 1 file moves.
2. Make the `questions` change in `jev/client.py` and run the existing tests to show Tier 1 is unchanged.
3. Update docs and policy.
4. Roll back by deleting `tier2/`, the wrapper scripts and the docs changes. The `questions` argument is backward compatible and can stay.

## Open Questions

- Which toggle path (file edit or flagd-ui write) is reliable? Decided in task group 1.
- What exactly is the ground truth for `paymentUnreachable`: `checkout` or `payment`? Decide after seeing the real telemetry in task group 1, then fix it in the scenario with a rationale.
- ~~Does `intlShippingSlowdown` get triggered by the Locust traffic?~~ No. It had no measurable effect and was replaced by `imageSlowLoad` for F005. A workload that sends non-US addresses might trigger it; not pursued.
- Should `kafkaQueueProblems` and `adHighCpu` be added as scenarios 7 and 8 once the first six work?
