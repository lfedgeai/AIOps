## 1. Verify the demo before coding against it

- [x] 1.1 Check host memory, CPU and Docker; decide full or minimal profile (spec: otel-testbed resource pre-check)
- [x] 1.2 Clone the OpenTelemetry Demo at tag 3.1.0 into `tier2/testbed/opentelemetry-demo` and start it with Compose; note the image versions that actually run
- [x] 1.3 Read the raw `src/flagd/demo.flagd.json` and confirm the variants and targeting of each planned flag, especially `productCatalogFailure` (both targeting branches are "off"; enabling it means setting `targeting.if[1]`, see design D3)
- [x] 1.4 Prove the toggle path: edit `defaultVariant` in the file, and separately try flagd-ui `POST /feature/api/write`; record which one hot-reloads and how long the effect takes
- [x] 1.5 For each planned flag, toggle it and observe the effect in Prometheus span metrics, Jaeger and OpenSearch; record error rate, latency and which service shows the symptom
- [x] 1.6 Confirm span-metric label names and values, the Jaeger path through Envoy, OpenSearch reachability from the host, and the Locust stats and swarm endpoints
- [x] 1.7 Confirm `intlShippingSlowdown` is triggered by the default Locust traffic. Result: it is not (no measurable effect); a workload change was not pursued and the flag was replaced by `imageSlowLoad` (see docs/otel-demo-notes.md)
- [x] 1.8 Decide the ground truth for `paymentUnreachable` (checkout or payment) from the real telemetry
- [x] 1.9 Write `docs/otel-demo-notes.md` with the verified flags, ports, PromQL examples, gotchas, and any dropped scenarios with reasons

## 2. Testbed scripts

- [x] 2.1 Create `tier2/testbed/setup.sh`: Docker and memory pre-check, clone at the pinned tag, pull images with `DEMO_VERSION=3.1.0` set (the demo's `.env` defaults to `latest`), write `tier2/testbed/VERSION` with the tag and image versions
- [x] 2.2 Add start and stop helpers using the demo's compose files (full and minimal profiles)
- [x] 2.3 Create `tier2/testbed/health-check.sh` covering frontend, Prometheus ready, span metrics present, Jaeger services, OpenSearch health, flag state all `off`, Locust running; exit non-zero and name the failing check
- [x] 2.4 Add root wrappers `setup.sh` and `health-check.sh`
- [x] 2.5 Add `.gitignore` entry for the demo checkout and update `.env.example` with new variables (hosts, ports, windows, timeouts)

## 3. Fault injection and workload

- [x] 3.1 Create `tier2/faults.py` with `FaultInjector.inject`, `reset`, read-back confirmation and previous-value restore, using the path chosen in 1.4 and handling both flag shapes (`defaultVariant`, and `targeting.if[1]` for `productCatalogFailure`)
- [x] 3.2 Add `reset-faults` (idempotent, all flags to `off`)
- [x] 3.3 Add flag-file validation: a scenario's flag and variant must exist in the pinned flag file
- [x] 3.4 Create `tier2/workload.py`: verify Locust is running, set user count
- [x] 3.5 Add manifestation polling (Prometheus deviation versus baseline with a timeout) and the return-to-baseline wait
- [x] 3.6 Unit tests with a fake flag store and fake Prometheus: inject/reset round trip, read-back mismatch, idempotent reset, manifestation and no-manifestation, recovered and not recovered

## 4. Scenarios

- [x] 4.1 Define the scenario dataclasses, the Tier 2 vocabularies and the demo service list in `tier2/scenarios.py`; add the pinned YAML dependency to `requirements.txt`
- [x] 4.2 Write the YAML loader and validation (required fields, vocabulary, known service, known flag, non-empty rationale)
- [x] 4.3 Write `F000.yaml` (healthy control)
- [x] 4.4 Write `F001` and `F003` (direct failure and transaction failure) with rationales; `F002` (cartFailure) was dropped after verification
- [x] 4.5 Write `F004` (misleading symptoms), `F005` (latency, now imageSlowLoad after intlShippingSlowdown failed verification) and `F006` (database lock contention), keeping only those that passed task group 1; dropped flags are recorded in the notes
- [x] 4.6 Add the scenario list command
- [x] 4.7 Unit tests: valid file, each kind of invalid file, control has no fault, catalogue coverage

## 5. Collectors and observation

- [x] 5.1 Create `tier2/collectors/prometheus.py` with injectable `get`; queries for request rate, error rate, p50 and p95 latency per service for a window
- [x] 5.2 Create `tier2/collectors/jaeger.py`: failing traces and dependency edges from spans
- [x] 5.3 Create `tier2/collectors/opensearch.py`: error logs per service for a window
- [x] 5.4 Define the `Observation` model in `tier2/observation.py`, with fixed field groups and named caps
- [x] 5.5 Implement the builder: baseline-relative change, ranking, dedup of log lines, caps and the byte budget with trim notes kept outside the request
- [x] 5.6 Implement the scrubber (`feature_flag` keys and values, flag names and variants from the flag file, experiment ID)
- [x] 5.7 Make the builder's signature take only collector output and the window
- [x] 5.8 Create fixture telemetry for a healthy and a faulty case, deliberately containing flag names, variants, an experiment ID and expected labels
- [x] 5.9 Unit tests: field groups, same shape across runs, budget and trim order, trim notes not in the request, scrubber cases, builder signature has no scenario type
- [x] 5.10 Leak test over every scenario: serialized Jev request contains none of the flag names, variants, experiment ID, diagnosis labels or acceptable actions

## 6. Engine interface and Jev adapter

- [x] 6.1 Add the optional `questions` argument to `JevClient.build_request` and `decide`, defaulting to the Tier 1 questions
- [x] 6.2 Confirm all existing Tier 1 tests pass unchanged and Tier 1 request bytes are identical
- [x] 6.3 Create `tier2/engine.py` with the `AIOpsEngine` protocol and the normalized `Decision`
- [x] 6.4 Implement `JevAdapter`: Tier 2 questions, strict parsing, `confidence` from probability or `null`, reasoning if available, raw response always kept
- [x] 6.5 Build the `affected_service` choice list from the demo service list so it is identical every run
- [x] 6.6 Unit tests with canned HTTP replies: valid, out-of-set service, missing field, transport error, no confidence, raw response preserved
- [x] 6.7 Test that no Tier 2 module other than the adapter imports `jev.client`
- [x] 6.8 Update `docs/jev-api-notes.md` if question handling differs from the notes; check the current Jev docs first, do not guess parameters

## 7. Evaluation

- [x] 7.1 Create `tier2/evaluate.py`: `GroundTruth` and pure `evaluate(decision, ground_truth)` returning PASS, FAIL or UNKNOWN for each dimension
- [x] 7.2 Handle the healthy control (detection, localization with no service named) and record false positives
- [x] 7.3 Implement the aggregate: counts per dimension per scenario and overall, rates over PASS and FAIL only with UNKNOWN shown, false-positive rate, latency and token stats (reuse Tier 1 `telemetry()` where it fits)
- [x] 7.4 Implement the timing calculation with `null` for anything unmeasured
- [x] 7.5 Unit tests: each dimension, mixed result, invalid and error give UNKNOWN, missing field, dimensions independent when detection fails, no composite field, false-positive rate, timing nulls

## 8. Runner and results

- [x] 8.1 Create `tier2/runner.py` implementing the lifecycle in order, with injectable testbed, injector, collectors, engine and clock
- [x] 8.2 Wrap injection through return-to-baseline in `try/finally` so the reset always runs
- [x] 8.3 Implement the contamination guard that stops the batch
- [x] 8.4 Make the control run the same path with injection skipped
- [x] 8.5 Write the result file `results/tier2/<run-id>/<experiment_id>.json` with all required fields, versions and git commit; never store the Jev key
- [x] 8.6 Add `run-experiment <ID>` and `run-all-experiments [--repeats N]` entry points, the batch summary, and root wrapper scripts
- [x] 8.7 Unit tests with fakes: step order, unhealthy start injects nothing, reset on engine error and on collector error, contamination stops the batch, control path, result contents, ground truth absent from the stored request, raw response kept on invalid, repeats, unknown ID, key not in result

## 9. Docs, policy and CI

- [x] 9.1 Rewrite the README with Tier 1 and Tier 2 sections covering the 11 required topics, the exact observation contents, and how to add an experiment
- [x] 9.2 Update `CLAUDE.md`: Compose allowed for Tier 2 only, Kubernetes still out, new Tier 2 rules and layout, commands table
- [x] 9.3 Amend the v1 design non-goals note so it points to this change for the Compose exception
- [x] 9.4 Extend `run-all.sh --check` to run the Tier 2 unit tests; confirm the CI workflow passes offline on Python 3.10 to 3.13
- [x] 9.5 Confirm `requirements.txt` has the pinned YAML library and `.env.example` documents every new variable

## 10. Live verification

- [x] 10.1 Run `./setup.sh` and `./health-check.sh` on a clean machine state and confirm the checks pass
- [x] 10.2 Run `F000` (healthy control) against real Jev, several times, and record the false-positive count
- [x] 10.3 Run one fault experiment against real Jev (the first scenario that passed task group 1)
- [x] 10.4 Open the stored Jev request payload from a real run and confirm it contains no flag, fault or ground-truth strings
- [x] 10.5 Confirm reset: the flag is `off` after the run and the target service is back within the baseline band
- [x] 10.6 Confirm the result file has the four verdicts, the raw response and a timing block with `null` where unmeasured
- [x] 10.7 Run the remaining scenarios and `./run-all-experiments --repeats 3`
- [x] 10.8 Write up the architecture, scenarios, observation format, evaluation method, example result, known limitations and the recommended next experiment
