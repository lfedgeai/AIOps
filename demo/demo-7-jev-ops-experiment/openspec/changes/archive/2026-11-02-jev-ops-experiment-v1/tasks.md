## 1. Jev API discovery (blocks everything else)

- [x] 1.1 Inspect the current Jev API/SDK documentation: endpoint or SDK, auth scheme, request/response format, structured-output support, confidence/probability field, rate limits, Python version requirements
- [x] 1.2 Record findings (source URL, date, confidence field present or absent) in `docs/jev-api-notes.md`
- [x] 1.3 Confirm environment variable names and minimum Python version from the findings; update design open questions

## 2. Project scaffolding

- [x] 2.1 Create package layout: `generator/`, `scenarios/`, `jev/`, `evaluation/`, `tests/`, `results/.gitkeep`, each package with `__init__.py`
- [x] 2.2 Add `requirements.txt` (Jev SDK or HTTP client per 1.1, `python-dotenv`, `pytest`)
- [x] 2.3 Add `.env.example` with placeholder values for every required variable
- [x] 2.4 Confirm `.gitignore` covers `.env` and `results/*` (keeping `.gitkeep`)

## 3. Scenario catalogue and ground truth

- [x] 3.1 Define allowed value sets (severity, action, human-review) and an `Expected` dataclass in `scenarios/definitions.py`
- [x] 3.2 Define `Scenario` dataclass (name, metric profile, expected, rationale) and a lookup that lists valid names on error
- [x] 3.3 Define `healthy`, `degraded`, `critical` scenarios with expected outcomes and rationale
- [x] 3.4 Define `ambiguous` and `contradictory` scenarios with expected outcomes and explicit written rationale for each choice

## 4. Server-state generator

- [x] 4.1 Define the `ServerState` dataclass with all required fields and a JSON-safe `to_dict`
- [x] 4.2 Implement generation from a latent stress value plus bounded noise, using a private `random.Random(seed)`
- [x] 4.3 Implement service statuses and recent events/restarts consistent with each scenario
- [x] 4.4 Implement the deliberately conflicting signal patterns for `contradictory`
- [x] 4.5 Add value clamping so every field stays in its valid range

## 5. Jev adapter and minimal end-to-end path

- [x] 5.1 Define `Decision` result type with severity, action, human_review, confidence, latency_ms, raw_response and status (`ok`, `invalid`, `error`)
- [x] 5.2 Build the bounded-decision prompt/request from observations, per the Jev docs
- [x] 5.3 Implement `decide()` in `jev/client.py` with response validation against the allowed sets (no coercion)
- [x] 5.4 Implement env-based config loading with a clear error naming any missing variable
- [x] 5.5 Implement bounded retries for transient failures, returning an `error` result on exhaustion
- [x] 5.6 Ensure credentials never appear in returned or stored data
- [x] 5.7 Minimal `run.py`: one scenario, generate, call Jev, print decision and compare (smoke-test against real Jev once) (real-Jev batches 2026-10-05 and 2026-10-18)

## 6. Evaluation

- [x] 6.1 Implement `compare(expected, decision)` returning per-field and overall correctness
- [x] 6.2 Implement `aggregate(records)`: totals, accuracy by field and scenario, confidence overall/correct/incorrect, latency stats, invalid and error categories, empty-input safe
- [x] 6.3 Implement text summary rendering (`Overall`, `By scenario`, by decision type, confidence, latency)

## 7. Runner CLI and persistence

- [x] 7.1 Add `argparse` options `--scenario`, `--seed`, `--runs`, `--output-dir`; validate scenario names
- [x] 7.2 Implement per-run output in the format from PLAN.md (scenario, seed, server state, expected, Jev, confidence, PASS/FAIL with differing fields)
- [x] 7.3 Implement batch mode: balanced round-robin over scenarios, per-run seeds derived from a base seed and recorded, continue on per-run errors
- [x] 7.4 Write per-run JSONL and summary JSON to timestamped files under `results/`
- [x] 7.5 Verify a single run replays identically from a recorded scenario and seed
- [x] 7.6 Print the batch summary at the end; confirm printed figures equal the summary file

## 8. Tests

- [x] 8.1 Scenario generation tests: all fields present and in range, correlation and severity ordering across many seeds
- [x] 8.2 Seed determinism tests: same seed equal, different seed differs, global RNG untouched
- [x] 8.3 Expected-outcome definition tests: all required scenarios exist, values in allowed sets, rationale present, outcome independent of seed
- [x] 8.4 Comparison logic tests: full match, partial match, invalid decision
- [x] 8.5 Aggregation tests: counts, per-scenario and per-field accuracy, confidence split, no-confidence case, empty input, invalid/error categories
- [x] 8.6 Runner tests using a fake adapter with scripted decisions (no test asserts real Jev answers)
- [x] 8.7 Confirm `pytest` passes offline with no credentials

## 9. Documentation and packaging

- [x] 9.1 Write README: plain-language description, experimental-benchmark disclaimer, prerequisites, Python version, install, Jev config and env vars, single run, batch, seed reproduction, results storage
- [x] 9.2 Document the seed/non-determinism caveat and the rationale for ambiguous/contradictory expectations
- [x] 9.3 Add minimal `Dockerfile` (slim Python, `ENTRYPOINT ["python", "run.py"]`, no baked credentials, add `.dockerignore` excluding `.env` and `results/`)
- [x] 9.4 Build and run the image once with env vars and a mounted results directory (2026-10-19: `--runs 7` against real Jev, 7/7 HTTP 200, report from the image; no `.env` in the image, no key in the results; README now passes `--user` so result files are not root-owned)

## 10. Application logs

- [x] 10.1 Add a seeded application-log generator (`generator/app_logs.py`) with templates by kind, attributed to affected services
- [x] 10.2 Add log specs to every scenario profile; contradictory logs conflict with metrics
- [x] 10.3 Add `hung_worker` (expects restart) and `log_only_errors` (evidence only in logs) with rationale
- [x] 10.4 Add `--no-logs` ablation: same metrics, logs withheld; record `include_logs`, `-nologs` file suffix, report states mode
- [x] 10.5 Show logs in per-run output (level counts plus last 15 lines)
- [x] 10.6 Tests: well-formed, deterministic, match severity, name affected service, no answer leakage, metrics unchanged by logs, ablation
- [x] 10.7 Update README, CLAUDE.md, run-all default (70 runs = 10 per scenario)

## 11. Probable cause

- [x] 11.1 Add `CAUSES` and a required `probable_cause` to `Expected`; set and comment the expected cause for every scenario
- [x] 11.2 Add a `probable_cause` choice question to the Jev adapter, validated like the others; keep overall confidence to severity and action
- [x] 11.3 Score cause separately: in `by_field`, not in `overall`; tolerate old records without it
- [x] 11.4 Show cause in per-run output, summary, report tables and misses
- [x] 11.5 Tests: invalid cause rejected, wrong cause does not fail a decision, cause accuracy reported, old records compare

## 12. Telemetry

- [x] 12.1 Capture per-run tokens, Jev `elapsedMs`, round trip, total time, attempts, HTTP status, payload sizes, raw usage
- [x] 12.2 Cost: reported by Jev if present, else estimated from optional `JEV_PRICE_*` env prices, else n/a; source recorded
- [x] 12.3 Aggregate telemetry (totals, means, percentiles, overhead, retries, status counts, per scenario) and show it in the summary, per-run output and report
- [x] 12.4 Record batch meta: start/end, wall time, runs/min, git commit, Python, platform, Jev URL and model
- [x] 12.5 Tests for capture, cost rules, retries, errors, aggregation, meta and report

## 13. Easy checkout for other users

- [x] 13.1 `./run-all.sh --check`: setup and tests with no key; Python >= 3.10 check; venv error hint; creates `.env`
- [x] 13.2 Reject the placeholder key with a clear message
- [x] 13.3 Pin dependencies; `.gitattributes` for LF shell scripts
- [x] 13.4 CI: fresh-checkout `--check` on Python 3.10-3.13, plus Docker build
- [x] 13.5 README quick start, manual and Windows setup, Docker with `--env-file`, troubleshooting
- [x] 13.6 Verify from a fresh clone (local and in clean containers)

## 14. First experiment and wrap-up

- [x] 14.1 Run `python run.py --runs 70` and `--runs 70 --no-logs` with the same seed against real Jev and keep the JSONL and summary locally (2026-10-18, seed 1000: `results/20261018-134025-70runs.jsonl`, `results/20261018-134058-70runs-nologs.jsonl`)
- [x] 14.2 Review ambiguous/contradictory results and note observations in the README or a `docs/` findings note (calculated from real output only) (`docs/results-2026-10-18-tier1-jev.md`)
- [x] 14.3 Update `PLAN.md` status to implemented and tag v0.1.0 (2026-10-19)
