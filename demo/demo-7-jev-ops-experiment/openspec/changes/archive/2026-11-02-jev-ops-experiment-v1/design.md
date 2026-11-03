## Context

Greenfield Python repo. The goal is a credible benchmark: known scenario → known server state → Jev → bounded decision → comparison with independent ground truth. Audience is the LF Edge AIOps forum, so reproducibility and honesty of results matter more than features. The Jev API is not yet inspected; nothing about its request shape, auth or confidence output may be assumed.

## Goals / Non-Goals

**Goals:**
- Smallest working end-to-end path first, then batch, evaluation, persistence, tests, Docker.
- Exact reproducibility from a seed.
- Ground truth defined in code, independent of any LLM.
- Jev integration replaceable without touching generator, scenarios or evaluator.

**Non-Goals:**
- Production monitoring, real telemetry, remediation actions.
- Kubernetes, Prometheus, Grafana, Docker Compose. (Amended by change `otel-demo-tier2`: Docker Compose is allowed for the Tier 2 OpenTelemetry Demo testbed only. Tier 1 itself remains Python-only, and Kubernetes remains out of scope.)
- Multi-model comparison (the adapter seam allows it later, but it is not built now).

## Decisions

**D1. Flat package layout as in PLAN.md** (`generator/`, `scenarios/`, `jev/`, `evaluation/`, `run.py`). Alternative: single `src/` package. Rejected as unneeded abstraction; the plan's layout is already small.

**D2. Ground truth lives in `scenarios/definitions.py` as frozen dataclasses.** Each `Scenario` carries a name, a metric-profile (ranges per metric, with correlations), and an `Expected(severity, action, human_review)`. The expected outcome is per scenario, not derived from generated numbers, so it cannot depend on Jev. For `ambiguous` and `contradictory`, the expected outcome is a deliberate, documented design choice (e.g. ambiguous → `degraded`/`investigate`/human review yes; contradictory → `degraded`/`investigate`/human review yes). The rationale for each is recorded next to the definition. Alternative: allow a set of acceptable answers. Deferred; v1 uses a single expected value and reports per-field accuracy so partial credit is visible.

**D3. Correlated generation via per-scenario profiles.** Generator draws one latent "stress" value per run inside the scenario's band, then derives CPU, memory, load, latency and error rate from it with bounded noise, so metrics move together. `contradictory` deliberately breaks correlation (e.g. low CPU/memory with high error rate and a service marked `down`; or high resource use with all services `healthy` and zero errors). Randomness comes only from a `random.Random(seed)` instance passed in; no global RNG. Alternative: independent uniform draws. Rejected by the plan.

**D4. Seed semantics.** A run's seed fully determines observations. Batch mode derives per-run seeds from a base seed (`base_seed + i`) and records each one, so any single run in a batch can be replayed with `--scenario X --seed N`. If no seed is given, one is chosen and printed and stored.

**D5. Jev adapter contract.** `jev/client.py` exposes one function/class: `decide(observations) -> Decision` where `Decision` holds severity, action, human_review, optional confidence, latency_ms, and the raw response. It builds the prompt/request, constrains output to the enumerated values, and validates the reply. Invalid or unparseable replies become an explicit `invalid` result (counted as incorrect and reported separately), never silently coerced. Transport errors are retried a small bounded number of times, then recorded as errors and excluded from accuracy denominators but counted and shown. Exact SDK/endpoint/params come from the Jev docs, inspected first (task 1). Alternative: call Jev from the runner directly. Rejected to keep the seam swappable.

**D6. Confidence is optional.** If Jev exposes no confidence/probability, the field is `None` and confidence metrics are reported as "n/a" rather than fabricated.

**D7. Evaluation is pure functions over result records.** `compare(expected, decision)` returns per-field booleans plus overall match (all three fields correct). `aggregate(records)` returns totals, accuracy by field and by scenario, and confidence/latency stats split by correct/incorrect. Pure functions make them unit-testable without Jev.

**D8. Persistence as JSONL, one record per run,** written to `results/<timestamp>-<n>runs.jsonl`, plus a `summary.json` alongside. Each record stores scenario, seed, observations, expected, decision, raw Jev response, confidence, latency, match flags, and code/version info. `results/*` is git-ignored except `.gitkeep`. Alternative: SQLite. Rejected as heavier than needed.

**D9. CLI via `argparse`.** Flags: `--scenario {name|random}`, `--seed`, `--runs`, `--output-dir`. `--runs N` with no scenario cycles scenarios evenly (round-robin) so per-scenario counts are balanced. Stdlib only, no CLI dependency.

**D10. Config via environment variables** loaded from `.env` (`python-dotenv`), documented in `.env.example`. Names finalised after reading the Jev docs. Secrets never logged or persisted (raw responses are stored, request headers are not).

**D11. Tests use `pytest` and a fake Jev client.** Tests cover generation, seed determinism, expected definitions, comparison, and aggregation. A fake adapter with scripted decisions exercises the runner and evaluator end to end. No test asserts what real Jev answers.

**D12. Docker optional:** a single slim Python image, `ENTRYPOINT ["python", "run.py"]`, env passed at run time. No Compose.

## Risks / Trade-offs

- [Expected outcome for ambiguous/contradictory is a judgment call, so "accuracy" there is contestable] → Document the rationale beside each definition, report per-field accuracy, and note in the README that these scenarios test behaviour, not a single objectively right answer.
- [Jev is non-deterministic, so repeated runs of one seed may differ] → Seed fixes the input only; record raw responses; document this in the README.
- [Synthetic data may be too easy or too artificial] → Keep profiles and noise configurable in one place; state the limitation in the README.
- [Jev API unknown, may lack confidence or structured output] → Adapter validates output and confidence is optional (D5, D6).
- [Cost and rate limits on `--runs 100+`] → Bounded retries, per-run error recording, and a note on expected call counts in the README.
- [Invalid replies could inflate or deflate accuracy] → Counted and reported separately, never coerced.

## Open Questions

- Resolved (task 1): the Jev endpoint, auth, question types and confidence fields are recorded in `docs/jev-api-notes.md`. Severity and action map to `choice` questions; human review maps to a `noul` question (yes if P >= 0.5). Noul has no confidence field, so overall confidence is the mean of the severity and action confidences. Direct HTTP with `requests`; Python 3.10+.
- Should ambiguous/contradictory allow more than one acceptable answer in a later version?
