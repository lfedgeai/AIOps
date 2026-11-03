## Why

We want evidence, not opinion, on whether Jev AI can make sound operational decisions from server-state data, for discussion in the LF Edge AIOps forum. No harness exists to test this. A small, reproducible experiment with independently defined ground truth gives a credible answer without building a monitoring platform.

## What Changes

- Add a synthetic server-state generator whose metrics correlate with a named scenario (not independent random numbers), seedable for exact reproduction.
- Add a scenario catalogue (`healthy`, `degraded`, `critical`, `ambiguous`, `contradictory`), each with an expected outcome defined in code, never produced by an LLM.
- Add an isolated Jev adapter that asks for bounded decisions (severity, action, human-review) and returns them with confidence and latency where available. API usage follows the current Jev documentation, verified before coding.
- Add a CLI (`run.py`) for single runs, random scenarios, seeded runs and batch runs.
- Add an evaluator that compares Jev's decisions to expected outcomes and aggregates accuracy, confidence and latency; results are calculated, never hard-coded.
- Persist per-run results as JSONL/JSON for later analysis.
- Add unit tests for everything except Jev's answers, plus README, `.env.example` and an optional minimal Dockerfile.
- Out of scope: Kubernetes, Prometheus, Grafana, Docker Compose, real telemetry, any remediation actions.

## Capabilities

### New Capabilities
- `scenario-catalog`: named scenarios with independently defined expected outcomes (severity, action, human review), including ambiguous and contradictory cases.
- `server-state-generation`: seedable, scenario-correlated synthetic server observations.
- `jev-decision-adapter`: isolated adapter that sends observations to Jev and returns a bounded, validated decision plus confidence and latency.
- `experiment-runner`: CLI for single, random, seeded and batch runs with clear per-run output.
- `batch-evaluation`: comparison of decisions against expected outcomes, aggregate metrics, and machine-readable result persistence.
- `reproducibility-packaging`: tests, documentation, environment template and optional Dockerfile so others can rerun the experiment.

### Modified Capabilities
<!-- None: new repository, no existing specs. -->

## Impact

- New Python codebase in `jev-ops-experiment/`: `generator/`, `scenarios/`, `jev/`, `evaluation/`, `run.py`, `tests/`, `results/`.
- External dependency: the Jev API (credentials via environment variables, never committed).
- Runtime dependencies kept minimal (HTTP client or Jev SDK per its docs, `python-dotenv`, `pytest`).
- Open item: exact Jev endpoint, SDK and confidence field are unknown until the docs are inspected (first implementation task).
