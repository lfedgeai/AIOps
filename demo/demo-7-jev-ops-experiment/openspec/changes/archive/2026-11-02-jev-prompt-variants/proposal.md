## Why

On [lfedgeai/AIOps#41](https://github.com/lfedgeai/AIOps/issues/41) (2026-10-27) the group asked us to "explore better prompting". Today every tier asks Jev one fixed question set, hard-coded in Python (`jev/client.py`, `tier2/jev_adapter.py`, `tier3/questions.py`). So we cannot say whether a result comes from Jev or from how we asked. The results so far point at the wording. At Tier 2, Jev reported an incident in 4/4 healthy runs. At Tiers 3 and 4 it raised false alarms on 4 to 5 of 22 no-incident runs. At Tier 1, it asked for human review in 21 of the 30 runs that did not need it (`docs/results-2026-11-01-precision-recall-f1.md`). We want to try other ways of asking, switch between them by name, and get one table that says "asked this way 27/30 (90%), that way 9/30 (30%)" on the same observations.

## What Changes

- Add **prompt variants**: named, versioned question sets kept as YAML files, one directory per tier (`prompts/tier1/`, `prompts/tier2/`, `prompts/tier3/`). A variant may change each question's `instructions` and every option's description. It may not change question IDs, question types or option keys. The vocabularies, the parsers and the expected outcomes therefore stay exactly as they are.
- Add **`baseline`**: today's question set, moved into YAML. A test checks that it produces exactly the questions sent today, so every earlier result counts as `baseline`.
- Add a **`--prompt <variant>`** option to `run.py`, `python -m tier2`, `python -m tier3` and `python -m tier4` (default `baseline`). Every engine gets the selected variant: Jev, the Claude and Ollama baselines, and the Tier 3/4 Claude baseline. So "every engine gets the same questions" still holds within a variant.
- **Record the variant** in every run record: its name, version and a SHA-256 of the exact questions sent. Records from before this change count as `baseline`.
- Add **replay**: `python -m tier2 replay` and `python -m tier3 replay` re-ask the observations stored in earlier run records under another variant. No Docker or cluster is needed, and the scoring rules are the same. So variants are compared on identical observations, which live Tier 2/3 runs cannot do. Replay never sends anything that was not in the original request's observation.
- Add a **variant comparison report**: one table per tier with one column per variant (and engine). Each cell is a count, its denominator and a percentage, for example `27/30 (90%)`. Tier 1 compares operational-decision accuracy, per-field F1 and detection. Tiers 2 and 3 compare detect, RCA, diagnosis, remediation correct, FP runs, and detection precision, recall and F1. A name used with two different hashes is an error, not a merge.
- Add **two hand-written candidate variants per tier**, each with a written rationale: one aimed at false alarms (stricter incident and human-review wording), one with option descriptions written as observable evidence. They go in the variant files.
- Add a **dev / confirm split**. Variants are tuned on one set of records and judged on another set they were not tuned on: Tier 1 seeds 1000 and 2000; Tier 3 batch `10.3` and the Tier 4 Jev batch `6.2`, which saw the same faults separately. The headline figure for a variant is its confirm-set figure.

## Capabilities

### New Capabilities
- `prompt-variants`: the variant file format, what a variant may and may not change, the registry and loader, `baseline` equivalence, hashing and recording, the leak checks every variant must pass, and one variant shared by all engines.
- `prompt-replay`: re-asking stored Tier 2 and Tier 3 observations under a variant, the scoring rules, what replay cannot measure (MTTD and remediation are `null`), and keeping replay records apart from live ones.
- `prompt-comparison`: the cross-variant report, the dev/confirm split, counts with denominators, and refusing to mix incompatible records.

### Modified Capabilities
<!-- None. openspec/specs/ is empty; earlier changes are not archived. The question wording behind every existing result does not change: `baseline` reproduces it exactly. -->

## Impact

- **Code:** new `prompts/` package (loader, registry, YAML files). Question sets in `jev/client.py`, `tier2/jev_adapter.py` and `tier3/questions.py` are built from the selected variant. The `llm/` engines and `tier3/claude_engine.py` take the variant. `run.py`, `report.py`, `tier2/cli.py`, `tier3/cli.py` and `tier4/cli.py` get `--prompt`. Tier 2 and Tier 3 get a replay command. The parsers do not change.
- **Rules:** the `CLAUDE.md` rule "all Jev HTTP and API code lives in `jev/client.py`" stays. Variants are data, not HTTP code. The adapters remain the only modules that import `jev.client`. The leak tests run over every variant.
- **Jev usage:** a replay asks Jev once per stored poll. A Tier 3 batch is about 351 calls per variant. A Tier 1 batch is 70 calls per variant.
- **Dependencies:** none new (`pyyaml` is already pinned).
- **Docs:** README section and `CLAUDE.md` rules for variants. A results page once live comparisons have run.
- **CI:** stays offline. Fake clients and stored fixture records; no Jev, Docker or cluster in tests.
