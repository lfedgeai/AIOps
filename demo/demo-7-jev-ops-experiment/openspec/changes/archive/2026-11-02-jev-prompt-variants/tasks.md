## 1. Check the API before coding against it

- [x] 1.1 Re-read https://docs.typesafe.ai/api for `instructions` and `criteria`: length limits, whether option descriptions are used by the model, and any other request field that affects the answer. Record it in `docs/jev-api-notes.md` (a "Prompt variants" section). Do not guess parameters (2026-11-01: only `instructions` and `criteria` text can vary; no length limits documented; no context, example, system or temperature fields; option descriptions reach the model)
- [x] 1.2 Freeze today's three question sets as JSON fixtures (`tests/fixtures/questions-tier{1,2,3}-pre-variants.json`), from the current constants, before any refactor

## 2. Variant loader and baseline

- [x] 2.1 `prompts/loader.py`: load `prompts/<tier>/<name>.yaml`, validate against the tier's question IDs, types and option keys (D1), build the question dict, and compute `sha256` over canonical JSON (D3). List the variants for each tier
- [x] 2.2 `prompts/tier1/baseline.yaml`, `prompts/tier2/baseline.yaml`, `prompts/tier3/baseline.yaml` with today's text (Tier 3 component and node text as `criteria_templates`)
- [x] 2.3 Build `jev/client.py`, `tier2/jev_adapter.py` and `tier3/questions.py` `QUESTIONS` from `baseline`. Keep the parsers unchanged (also `COPY prompts` in the Dockerfile)
- [x] 2.4 Offline tests: baseline equals the frozen fixtures byte for byte; each loader rejection (missing or extra question, missing or extra option, `type` key, name/file mismatch, bad version); hash stability

## 3. Selecting and recording a variant

- [x] 3.1 `--prompt` on `run.py`, `python -m tier2`, `python -m tier3` and `python -m tier4` (`run`, `batch`), defaulting to `baseline`. An unknown name exits before any run and lists the variants
- [x] 3.2 Pass the built questions through the engines: the Jev adapters (`build_request(questions=...)`), `llm/prompt.py`, `llm/claude.py`, `llm/ollama.py` and `tier3/claude_engine.py` take the question dict instead of importing the constant (D4) (the helpers in `llm/prompt.py` and `tier3/claude_engine.py` keep the baseline as a default argument; every engine passes `self.prompt.questions`)
- [x] 3.3 Add `prompt: {name, version, sha256}` to Tier 1 JSONL lines and summaries, and to Tier 2, 3 and 4 records and summaries. Readers treat a missing field as `baseline` v1 and say so (`prompts.loader.record_prompt`; the reports use it in task 5. A non-baseline Tier 1 file name gets `-prompt-<name>`)
- [x] 3.4 Parametrise the existing leak tests over every variant file, and add the variant-text term check (D9)
- [x] 3.5 Offline tests: `--prompt` reaches every engine's request or prompt; the record field; the unknown-variant exit; isolation tests still pass (the adapters are still the only `jev.client` importers)

## 4. Replay

- [x] 4.1 `tier3/replay.py` and `python -m tier3 replay <records> --prompt <name> [--engine ...]`: load Tier 3 and Tier 4 records, ask every stored poll in order, score with `tier3/evaluate.py` (first detecting poll, else last), false-positive polls on no-incident records, `censored`, null timings, `detect_poll`, `replay_detect_seconds_since_inject` (D5)
- [x] 4.2 `tier2/replay.py` and `python -m tier2 replay`: the same for the single stored `request`, scored with `tier2/evaluate.py`
- [x] 4.3 Write replay records to `results/<tier>/replay/<run-id>/` with `mode`, `source`, `prompt` and `engine`. Skip non-`ok` sources and count them. Never touch the sources. Live scorecards refuse replay records without `--include-replay` (also `--dry-run`, which prints the record and call counts and calls nothing; checked against real records: 351 calls for `10.3-combined`, 280 for `6.2-combined`; an engine that echoes the stored answers reproduces the live Tier 2, 3 and 4 scorecards exactly)
- [x] 4.4 Small Tier 2, Tier 3 and Tier 4 fixture records under `tests/fixtures/replay/` (hand-made, no live data) (done differently: source records are made at test time by the offline fake runners for every Tier 2 and Tier 3 scenario and four Tier 4 ones, leak-bait telemetry included, plus hand-built dicts for censoring and scoring cases in `tests/test_replay.py`. No static fixture files)
- [x] 4.5 Offline tests with a fake engine: request equals stored `request` plus the variant's questions; no ground truth in any request (every fixture × every variant); censoring; control false-positive counting over all polls; source files unchanged; skipped sources

## 5. Comparison report

- [x] 5.1 `python -m tier3 compare` and `python -m tier2 compare`: group by `(engine, prompt, version)`, the rows in the comparison spec, `k/n (p%)` cells, `n/a` for empty denominators, F1 as a fraction plus a decimal (shared layout in `prompts/compare.py`, rows in `tier3/compare.py` and `tier2/compare.py`; `--unpaired` added for batches that saw different observations, such as live Jev against live Opus; checked on the real records: Tier 3 dev and Tier 4 Jev/Opus figures match the published scorecards)
- [x] 5.2 Common-source check with a warning and the differing run IDs; never mix live and replay in one group; error on a hash conflict
- [x] 5.3 `--dev` and `--confirm`: confirm table first, flag variants whose `dev_records` overlap the confirm set; show the same-day replayed baseline next to the original live records
- [x] 5.4 `report.py --compare`: columns labelled `engine / prompt`; `--by-prompt` grouping (also `--metrics --by-prompt`, the hash check, and a note for records without a prompt field. Tier 1 has no `--dev`/`--confirm`: its sets are separate seeds, so pass the seed-1000 and seed-2000 files to separate `--compare` calls)
- [x] 5.5 Offline tests over fixture records for every row, the warnings, the hash error and the dev/confirm layout

## 6. Candidate variants

- [x] 6.1 Read the dev-set records (Tier 1 seed 1000; Tier 2 `20261012-190447-369b`; Tier 3 `10.3-combined`), the false alarms first, and write a short note of what the baseline wording gets wrong (`docs/prompt-variants-dev-notes.md`, 2026-11-01. The open question was decided by the user the same day: policy text is allowed in general terms, marked `style: policy` and labelled in the reports, design D10)
- [x] 6.2 Write `strict-incident` and `evidence-options` for each tier, by hand, with `rationale` and `dev_records` (D8). No scenario-specific rules (D9) (2026-11-01: six files. `strict-incident` is `style: policy` and changes only the incident question (Tiers 2/3) or the human-review question (Tier 1); `evidence-options` is `style: wording` and changes only option descriptions: components and diagnoses (Tiers 2/3), severity and cause (Tier 1). Questions not named keep the baseline text exactly)
- [x] 6.3 Have someone other than the author review the variants for hints and leaks before any live call, and record the review in the variant's commit message (2026-11-01: two independent reviewer agents, neither shown the dev notes or results. Round 1 on v1 -> v2 in 6cc5343; round 2 on v2 -> v3 in the next commit. No leaks in either round. Stopped after round 2 on purpose: each reviewer reads the expected outcomes, so further rounds would fit the variants to the test set. Final: tier1/tier3 strict-incident v3, tier1/tier2/tier3 evidence-options v3, tier2/strict-incident v1; remaining risks are in each rationale)

## 7. Live comparison (real Jev calls)

- [x] 7.1 Tier 1: run 70 runs with `--seed 1000` for each variant (dev), then 70 runs with `--seed 2000` for `baseline` and each variant (confirm) (2026-11-01, Jev API, jev-1.14.0, v3 variants: dev `results/20261101-182029-70runs-prompt-strict-incident.jsonl`, `...182055-...-evidence-options.jsonl`; confirm `...182121-70runs.jsonl` (baseline), `...182146-...-strict-incident.jsonl`, `...182212-...-evidence-options.jsonl`. 350 calls, 0 errors, 0 invalid, $0.0222 estimated. The dev baseline `20261018-134025-70runs.jsonl` was answered by jev-1.13.0, so dev comparisons against it mix a model change)
- [x] 7.2 Tier 2: replay `baseline` and each variant on both Tier 2 directories (2026-11-01, jev-1.14.0: dev `20261012-190447-369b` (6 runs), confirm `20261012-195728-7983` (18 runs); replays `results/tier2/replay/20261101-1626*`, 72 calls, all ok; report `results/tier2/20261101-prompt-variants-compare.md`. A fresh Tier 1 dev baseline on jev-1.14.0 is `results/20261101-182620-70runs.jsonl`)
- [x] 7.3 Tier 3/4: replay `baseline` and each variant on `results/tier3/10.3-combined` (dev) and `results/tier4/6.2-combined` (confirm) (2026-11-01, jev-1.14.0: replays `results/tier3/replay/20261101-16*`, 1,893 calls, 0 invalid, 0 censored; report `results/tier3/20261101-prompt-variants-compare.md`)
- [x] 7.4 Optional: if a variant clearly beats baseline on the confirm set, run one live Tier 3 batch with it (`./run-all-k8s-experiments --prompt <name>`) to check it without censoring and with real MTTD (2026-11-01: not run. No variant clearly beats baseline on the confirm set: evidence-options gains RCA 3/18 but loses diagnosis 2/18; strict-incident changes no detection)
- [x] 7.5 Write `docs/results-<date>-prompt-variants.md`: confirm tables first, censoring and sample-size limits, what changed and what did not. Every figure comes from the records (`docs/results-2026-11-01-prompt-variants.md`)

## 8. Docs and rules

- [x] 8.1 README: a "Prompt variants" section (how to switch, replay, compare, and the dev/confirm rule) and the `--prompt` option in the command tables (2026-11-02)
- [x] 8.2 `CLAUDE.md`: rules for variants (text only, hand-written with a rationale, no scenario-specific hints, version bump on any edit, confirm set not used to write variants); `prompts/` in the layout; this change's `tasks.md` in the conventions list (2026-11-02)
- [x] 8.3 `./run-all.sh --check` passes on a fresh clone with no key; CI green on Python 3.10-3.13 (2026-11-02: fresh clone of 385eb80, no key: 596 passed, 1 skipped (the Tier 2 demo-checkout test); CI on the push of this change)
