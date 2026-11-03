## 1. Layout registry and raw

- [x] 1.1 `tier3/layouts.py`: registry, `raw` (identity), `VERSION` per layout, source hash (D5); no import of `tier3.scenarios` or `tier3.faults` (isolation test)
- [x] 1.2 Apply the layout after `build_observation` in `tier3/runner.py`, and in `tier3/replay.py` on the stored `request`; `--observation` on `python -m tier3` and `python -m tier4` (`run`, `batch`, `replay`); unknown name exits and lists the layouts
- [x] 1.3 Record `observation: {name, version, sha256}` in live and replay records; readers treat a missing field as `raw` v1
- [x] 1.4 Compare groups by `(engine, prompt, observation)`; error on one name with two hashes
- [x] 1.5 Offline tests: `raw` is byte-identical on every fixture poll; the flag reaches Jev and Opus requests; leak tests parametrised over layouts

## 2. facts layout (Phase A)

- [x] 2.1 Write `docs/observation-facts-dev-notes.md` from the dev set only (`results/tier3/10.3-combined`): what the raw layout makes Jev compute or cross-reference, before writing any fact (2026-11-02)
- [x] 2.2 Implement `facts` (D3, D4): per-service statements, readiness, callees with their own statements, seconds before window end; fixed vocabulary; byte budget re-check
- [x] 2.3 Offline tests: "no new tokens", the no-suspect word scan, the callee scenario, determinism, zero-baseline wording, the budget trim
- [x] 2.4 Independent review of the vocabulary and an example output by two reviewers who are not the author, before any live call; record it in the commit message (2026-11-02: two independent reviewer agents, neither shown scenarios, results or dev notes. Fairness review: no blockers; fixed one-way callee placement (callers added), the "same as baseline" judgement, conditional statements, ready-vs-available, per-row budget trimming, time field names; recorded the baseline-end/injection-time confound in design Risks. Correctness review: fixed a Python None in text, ratio from shown values, budget notes, hash covering BYTE_BUDGET and a helper-completeness test, Tier 4 summary identity. Still v1: no facts record existed before the fixes)

## 3. Phase A replays (real Jev and Opus calls)

- [x] 3.1 Dry-run, then replay `raw` and `facts` with `--prompt baseline` on dev (`10.3-combined`) and confirm (`6.2-combined`), engine Jev
- [x] 3.2 The same with `--prompt strict-incident` under `facts` (D7)
- [x] 3.3 The same pairs with `--engine claude-opus` (API only, `ANTHROPIC_API_KEY`), confirm set
- [x] 3.4 Write `docs/results-<date>-observation-facts.md`: confirm tables first, K002/K010 and RCA per scenario, sizes, censoring, limits. Every figure from the records (2026-11-02: `docs/results-2026-11-02-observation-facts.md`, `docs/analysis/observation_facts.py`)

## 4. Phase B gate and time-split

- [x] 4.1 Decide the gate (D6) from 3.4 and record the decision here (2026-11-02: no-go. On the confirm set K002/K010 did not change with `facts` or `facts` + `strict-incident`, and RCA moved +2/-0, inside the pre-stated "2-3 runs is noise" band. The user chose not to start Phase B anyway)
- [x] 4.2 Skipped (gate no-go). If go: recent and earlier windows in `tier3/collect.py`, the `time-split` layout, replay refusal, null rules, budget per block, offline tests
- [x] 4.3 Skipped (gate no-go). If go: live Tier 4 batch with `time-split`, Jev and Opus, and a results page

## 5. Docs and rules

- [x] 5.1 README: "Observation layouts" section and `--observation` in the command tables
- [x] 5.2 `CLAUDE.md`: layout rules (pure function for replayable layouts, no suspect field, same treatment for every service, version bump, leak tests over every layout); `tier3/layouts.py` in the layout; this tasks file in the conventions
- [x] 5.3 `./run-all.sh --check` passes on a fresh clone with no key; CI green (2026-11-02, CI run 41462979237 on de7dcbc)

Note (2026-11-02): the `observation-time-split` spec delta was removed before archiving, because Phase B was not started (4.1). It is in git history at 86376e7.
