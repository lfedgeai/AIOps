## Why

Changing the question wording (`docs/results-2026-11-01-prompt-variants.md`) and the incident cutoff (`docs/results-2026-11-02-incident-threshold.md`) did not move Jev's two main errors at Tiers 3/4:

- **False alarms on recovered failures.** All of them are K002 and K010. Jev gives these p 0.75 to 0.94, inside the range of real faults.
- **Blaming the caller.** Jev names `frontend` or `frontend-proxy` when `cart` or `valkey-cart` is the failing component. RCA is 9 to 10 of 18.

Jev's own documentation names three likely causes (https://docs.typesafe.ai/model-jaggedness/jev-1.13.md, read 2026-11-02):

- It "reads dates as text, not as ordered quantities".
- "Jev is not a calculator".
- "multiple hops of reasoning costs accuracy".

Today's observation gives raw numbers and ISO timestamps. A dependency's health sits in a different section from its caller's errors. And the 180 s window rolls over the moment of injection, so on the first poll most of the window is from before the fault. The docs advise asking "the most explicit, narrow, specific, atomic questions" and keeping deterministic work in code (https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md). The question is whether the same evidence, laid out the way the docs recommend, changes Jev's answers.

## What Changes

- Add **observation layouts**: named, versioned transforms of the Tier 3 observation, selected with `--observation <name>` (default `raw`, today's observation byte for byte). Every record stores the layout's `{name, version, sha256 of the transform source}`. A record without the field counts as `raw` v1.
- **Phase A, `facts` (replayable).** This layout is a pure function of today's observation dict, so it adds no information, only arithmetic and placement:
  - Each service row gets plain statements computed in code: "error rate 0.40, baseline 0.00", "latency p95 3.1× baseline", "ready 0 of 1, baseline 1 of 1".
  - Each service row also lists its callees, with each callee's own readiness and error statements.
  - Every timestamped item gets "N s before the end of the window" in place of a bare timestamp.
  - Because the input is the stored observation, Phase A can be replayed on the existing dev and confirm records (`10.3-combined`, `6.2-combined`) at replay cost, with no cluster.
- **Phase B, `time-split` (live only).** The collectors query a "recent" window (the last 60 s) and the rest of the window separately. The observation shows them as two blocks, with durations computed in code. Old records lack the data, so this needs fresh live Tier 3/4 batches. It starts only if Phase A or the time question below shows an effect worth the cluster time.
- A **time-anchored incident wording**, run as `strict-incident` on top of `facts`. Variants cannot add questions, so the existing incident question asks about the most recent part of the window, and `facts` now gives Jev the time distances it needs to answer. Scoring does not change (design D7).
- **No suspect field, ever.** A layout may compute and place facts, but must never rank, flag or name a likely cause or component. It must treat every service the same way. The engine still does all the diagnosis.
- **Replay and compare** take `--observation` alongside `--prompt`. Compare groups by `(engine, prompt, observation)`.

## Capabilities

### New Capabilities
- `observation-layouts`: the layout registry, the `raw` byte-for-byte equivalence, the `facts` transform and its limits (pure function of the observation, no added information, no suspect field, same treatment for every service), recording, the leak tests run over every layout, and selection on live runs and replay.
- `observation-time-split`: Phase B. The recent and earlier collection windows, the two-block layout, null for anything not measurable, and that the layout is live only.

### Modified Capabilities
<!-- None. openspec/specs/ is empty; earlier changes are not archived. Prompt replay and comparison (change jev-prompt-variants) gain an --observation dimension, described in this change's specs. -->

## Impact

- **Scope:** Tiers 3 and 4 only, where the false alarms and localization data are. Tier 2 can follow with the same mechanism. Tier 1 is synthetic and out of scope.
- **Code:**
  - A new `tier3/layouts.py` with `raw` and `facts`, imported by the runner and replay. It must not import `tier3.scenarios` or `tier3.faults`, like `tier3/observation.py`. `build_observation` is unchanged; layouts are applied after it.
  - `tier3/replay.py`, `tier3/compare.py`, and `--observation` on `python -m tier3` and `python -m tier4`.
  - Phase B adds window parameters to `tier3/collect.py`.
- **Rules:** a `CLAUDE.md` section for layouts, which are code, not data like prompt variants. The whole-path leak tests in `tests/tier3/test_runner.py` and `tests/tier4/test_runner.py` are parametrised over every layout. `facts` must also pass a "no new tokens" test: every name in its output already appears in its input.
- **Jev usage:** Phase A replays cost about 631 calls per layout and prompt pair, under $0.20 at the published price. Phase B costs a live batch per layout, 40 runs on the cluster, about 6 hours.
- **Fairness:** the Claude Opus baseline gets the same layout, so a gain is attributed to the layout and not to Jev alone.
- **CI:** offline as before.
