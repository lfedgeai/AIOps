## Context

Jev's errors at Tiers 3/4 did not move with question wording or the incident cutoff (proposal). The stored records keep the built observation (`polls[i].request`), not the raw collector output. So anything computable from the observation alone can be tested by replay on the existing dev (`results/tier3/10.3-combined`) and confirm (`results/tier4/6.2-combined`) sets. Anything that needs new telemetry must run live.

The Jev API takes only `state`, `questions` and `assets` (`docs/jev-api-notes.md`), so the layout of `state` and the question text are the only levers. Jev's docs on jev-1.13 name dates read as text, arithmetic and multi-hop reasoning as weak spots.

## Goals / Non-Goals

**Goals:**
- Test whether the same evidence, with arithmetic and time distances done in code and each callee's state placed next to its caller, changes Tier 3/4 detection, localization and diagnosis.
- Keep it comparable: the same records, prompts, engines and scoring, with only the layout changed, judged on the confirm set against a same-day `raw` replay.
- Keep the engine doing all the reasoning.

**Non-Goals:**
- No suspect ranking, anomaly scoring or pre-diagnosis in the harness.
- No change to `build_observation`, the collectors (Phase A) or scoring.
- Tier 1 and Tier 2 are out of scope here.
- No per-service question fan-out (one call per candidate). It multiplies calls and changes what is measured. A possible later change.

## Decisions

**D1. Layouts are applied after `build_observation`, as a separate module.** `tier3/layouts.py` maps an observation dict to an observation dict. The builder, scrubber and byte budget stay as tested. The facts layout re-checks the byte budget and drops whole fact statements, never source rows, if it grows past it. Alternative: change the builder. Rejected, because `raw` must stay byte-identical and replayable.

**D2. `facts` is a pure function of the observation.** This makes it replayable and leak-safe by construction: it cannot add information it does not have. The "no new tokens" test enforces it. Alternative: compute facts from collector output. Rejected for Phase A because it is not replayable.

**D3. Statements are short strings in a `facts` list on each existing row, plus `callers` and `callees` lists on each service row.** Rows, keys and order stay as in `raw`, so a reader can diff them. Numbers stay alongside the statements. Example on a service row: `"facts": ["error rate 0.400 (baseline 0.000)", "p95 latency 90.0 ms, 3.0 times baseline (30.0 ms)"]`, and `"callees": [{"service": "cart", "calls": 10, "errors": 7, "facts": ["errors on 7 of 10 calls", "available 0 of 1 (baseline available 1 of 1)", ...]}]`. Every statement always shows both values; there is no "same as baseline" judgement, because the cut-off would differ by metric and could be false for the actual values. Error rate has three decimals, latency one, request rate two. The ratio is computed from the shown values, and is left out when the shown baseline is 0.0. Callers and callees get the same entry, so state is not placed one way only (review round 1).

**D4. Time distances replace timestamps.** `ts` becomes `ts_s_before_window_end` and `last_seen` becomes `last_seen_s_before_window_end` (integers). They keep the source field's meaning: an error log's `ts` is its first deduplicated line, an event's `last_seen` its newest. The window itself is shown as `length_s`. Absolute times are removed from the layout, because Jev reads them as text; they stay in the record.

**D5. Version by hand, hash by source.** Each layout has a version. The record hash is SHA-256 of `inspect.getsource` of the layout function and its listed helpers, plus, for `facts`, its vocabulary and `BYTE_BUDGET`. A test walks `_facts`'s calls to check that the helper list is complete. A changed hash with the same version is a compare error, as with prompt variants.

**D6. Phase B (`time-split`) only after Phase A.** It needs collector changes (two Prometheus and Jaeger windows per poll, about twice the query load) and roughly 6 hours of cluster time per layout. Gate: start Phase B only if Phase A, or the recent-window question, changes K002/K010 or RCA beyond noise on the confirm set, or if the user asks for it anyway.

**D7. The recent-window question is reported, not scored.** It is a prompt variant on top of `facts` that adds no question (variants cannot add questions). Instead, the incident question's instructions ask about the most recent part of the window. That is `strict-incident`'s idea again, but now with time distances Jev can use. Its effect shows in the existing detection rows.

## Risks / Trade-offs

- [Baseline end equals injection time] → The runner ends the baseline window at `t0`, just before injection. So `baseline_window.ends_s_before_window_end` is about the seconds since injection. The raw timestamps already carry this, but `facts` makes it easy to read, and a real operator's baseline would be arbitrary history. The healthy control has the same timing, so it does not reveal whether a fault exists. It may help on "is it still happening?". This is a known confound for any K002/K010 change and is reported with the results. Moving the baseline away from `t0` would be a harness change, out of scope here.

- [Facts read as hints] → The vocabulary is fixed and describes values only, never meaning. It passes the "no suspect field" word scan. Two reviewers who are not the author read it before any live call, as in jev-prompt-variants task 6.3.
- [Bigger observation, Jev accuracy "falls as the state grows"] → The added text is bounded per row. The size is reported per layout, and the 64k request limit is checked.
- [Small samples] → 18 fault runs. A change of 2-3 runs is noise. The report says so, and headline figures come from the confirm set.
- [Overfitting the vocabulary to K002/K010] → Facts are generic per-row computations, written once and applied to every service. The dev notes are written before looking at confirm.
- [Opus also improves] → That is a valid finding: the layout helps engines in general. The comparison table shows both.

## Migration Plan

Additive. `raw` is the default, so existing commands, records and reports are unchanged. Rollback means removing the layout module and flag; no record changes.

## Open Questions

- Should `facts` also add per-pod "restarted N s before window end"? The pod rows carry `restarts_in_window` but not a time. Probably no time is available; check the collector before deciding.
- Does Tier 2 get the same layout later? Its observation is close to Tier 3's service sections, so most of `facts` should carry over.
