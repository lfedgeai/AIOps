# Results: does the `facts` observation layout help? Tier 3/4 replays (2026-11-02)

Change `openspec/changes/archive/2026-11-02-observation-facts`, Phase A (tasks 3.1-3.4). The `facts` layout (`tier3/layouts.py`, v1, sha256 `780097d0…`) adds plain per-row statements to the same observation. Examples are a value next to its baseline, deployment readiness next to a service, each edge's callers and callees, and seconds before the window end. It adds no new information and never names a suspect. The question: with the same stored observations laid out this way, do Jev's localization and the K002/K010 false alarms change? Every figure below comes from the replay records, through `docs/analysis/observation_facts.py`.

## Short answer

Facts helps Jev's localization a little and does nothing for the false alarms. On the confirm set, Jev localized 12/18 fault runs with `facts` against 10/18 with `raw` (2 gained, 0 lost). The same prompt and layout asked twice that day flipped 0 runs, so the gain is above the measured noise. It is still only two runs, and both gains repeat on the dev set (K001, K008). Diagnosis lost one run on confirm (K007) and the dev set shows the same loss. False-alarm runs did not change: K002 3/3 and K010 2/3 with both layouts. Adding the `strict-incident` prompt on top of `facts` changed nothing useful and lost one detection. Opus localized 18/18 with either layout, so it gives no signal for the layout. Its false alarms (9/22) did not change either.

## Sources

All calls ran on 2026-11-02, from `results/facts-phaseA-20261102.sh`. Jev: `jev-latest`, 1,893 calls, 0 errors, 0 invalid replies, $0.484 estimated from the `JEV_PRICE_*` values in `.env`. Opus: Anthropic API, 560 calls, 0 errors. Cost `n/a`, because no `CLAUDE_*_PRICE_*` is set.

| Set (source) | Jev raw (noise floor, earlier same day) | Jev raw | Jev facts | Jev facts + strict-incident | Opus raw | Opus facts |
|---|---|---|---|---|---|---|
| confirm (`results/tier4/6.2-combined`) | `results/tier3/replay/20261102-131914-5400` | `…135216-1356` | `…135410-0041` | `…135616-e810` | `…135824-276d` | `…141431-4f45` |
| dev (`results/tier3/10.3-combined`) | `results/tier3/replay/20261102-131652-587a` | `…134431-028c` | `…134656-4668` | `…134934-bffa` | — | — |

The noise floor is the `baseline-reversed` probe's same-order baseline (`docs/results-2026-11-02-option-order.md`), replayed about 25 minutes before the Phase A `raw` replay. It uses the same prompt and the same layout. The full compare table is `python -m tier3 compare <dirs> --dev results/tier3/10.3-combined --confirm results/tier4/6.2-combined`.

## Confirm set (headline)

18 fault runs, 22 no-incident runs (K000 ×10, K002, K004, K006, K010 ×3).

| | Jev raw | Jev facts | Jev facts + strict-incident | Opus raw | Opus facts |
|---|---|---|---|---|---|
| detect (fault runs) | 18/18 | 18/18 | 17/18 | 18/18 | 18/18 |
| RCA (localization) | 10/18 | **12/18** | 12/18 | 18/18 | 18/18 |
| diagnosis | 10/18 | 9/18 | 8/18 | 18/18 | 18/18 |
| remediation correct (action) | 18/18 | 18/18 | 18/18 | 18/18 | 18/18 |
| FP runs | 5/22 | 5/22 | 5/22 | 9/22 | 9/22 |
| detection F1 | 0.88 | 0.88 | 0.85 | 0.80 | 0.80 |
| censored runs | 0/40 | 0/40 | 2/40 | 0/40 | 0/40 |

Run-level flips against Jev raw on the same observations:

| Pair | Localization flips (gained / lost) | Diagnosis flips | FP-run flips |
|---|---|---|---|
| Jev raw, asked twice (noise) | 0 | 1 | 0 |
| Jev raw → facts | 2 (+2 / −0) | 1 | 0 |
| Jev raw → facts + strict-incident | 2 (+2 / −0) | 2 | 2 |
| Opus raw → facts | 0 | 0 | 0 |

## Per scenario

Fault scenarios show localization / diagnosis PASS out of 3. No-incident scenarios show FP runs out of 3 (K000 out of 10).

| Scenario | Confirm: Jev raw | Jev facts | Jev facts+strict | Opus raw | Opus facts | Dev: Jev raw | Jev facts | Jev facts+strict |
|---|---|---|---|---|---|---|---|---|
| K000 (control) | FP 0 | 0 | 1 | 0 | 0 | FP 0 | 0 | 0 |
| K001 | 2 / 1 | **3** / 1 | 3 / 1 | 3 / 3 | 3 / 3 | 0 / 0 | **2** / 0 | 2 / 0 |
| K002 (no incident) | FP 3 | 3 | 3 | 3 | 3 | FP 3 | 3 | 3 |
| K003 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 |
| K004 (no incident) | FP 0 | 0 | 0 | 0 | 0 | FP 0 | 0 | 0 |
| K005 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 |
| K006 (no incident) | FP 0 | 0 | 0 | 3 | 3 | FP 0 | 0 | 0 |
| K007 | 1 / 3 | 1 / **2** | 1 / 1 | 3 / 3 | 3 / 3 | 1 / 3 | 0 / **1** | 1 / 2 |
| K008 | 1 / 0 | **2** / 0 | 2 / 0 | 3 / 3 | 3 / 3 | 1 / 0 | **2** / 1 | 2 / 0 |
| K009 | 0 / 0 | 0 / 0 | 0 / 0 | 3 / 3 | 3 / 3 | 0 / 0 | 0 / 0 | 0 / 0 |
| K010 (no incident) | FP 2 | 2 | 1 | 3 | 3 | FP 1 | 1 | 1 |

On dev, Jev raw → facts flipped 4 localization runs (+3 / −1), against 1 for the noise pair. It also flipped 3 diagnosis runs, against 0 for the noise pair.

What moved, and what did not:

- K001 and K008 gain in both sets. Both are `cart` scaled to zero. The dev notes (`docs/observation-facts-dev-notes.md`) found Jev naming `frontend` there because `cart`'s readiness sat in a different list from its request metrics. `facts` puts the deployment's "ready R of D (baseline available A of B)" on the `cart` service row, which is the join the notes described.
- K007 diagnosis drops in both sets (confirm 3 → 2, dev 3 → 1), and localization stays at 1/3. `valkey-cart` emits no spans, so there is no edge to put its readiness next to (dev notes, "some edges are missing"). `facts` cannot help localization here by design.
- K002, K009 and the K010 false alarms do not move. These are the multi-hop and "is it still happening?" cases. Phase B (`time-split`) was meant for these. `facts` was not expected to fix them, and it did not.
- Opus is at the ceiling on RCA and diagnosis with either layout, so it cannot show a layout effect there. Its false alarms (K002, K006, K010 all 3/3) are the same with both layouts.

## Size and cost

| Layout | Request bytes, mean / max (confirm) | Jev input tokens (confirm) | Jev cost (confirm) |
|---|---|---|---|
| raw | 7,008 / 13,363 | 1,176,975 | $0.049 |
| facts | 13,381 / 21,748 | 1,945,262 | $0.082 |

`facts` roughly doubles the request size and costs about 65% more Jev input tokens. Every request stays under the 64k limit (max 21,748 bytes over both sets).

## Limits

- 18 fault runs per set, 3 per scenario. The gain is 2 runs on confirm. The design (Risks) said beforehand that a change of 2-3 runs is noise. The measured same-day noise is lower (0 flips on confirm, 1 on dev). This page reports both readings, and neither is a significance test.
- These are replays: no MTTD, no recovery, no MTTR. Live runs could differ if Jev's answer timing changes which poll is scored.
- The baseline window ends at injection (design, Risks). `facts` makes "seconds before window end" easy to read. The K002/K010 rows did not move, so the confound did not help here.
- `jev-1.13.0` answered both source sets live. All replays here went to `jev-latest` on 2026-11-02, so compare replay columns with each other, not with the live column.

## Gate for Phase B (task 4.1)

Design D6: start Phase B only if Phase A, or the recent-window question, changes K002/K010 or RCA beyond noise on the confirm set, or if the user asks for it anyway.

- K002/K010: no change with `facts`. `strict-incident` removed one K010 false alarm on confirm and none on dev, and it lost a K000 run and a detection. Not beyond noise.
- RCA: +2 on confirm (0 lost), against 0 same-day flips. That is above the measured noise but inside the pre-stated "2-3 runs is noise" band.

Decision (task 4.1): no-go. The gate is not met, and we did not start Phase B (`time-split`). `facts` stays available as an opt-in layout (`--observation facts`). `raw` stays the default.
