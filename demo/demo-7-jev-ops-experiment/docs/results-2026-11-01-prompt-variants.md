# Results: prompt variants against the baseline wording, Tiers 1-4 (2026-11-01)

This page answers the "better prompting" next step on [lfedgeai/AIOps#41](https://github.com/lfedgeai/AIOps/issues/41). Can a change to the question wording alone, with the same observations, model and scoring, improve Jev's decisions? We tested two hand-written variants against the baseline wording on every tier (change `openspec/changes/archive/2026-11-02-jev-prompt-variants`). Every figure here is computed from the run records named below.

## Short answer

No variant clearly beats the baseline on the held-out (confirm) sets. Detection, the operational decision and the action do not move. The one consistent change is localization at Tiers 3/4 with `evidence-options`, which comes with a drop in diagnosis on the same runs. The samples are small (18 fault runs at Tiers 3/4, 18 runs at Tier 2), so a difference of two or three runs is within run-to-run noise. For that reason we did not run task 7.4 (a live Tier 3 batch with a winning variant).

## The variants

| Variant | Style | What it changes |
|---|---|---|
| `baseline` v1 | wording | The questions used in every earlier results page. |
| `strict-incident` v3 (Tiers 1, 3) / v1 (Tier 2) | policy stated | One question only. Tiers 2/3: the incident question asks about the end of the observation window and says a failure that has already recovered is not an incident now. Tier 1: the human-review question states a rule (review when the diagnosis cannot be made with confidence, or when the fix carries risk beyond a routine, reversible step). |
| `evidence-options` v3 | wording | Option descriptions only: what each component, cause and failure mode looks like in the evidence. Question texts are the baseline's. |

We wrote the variants from the dev sets only. Reviewers other than the author then reviewed them twice before any live call (task 6.3, commits `6cc5343` and `278d41d`). Each file's `rationale` lists the risks the reviews named. The reports label `policy stated` columns because they add an operational rule, not only clearer wording.

## Sources

All calls went to the Jev API (`api.typesafe.ai/v1/systemone`, `jev-latest`, answered by `jev-1.14.0`), on 2026-11-01. 0 errors and 0 invalid replies in 2,385 calls. Estimated cost $0.40 in total (from the `JEV_PRICE_*` prices in `.env`. Jev reported no cost.).

| Tier | Dev (variants written on) | Confirm (held out) | Variant records |
|---|---|---|---|
| 1 | seed 1000, 70 runs | seed 2000, 70 runs | `results/20261101-182620-70runs.jsonl` (dev baseline), `...182029-...-strict-incident.jsonl`, `...182055-...-evidence-options.jsonl`; confirm `...182121-70runs.jsonl`, `...182146-...-strict-incident.jsonl`, `...182212-...-evidence-options.jsonl` |
| 2 | `results/tier2/20261012-190447-369b` (6 runs) | `results/tier2/20261012-195728-7983` (18 runs) | replays `results/tier2/replay/20261101-1626*`; report `results/tier2/20261101-prompt-variants-compare.md` |
| 3/4 | `results/tier3/10.3-combined` (40 runs, 351 polls) | `results/tier4/6.2-combined` (40 runs, 280 polls) | replays `results/tier3/replay/20261101-16*`; report `results/tier3/20261101-prompt-variants-compare.md` |

Tier 1 runs live: each run generates its observation from the seed, so every prompt sees the same 70 observations per seed. Tiers 2-4 are replays: we send the stored observations of earlier live runs again under each prompt, and ask a fresh baseline replay at the same time. Compare variants with the replayed baseline, not the original live one.

Commands: `python report.py --compare <files>`, `python -m tier2 compare ... --dev ... --confirm ...`, `python -m tier3 compare ... --dev ... --confirm ...`.

## Confirm sets

### Tier 1 (seed 2000, 70 runs)

| | baseline | strict-incident v3 | evidence-options v3 |
|---|---|---|---|
| PASS (severity, action, human review) | 33/70 (47%) | 33/70 (47%) | 32/70 (46%) |
| severity | 57/70 | 57/70 | 56/70 |
| action | 59/70 | 60/70 | 59/70 |
| human_review | 47/70 | 48/70 | 47/70 |
| probable_cause (scored separately) | 45/70 | 44/70 | 47/70 |

degraded, contradictory and hung_worker are 0/10 under every prompt, on both seeds. Wording does not reach those failures.

### Tier 2 (18 runs, replay)

| | baseline (replay) | strict-incident v1 | evidence-options v3 |
|---|---|---|---|
| detection | 16/18 | 16/18 | 16/18 |
| localization | 12/18 | 12/18 | 12/18 |
| diagnosis | 11/18 | 11/18 | 11/18 |
| action | 12/18 | 13/18 | 14/18 |
| FP runs (healthy control) | 2/3 | 2/3 | 2/3 |
| detection F1 | 0.94 | 0.94 | 0.94 |

### Tiers 3/4 (Tier 4 Jev records, 40 runs, 280 polls, replay)

| | baseline (replay) | strict-incident v3 | evidence-options v3 |
|---|---|---|---|
| detect (fault runs) | 18/18 | 18/18 | 18/18 |
| RCA (localization) | 10/18 (56%) | 12/18 (67%) | 13/18 (72%) |
| diagnosis | 10/18 (56%) | 9/18 (50%) | 8/18 (44%) |
| remediation correct (action) | 18/18 | 18/18 | 18/18 |
| FP runs (no-incident runs) | 5/22 | 5/22 | 5/22 |
| FP polls | 5/260 | 5/260 | 5/260 |
| detection F1 | 0.88 | 0.88 | 0.88 |
| censored runs | 0/40 | 0/40 | 0/40 |

## Dev sets

| Tier 1 (seed 1000) | baseline | strict-incident v3 | evidence-options v3 |
|---|---|---|---|
| PASS | 36/70 (51%) | 37/70 (53%) | 31/70 (44%) |
| severity | 62/70 | 63/70 | 57/70 |
| critical (scenario PASS) | 10/10 | 10/10 | 7/10 |

The baseline here is a fresh run on `jev-1.14.0`. The dev records the variants were written on (`results/20261018-134025-70runs.jsonl`) were answered by `jev-1.13.0`, 37/70.

| Tiers 3/4 dev (`10.3-combined`) | baseline (replay) | strict-incident v3 | evidence-options v3 |
|---|---|---|---|
| RCA | 7/18 | 9/18 | 13/18 |
| diagnosis | 9/18 | 7/18 | 6/18 |
| remediation correct | 18/18 | 17/18 | 18/18 |
| FP runs / FP polls | 4/22, 47/330 | 4/22, 47/330 | 4/22, 47/330 |

Tier 2 dev (6 runs): no dimension moves by more than one run.

## What changed and what did not

strict-incident did not remove a single false alarm. We wrote it to stop Jev calling recovered failures incidents (K002, K010 at Tiers 3/4). The wording does reach Jev: on those runs the probability of "incident" falls from about 0.85 to about 0.70-0.78. But it never falls below the 0.5 threshold, so the yes/no answer and every detection figure stay the same. Over the 351 dev polls only 2 answers flip. The threshold is our choice (`docs/jev-api-notes.md`), and moving it would be a change to the scoring, not to the prompt.

evidence-options moves localization up and diagnosis down at Tiers 3/4. Per scenario (3 runs each), confirm then dev:

| Scenario | RCA: baseline / evidence-options | diagnosis: baseline / evidence-options |
|---|---|---|
| K001 cart scaled to zero | 2 → 3, 0 → 3 | 1 → 1, 0 → 0 |
| K007 valkey-cart scaled to zero | 1 → 1, 0 → 1 | 3 → 1, 3 → 0 |
| K008 cart to zero under load | 1 → 3, 1 → 3 | 0 → 0, 0 → 0 |
| K010 write to /tmp, OOMKilled | 1 → 3, 1 → 1 | 3 → 3, 2 → 2 |

The "origin is the component itself, not the caller that reports errors" wording does what it was written for on cart. On K007 it costs diagnosis: with the wording, Jev answers `network_blocked` instead of `service_unavailable` on 2 of 3 confirm runs. Net, on the confirm set: RCA +3, diagnosis −2, out of 18.

Nothing else moves beyond noise. Detection, actions, the Tier 1 operational decision and the Tier 2 dimensions change by at most two runs. At Tier 1 `evidence-options` scores lower on the dev seed (31 vs 36, mostly `critical` severity) but not on the confirm seed (32 vs 33).

## Limits

- Sample size. 18 fault runs at Tiers 2-4 and 10 runs per scenario at Tier 1. We ran no significance test. Do not read a two- or three-run difference as an effect.
- Replay, not live. Tier 2-4 variants answered stored observations. Replay has no wall-clock MTTD and carries out no remediation, so Tier 4 recovery and MTTR were not measured for any variant. No run was censored (a run whose live polling stopped at a detection the variant does not make), so the replayed detection figures are complete for the stored polls.
- One model version. Every variant figure is `jev-1.14.0`. Earlier results pages used `jev-1.13.0`.
- The reviews read the test sets. Both reviewers checked the variants against the expected outcomes, which nudges the wording toward the catalogue. For that reason we stopped the reviews after two rounds (`openspec/changes/archive/2026-11-02-jev-prompt-variants/tasks.md`, 6.3).
- Two variants only. These results say these two rewrites did not help. They do not show that no wording would.
