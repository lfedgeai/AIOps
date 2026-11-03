# Results: does option order change Jev's answers? Tiers 1-4 (2026-11-02)

Jev's docs say jev-1.13 "leans toward the option that comes first" (https://docs.typesafe.ai/model-jaggedness/jev-1.13.md, read 2026-11-02). Every result so far used one fixed option order. This probe asks the same observations twice: once in the baseline order, once with every choice question's options reversed (`prompts/<tier>/baseline-reversed.yaml`, the same text word for word). Yes/no questions are unchanged. The figures come from the records below, through `docs/analysis/option_order.py`.

## Short answer

The bias is real on the action question, and small. With the options reversed, 34 of 631 Tier 3/4 action answers changed (5%), and 33 of those 34 moved to an option listed earlier. The same prompt asked a day apart changed 12 of 631 answers, split both ways. On Tier 1, 7 of 140 action answers changed, all toward an earlier option, against 0 of 140 for the same prompt. Component, diagnosis and severity answers changed no more than the run-to-run noise. No headline score moved beyond noise, and no published figure needs revising. But option order is part of what this benchmark measures, and the order is now fixed and recorded (`prompt.sha256` covers it).

## Sources

All calls went to the Jev API (`jev-latest`, answered by `jev-1.14.0`) on 2026-11-02: 1,590 calls, 0 errors, 0 invalid replies, about $0.25 estimated from the `JEV_PRICE_*` values in `.env` (TypeSafe's published price, `docs/jev-api-notes.md`).

| Tier | Baseline (2026-11-02) | baseline-reversed | Noise floor: baseline (2026-11-01) |
|---|---|---|---|
| 1, seed 1000 / 2000 | `results/20261102-151614-70runs.jsonl`, `...151642-70runs.jsonl` | `...152141-70runs-prompt-baseline-reversed.jsonl`, `...152210-...` | `results/20261101-182620-70runs.jsonl`, `...182121-70runs.jsonl` |
| 2, dev / confirm | `results/tier2/replay/20261102-131643-31c6`, `...131645-49c3` | `...132210-abb9`, `...132213-dc85` | `results/tier2/replay/20261101-162634-0735`, `...162641-2d70` |
| 3/4, dev / confirm | `results/tier3/replay/20261102-131652-587a`, `...131914-5400` | `...132221-2217`, `...132454-145b` | `results/tier3/replay/20261101-162733-cf70`, `...163423-577f` |

Tiers 2-4 are replays of the stored observations (`results/tier3/10.3-combined`, `results/tier4/6.2-combined`, and the two Tier 2 directories). Tier 1 is live, but each seed gives the same 70 observations every time.

The noise floor is the same prompt and order asked on two days, 2026-11-01 and 2026-11-02, both answered by `jev-1.14.0`.

## Per-answer changes (Tier 3/4, dev and confirm pooled, 631 polls)

| Question | Changed, reversed order | of which toward an earlier option | Changed, same order a day apart | Mean probability shift, reversed / same |
|---|---|---|---|---|
| recommended_action | 34 (5.4%) | 33 | 12 (1.9%), 6 toward earlier | 0.033 / 0.014 |
| affected_component | 21 (3.3%) | 4 | 19 (3.0%), 9 toward earlier | 0.041 / 0.024 |
| diagnosis | 8 (1.3%) | 7 | 2 (0.3%), 0 toward earlier | 0.012 / 0.008 |

On Tier 1 (140 runs, both seeds):

| Question | Changed, reversed | of which toward earlier | Changed, same order |
|---|---|---|---|
| action | 7 | 7 | 0 |
| severity | 4 | 4 | 3 |
| probable_cause | 13 | 4 | 4 |

The probability shift is the total-variation distance between the two answers' option probabilities, averaged over polls. Tier 2 has only 24 runs. Its action question changed 6 of 24 (5 toward earlier) against 5 of 24 for the same order. That is too few to read.

In the baseline order the actions run from mild to drastic (`no_action`, `investigate`, ... `escalate`). Reversed, `escalate` comes first, and the changed answers moved toward the more drastic end.

## Scores

| | baseline | baseline-reversed |
|---|---|---|
| Tier 1 PASS, seed 1000 | 36/70 | 36/70 |
| Tier 1 PASS, seed 2000 | 34/70 | 32/70 |
| Tier 1 action, seed 1000 / 2000 | 58/70, 59/70 | 55/70, 55/70 |
| Tier 2 confirm: action | 14/18 | 11/18 |
| Tier 2 confirm: localization | 12/18 | 13/18 |
| Tier 3/4 confirm: RCA / diagnosis / remediation correct | 10, 9, 18 of 18 | 10, 10, 17 of 18 |
| Tier 3/4 dev: RCA / diagnosis / remediation correct | 7, 9, 18 of 18 | 10, 9, 18 of 18 |
| Detection and false alarms, every tier | unchanged | unchanged |

Comparison tables: `results/tier3/20261102-order-probe-compare.md`, `results/tier2/20261102-order-probe-compare.md`, `python report.py --compare ... --by-prompt` for Tier 1.

The baseline order scores slightly higher on action on every tier. That fits the bias if the expected actions are mostly mild ones, which the baseline order lists first. We did not check this per scenario. The Tier 3/4 dev RCA gain (7 to 10) is not repeated on the confirm set (10 to 10).

## What this means

- Keep the baseline order, and say it is fixed. Changing it now would break comparison with every earlier result. The reports already hash it.
- Averaging over orders is possible but not worth the calls yet. Asking each poll in both orders and averaging the probabilities would remove the bias. It doubles the calls to fix about 5% of action answers, which is less than any other open error.
- This does not explain the main errors. Blaming the caller instead of the failing component, and calling recovered failures incidents, do not change with order. The next step stays the observation layout (`openspec/changes/archive/2026-11-02-observation-facts`).

## Limits

- One reversed order, not every permutation. A bias toward the middle or last options would not show up.
- The noise floor compares two days. Within-day repeats might be quieter.
- The "toward earlier" count says which way an answer moved, not whether the move was right.
