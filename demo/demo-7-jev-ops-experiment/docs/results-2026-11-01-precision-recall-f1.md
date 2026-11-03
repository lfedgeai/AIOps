# Results: precision, recall and F1 across all four tiers (2026-11-01)

This page answers the first of the next steps posted on [lfedgeai/AIOps#41](https://github.com/lfedgeai/AIOps/issues/41) (2026-10-27): "Calculate F1 score / recall / precision". We made no new runs. Every figure comes from run records already used in the earlier results pages.

## Sources

| Tier | Records | Command |
|---|---|---|
| 1 | `results/20261018-134025-70runs.jsonl` (Jev, logs), `results/20261018-134058-70runs-nologs.jsonl` (Jev, no logs), `results/20261005-224548-70runs-claude-sonnet.jsonl`, `results/20261005-224835-70runs-claude-opus.jsonl`, `results/20261005-230609-70runs-ollama-qwen3-8b.jsonl` | `python report.py --metrics <files>` |
| 2 | `results/tier2/20261012-190447-369b/`, `results/tier2/20261012-195728-7983/` | script over the run records (see Tier 2 below) |
| 3 | `results/tier3/20261018-070851-60ee/` (fault runs) and `results/tier3/20261018-100756-41e8/` (control), combined as `results/tier3/10.3-combined/` | `python -m tier3 scorecard <dir>` |
| 4 | Jev: `results/tier4/6.2-combined/`. Opus (through `claude -p`): `results/tier4/7.5-combined/` | `python -m tier4 scorecard <dir>` |

Tier 1 runs used seed 1000, 70 runs each. The 2026-10-05 Jev batch gives almost the same figures as the 2026-10-18 batch with logs ([results-2026-10-18-tier1-jev.md](results-2026-10-18-tier1-jev.md)), so we leave it out here.

## Tier 1: incident detection

We treat the decision as a detector. An incident is any expected severity other than `normal`. The engine detects one when it answers a severity other than `normal`. Only the `healthy` scenario is a negative (10 runs), so precision and the false positive rate rest on few runs.

| | Jev (logs) | Jev (no logs) | Claude Sonnet | Claude Opus | Ollama qwen3:8b |
|---|---|---|---|---|---|
| TP / FP / FN / TN | 60 / 1 / 0 / 9 | 50 / 0 / 10 / 10 | 56 / 0 / 4 / 10 | 60 / 0 / 0 / 10 | 60 / 0 / 0 / 10 |
| Precision | 60/61 | 50/50 | 56/56 | 60/60 | 60/60 |
| Recall | 60/60 | 50/60 | 56/60 | 60/60 | 60/60 |
| F1 | 0.99 | 0.91 | 0.97 | 1.00 | 1.00 |

Detection is easy at Tier 1. The drop without logs is the `log_only_errors` scenario, which shows the problem only in the logs.

## Tier 1: the operational decision, per field

Macro F1 averages F1 over the classes, so rare classes count as much as common ones. Human review treats `yes` as the positive class.

| | Jev (logs) | Jev (no logs) | Claude Sonnet | Claude Opus | Ollama qwen3:8b |
|---|---|---|---|---|---|
| Severity macro F1 | 0.90 | 0.65 | 0.70 | 0.94 | 0.67 |
| Action macro F1 | 0.79 | 0.54 | 0.82 | 0.90 | 0.87 |
| Human review precision | 37/58 | 30/50 | 13/15 | 25/35 | 36/54 |
| Human review recall | 37/40 | 30/40 | 13/40 | 25/40 | 36/40 |
| Human review F1 | 0.76 | 0.67 | 0.47 | 0.67 | 0.77 |
| Probable cause macro F1 (diagnosis, not in PASS/FAIL) | 0.42 | 0.33 | 0.48 | 0.60 | 0.41 |

Jev's human-review recall is high and its precision is low: it asks for a person in 21 of the 30 runs where none is needed. Sonnet does the opposite. Per-class tables and confusion matrices come out of `python report.py --metrics`.

## Tiers 3 and 4: run-level incident detection

A fault run (an incident scenario, K001 to K010 except the no-incident ones) is a positive. A run of a no-incident scenario (K002, K004, K006, K010) or of the healthy control K000 is a negative. A run counts as detected when any poll reported an incident, which matches the LF Edge `detected` field. F1 is 2TP / (2TP + FP + FN).

| | Tier 3 Jev | Tier 4 Jev | Tier 4 Opus |
|---|---|---|---|
| TP / FP / FN / TN | 18 / 4 / 0 / 18 | 18 / 5 / 0 / 17 | 18 / 7 / 0 / 15 |
| Precision | 18/22 | 18/23 | 18/25 |
| Recall | 18/18 | 18/18 | 18/18 |
| F1 | 36/40 (0.90) | 36/41 (0.88) | 36/43 (0.84) |

Every engine caught every fault run. The differences are all false positives on no-incident runs. Opus raised more of them, which is also why it made fewer engine calls (a run stops polling at its first incident report).

Tier 4 runs the same faults as Tier 3 and carries out one action after the last poll. The action cannot change detection, because detection is scored before it.

## Tier 2: incident detection

Tier 2 asks Jev once per run. The 24 runs are 20 fault runs (F001, F003 to F006, four each) and four runs of the healthy control F000.

| | Tier 2 Jev |
|---|---|
| TP / FP / FN / TN | 20 / 4 / 0 / 0 |
| Precision | 20/24 |
| Recall | 20/20 |
| F1 | 40/44 (0.91) |

Here F1 hides the main result. Jev reported an incident in all 24 runs, healthy ones included, so it never told a healthy system from a faulty one at Tier 2. A detector that always says "incident" gets the same scores. A short script computed the figures over `results/tier2/20261012-190447-369b/` and `results/tier2/20261012-195728-7983/`, counting a run as detected when Jev's reply was `ok` and `incident_detected` was true. The Tier 2 CLI does not print them.

## Limits

- Few runs: 18 positives and 22 negatives at Tiers 3 and 4, 10 negatives at Tier 1. A single run moves the precision a lot.
- Four of the Tier 3 "no-incident" scenarios are no-incident because of how those faults behave on this demo version and cluster ([results-2026-10-18-tier3-jev.md](results-2026-10-18-tier3-jev.md)). Other harnesses may count them as incidents, which would change precision and recall.
- Detection is per run, not per poll. Poll-level false positives are in the scorecards' `FP polls` column.
- Tier 2 has only four negatives, all from one healthy scenario.
- F1 ignores true negatives. Read it with the TP / FP / FN / TN row, as the Tier 2 case shows.
- Localization (the group's RCA), diagnosis and action are single-label PASS/FAIL per fault run. Their accuracy is in the earlier pages. Precision and recall per label would need more runs per label than we have.
