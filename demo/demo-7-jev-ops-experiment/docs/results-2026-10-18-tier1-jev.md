# Results: Tier 1 Jev rerun, with and without logs (2026-10-18)

This page covers Tier 1 only (synthetic data). We ran two 70-run batches against real Jev with the same seed as the 2026-10-05 comparison (1000). One batch had application logs and one used `--no-logs`. Every figure below comes from these run records:

- `results/20261018-134025-70runs.jsonl` (with logs)
- `results/20261018-134058-70runs-nologs.jsonl` (no logs)
- `results/20261005-224157-70runs.jsonl` (the 2026-10-05 Jev batch, with logs, for comparison)

Running `python report.py --compare` on the three files gives the tables. The model that answered was `jev-1.13.0` in all three.

## Same data, same answers

With logs, the 2026-10-18 batch matches the 2026-10-05 batch almost exactly: 37/70 runs passed both times, severity 63/70 both times, action 55/70 against 56/70, human review 46/70 against 47/70, probable cause 42/70 both times. The per-scenario pass counts are identical. Jev's answers on the same observations barely moved in 13 days.

## Logs matter

| | with logs | no logs |
|---|---|---|
| Runs passed | 37/70 | 31/70 |
| Severity | 63/70 | 44/70 |
| Action | 55/70 | 41/70 |
| Human review | 46/70 | 40/70 |
| Probable cause (not in PASS/FAIL) | 42/70 | 33/70 |

The whole drop is in `log_only_errors`: 10/10 with logs, 0/10 without. That scenario is built so the problem shows up only in the logs, so this is the expected result, not a flaw. Without logs, `healthy`, `ambiguous` and `contradictory` each gain a run or two (10/10, 10/10, 1/10 against 9/10, 8/10, 0/10).

## Ambiguous and contradictory (task 14.2)

Both scenarios expect `degraded` / `investigate` / human review `yes` / probable cause `unknown`.

Ambiguous scenario: with logs, Jev gets the decision right on 8/10 runs. The two misses both say `high` instead of `degraded`. Action and human review are right on all 10. It never answers `unknown` for the cause. It names a specific one every time (configuration 3, application bug 2, dependency failure 3, resource exhaustion 2). Without logs, it says `degraded` on all 10 and `unknown` on 2.

Contradictory scenario: 0/10 with logs, 1/10 without. Jev does not answer "investigate" when the signals disagree:

| Answer (severity, action, human review) | with logs | no logs |
|---|---|---|
| degraded, observe, yes | 3 | 5 |
| degraded, observe, no | 3 | 0 |
| critical or high, restart, yes | 3 | 4 |
| high, investigate, yes | 1 | 0 |
| degraded, investigate, yes (expected) | 0 | 1 |

It splits between waiting (`observe`, 6/10 with logs) and acting (`restart`, 3/10). It names `resource_exhaustion` as the cause on 8 of the 10 runs with logs. The expected answer was written by hand. Our reasoning: when signals conflict, a person investigates before anyone changes anything. Jev takes one side of the conflict instead.

## Cost

The estimate uses the user-set prices `JEV_PRICE_INPUT_PER_MTOK=0.042` and `JEV_PRICE_OUTPUT_PER_MTOK=0` (USD per million tokens, not a published price). The cost is $0.0043 for the 70 runs with logs and $0.0025 without. Mean tokens per run: 1,466 in / 187 out with logs, 848 / 186 without.
