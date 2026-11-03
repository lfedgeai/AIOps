# Results: Jev vs Claude through the Anthropic API: Tier 1 (Haiku, Sonnet, Opus) and a Tier 3/4 Opus replay (2026-11-01)

This page repeats the Tier 1 baseline comparison of [results-2026-10-05-tier1-baselines.md](results-2026-10-05-tier1-baselines.md), with three changes:

- The Claude models answered through the Anthropic API (`transport: anthropic-api`), not the `claude -p` command-line tool. The API sends only our system prompt and question. The CLI wraps the model in the Claude Code harness. So these figures are the clean baseline, and the October Claude figures cannot be compared with them directly.
- We added Claude Haiku 5.5 (`--engine claude-haiku`, commit `83aa9dc`).
- Each engine ran on two seeds (1000 and 2000), against Jev's runs from the same day.

Every figure is computed from the run records below.

## Setup

| Engine | Model that answered | How |
|---|---|---|
| Jev | `jev-1.14.0` | Jev API, `jev-latest` |
| Claude Haiku 5.5 | `claude-haiku-5-5` | Anthropic API, effort `medium`, JSON-schema structured output |
| Claude Sonnet 5.5 | `claude-sonnet-5-5` | same |
| Claude Opus 5.5 | `claude-opus-5-5` | same |

Every engine gets the same observation and the same four questions with the same allowed answers (baseline prompt, `prompts/tier1/baseline.yaml`). Seven scenarios, 10 runs each, per seed. A run passes when severity, action and human review are all right. We score probable cause separately. We took the `claude` CLI off `PATH` for these runs, so a missing key would have stopped the run instead of falling back to the CLI. All 560 runs are `ok`: no errors, no invalid replies.

| Seed | Jev | Haiku | Sonnet | Opus |
|---|---|---|---|---|
| 1000 | `results/20261101-182620-70runs.jsonl` | `results/20261101-204207-70runs-claude-haiku.jsonl` | `results/20261101-204337-70runs-claude-sonnet.jsonl` | `results/20261101-204643-70runs-claude-opus.jsonl` |
| 2000 | `results/20261101-182121-70runs.jsonl` | `results/20261101-204908-70runs-claude-haiku.jsonl` | `results/20261101-205038-70runs-claude-sonnet.jsonl` | `results/20261101-205346-70runs-claude-opus.jsonl` |

Commands: `python report.py --compare <four files of one seed>` and `python report.py --metrics <same>`.

## Operational decision

| PASS | Jev | Haiku | Sonnet | Opus |
|---|---|---|---|---|
| seed 1000 | 36/70 (51%) [40–63%] | 32/70 (46%) [35–57%] | 34/70 (49%) [37–60%] | 31/70 (44%) [33–56%] |
| seed 2000 | 33/70 (47%) [36–59%] | 33/70 (47%) [36–59%] | 37/70 (53%) [41–64%] | 30/70 (43%) [32–55%] |

Brackets are 95% intervals from the metrics report. They all overlap: on this benchmark no engine is clearly ahead on the operational decision.

| Per field, seed 1000 / seed 2000 | Jev | Haiku | Sonnet | Opus |
|---|---|---|---|---|
| severity | 62 / 57 | 53 / 53 | 50 / 49 | 63 / 63 |
| action | 58 / 59 | 58 / 63 | 57 / 59 | 58 / 55 |
| human_review | 47 / 47 | 50 / 52 | 41 / 46 | 47 / 45 |
| probable_cause (not in PASS) | 40 / 45 | 45 / 52 | 41 / 51 | 45 / 51 |

Counts out of 70.

## By scenario

PASS out of 10, seed 1000 / seed 2000.

| Scenario | Jev | Haiku | Sonnet | Opus |
|---|---|---|---|---|
| healthy | 9 / 7 | 9 / 9 | 10 / 10 | 10 / 10 |
| degraded | 0 / 0 | 5 / 5 | 6 / 10 | 0 / 0 |
| critical | 10 / 10 | 7 / 9 | 10 / 10 | 10 / 10 |
| ambiguous | 7 / 7 | 0 / 0 | 0 / 0 | 0 / 0 |
| contradictory | 0 / 0 | 1 / 0 | 0 / 0 | 0 / 0 |
| hung_worker | 0 / 0 | 5 / 6 | 8 / 7 | 10 / 10 |
| log_only_errors | 10 / 9 | 5 / 4 | 0 / 0 | 1 / 0 |

The totals are close, but the engines fail different cases:

- Jev is the only engine that passes `ambiguous`. It is also the strongest on `log_only_errors`, where the problem shows only in the logs.
- Opus passes `hung_worker` every time, where Jev never does. Opus and Jev both fail `degraded`. Sonnet and Haiku pass it in half or more of the runs.
- No engine passes `contradictory` (one Haiku run).

## Human review

Human review "yes" is the positive class.

| | Jev | Haiku | Sonnet | Opus |
|---|---|---|---|---|
| seed 1000 precision / recall / F1 | 65% / 92% / 0.76 | 86% / 60% / 0.71 | 92% / 30% / 0.45 | 76% / 62% / 0.68 |
| seed 2000 precision / recall / F1 | 64% / 98% / 0.77 | 89% / 62% / 0.74 | 100% / 40% / 0.57 | 71% / 62% / 0.67 |

Jev asks for review most often: it misses almost no case that needs a person, at the price of more unneeded reviews. Sonnet asks least and misses most of them. Incident detection is easy for every engine (F1 0.98–1.00).

## Latency and tokens (seed 1000, means per run)

| | Jev | Haiku | Sonnet | Opus |
|---|---|---|---|---|
| Tokens in | 1,466 | 1,805 | 1,805 | 1,805 |
| Tokens out | 187 | 245 | 38 | 134 |
| Round trip p50 | 352 ms | 2,083 ms | 1,224 ms | 2,540 ms |
| Round trip p95 | 413 ms | 2,786 ms | 1,623 ms | 4,095 ms |
| Cost | $0.0043 total (estimated) | n/a | n/a | n/a |

Claude output tokens include adaptive thinking. Sonnet at effort `medium` used almost none. Claude cost is `n/a` because no `CLAUDE_*_PRICE_*` prices were set and the API reports no cost. The benchmark does not build in a price. Jev's cost is estimated from the user-set `JEV_PRICE_*` prices.

## Compared with October

October's Claude runs went through `claude -p` (seed 1000): Sonnet 29/70, Opus 34/70. Through the API on the same seed: Sonnet 34/70, Opus 31/70. The transport and the CLI harness differ, so we do not attribute the change to the model. October's Jev run was `jev-1.13.0` (37/70); today's is `jev-1.14.0` (36/70).

## Limits

- 10 runs per scenario per seed. Single-scenario differences of one or two runs are noise.
- One effort level (`medium`) for every Claude model. We did not try other effort levels.
- Tier 1 is synthetic data with hand-written expected outcomes.

## Tier 3/4: Claude Opus through the API, on Jev's stored observations

October's Tier 4 comparison ([results-2026-10-19-tier4-jev-vs-opus.md](results-2026-10-19-tier4-jev-vs-opus.md)) ran Opus live, through `claude -p`. To check that result without the CLI, we replayed the 280 stored polls of the live Jev Tier 4 batch (`results/tier4/6.2-combined`, 40 runs) to `claude-opus` through the Anthropic API: `python -m tier3 replay --engine claude-opus results/tier4/6.2-combined` (records `results/tier3/replay/20261101-190026-1fa0`, report `results/tier3/20261101-opus-api-replay-compare.md`). All 280 calls `ok`, `transport: anthropic-api`, 0 invalid, 0 censored. About 1.65M input and 30k output tokens (cost `n/a` in the records, as above).

| | Jev (replay, `jev-1.14.0`) | Opus API (replay) | Opus CLI (live, October) |
|---|---|---|---|
| detect (fault runs) | 18/18 | 18/18 | 18/18 |
| RCA (localization) | 10/18 | **18/18** | 18/18 |
| diagnosis | 10/18 | **18/18** | 18/18 |
| remediation correct | 18/18 | 18/18 | 18/18 |
| FP runs (no-incident runs) | 5/22 | 9/22 | 7/22 |
| detection precision | 18/23 | 18/27 | 18/25 |

| FP runs by no-incident scenario | Jev (replay) | Opus API (replay) | Opus CLI (live) |
|---|---|---|---|
| K000 healthy control | 0/10 | 0/10 | 0/10 |
| K002 cart pods deleted once | 3/3 | 3/3 | 3/3 |
| K004 network partition | 0/3 | 0/3 | 0/3 |
| K006 unread config variable | 0/3 | 3/3 | 3/3 |
| K010 write to /tmp, OOMKilled | 2/3 | 3/3 | 1/3 |

Through the API, Opus reproduces the October result: it localizes and diagnoses every fault run, where Jev gets 10 of 18. It also raises more false alarms than Jev, mainly on K006, where Opus reports an incident on all 45 polls of the three runs.

How to read this table:
- The two replay columns saw exactly the same observations. The live Opus column saw its own runs, so it differs from them by run-to-run noise as well as transport.
- The replay polls are the ones live Jev made. On no-incident runs Jev rarely detected anything, so all 15 polls were stored, and Opus answered every one. A live Opus run stops polling and acts at its first detection, so false-positive *poll* counts are not comparable with the live batch. The table shows only false-positive *runs*.
- Replay has no MTTD and carries out no action, so recovery and MTTR were not measured for this replay.
