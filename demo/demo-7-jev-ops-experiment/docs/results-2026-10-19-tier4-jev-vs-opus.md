# Results: Jev vs Claude Opus on Tier 4, closed-loop remediation on kind (2026-10-19)

This page compares Jev with a Claude Opus 5.5 baseline. Both went through the same Tier 4 path: the same cluster, the same eleven scenarios, the same observations and the same four questions. After the first poll that reports an incident, the harness carries out the engine's action once, then watches for 360 seconds. Only the engine that answers is different. The Jev batch is written up on its own in [results-2026-10-19-tier4-jev.md](results-2026-10-19-tier4-jev.md). That page explains how a run works and how it differs from the LF Edge harness, and none of that is repeated here.

Every figure comes from the 80 run records listed below, through `python -m tier4 scorecard` and `export-lfedge`.

**Every MTTR on this page is single-shot recommendation MTTR**: one answer, carried out once, with no feedback or retry. It is not operational MTTR.

## What ran

| | Jev | Claude Opus |
|---|---|---|
| Engine | `jev` (`jev-api`), model `jev-1.13.0` on every poll | `claude-opus` through the `claude` CLI (`claude -p`, transport `claude-cli`, CLI 2.1.303 (Tier 1 used 2.1.289), effort `medium`), model `claude-opus-5-5` on every poll |
| Fault runs | `results/tier4/20261018-232405-5ada/` (2026-10-19 01:24 to 07:00) | `results/tier4/20261019-070552-2d0e/` (09:05 to 14:29) |
| Control runs | `results/tier4/20261019-050044-6cc4/` (07:00 to 09:05) | `results/tier4/20261019-122929-7076/` (14:29 to 16:35) |
| Code | `f9960e6` / `8e14998` | `8e14998` (fault runs) / `6fef6c4` (control runs). Neither commit changes code. They only change docs and `tasks.md` |
| Exports | `results/tier4/6.2-combined-scorecard.md`, `6.2-combined-lfedge.jsonl` | `results/tier4/7.5-combined-scorecard.md`, `7.5-combined-lfedge.jsonl` |
| Runs | 40/40 `ok` | 40/40 `ok` |
| Engine calls | 280, none failed, none invalid | 250, none failed, none invalid |

No `ANTHROPIC_API_KEY` was set, so Opus answered through the `claude -p` fallback, with no tools, no settings, no MCP and no session, run from an empty directory (see [llm-baseline-notes.md](llm-baseline-notes.md)). The CLI wraps the model in the Claude Code harness, so these results are not directly comparable with Anthropic API runs. Every record and the LF Edge `approach` field (`claude-opus:claude-opus-5-5`) say which engine answered.

We searched every stored request of the 40 Opus runs for the ten fault names, `jev-fault`, `flagd`, `feature_flag`, the scenario ID, `ground_truth`, `expected`, `remediation`, `recovered`, other harness terms and the expected diagnosis and action labels (other than `none`, `unknown` and `no_action`). We found none. The offline whole-path test also checks the Opus prompt itself on every scenario (`tests/tier3/test_claude_engine.py`). The Opus engine makes fewer calls than Jev (250 against 280) because a run stops polling at its first incident report, and Opus reported incidents on more no-incident runs.

## The four verdicts

| | Jev | Opus |
|---|---|---|
| Detect (fault runs) | 18/18 | 18/18 |
| RCA (localization) | 10/18 | **18/18** |
| Diagnosis | 10/18 | **18/18** |
| Remediation correct | 18/18 | 18/18 |
| MTTD median (min-max) | 16.7 s (15.9-31.9) | 22.6 s (20.5-29.7) |
| False-positive runs (no-incident scenarios and control) | 5/22 | 7/22 |
| False-positive polls | 5/260 | 7/232 |
| Detection precision | 18/23 | 18/25 |

Opus named the right component and diagnosis on every incident run. Jev missed the component on 8 runs, most often by naming frontend, the caller that logged the errors, instead of cart or the dependency behind it.

MTTD is measured to the scored poll. Opus was scored on the first poll (about 15 s after the fault) on all 18 incident runs, and Jev on 16 of 18 (the second poll on the other two). So most of the MTTD difference comes from answer time: Jev answered in 360-1,011 ms (median 429 ms), Opus through the CLI in 3.5-15.4 s (median 4.8 s).

False positives, by scenario:

| Scenario | Jev | Opus |
|---|---|---|
| K000 healthy control | 0/10 | 0/10 |
| K002 kill_pod | 3/3 (frontend, `investigate`) | 3/3 (cart, `pod_restart`, `investigate`) |
| K004 network_partition | 0/3 | 0/3 |
| K006 config_corruption | 0/3 | 3/3 (cart, `pod_restart`, `investigate`) |
| K010 pvc_full | 2/3 (one `scale_up`, one `investigate`) | 1/3 (cart, `oom_killed`, `investigate`) |

K002, K004, K006 and K010 expect no incident on this cluster: the fault heals itself or has no effect the service-level checks can see (see [k8s-testbed-notes.md](k8s-testbed-notes.md)). On K002 and K006, Opus reported the pod restart it saw on cart, which is a correct reading of the observation even though our hand-written expectation scores it as a false positive. Jev named frontend on K002 and saw nothing on K006.

## Remediation

| | Jev | Opus |
|---|---|---|
| Incident runs | 18 | 18 |
| Executed | 7/18 | **13/18** |
| Not executable (`investigate`) | 11/18 | 5/18 |
| Recovered by action | 4/18 | **13/18** |
| Recovered without action | 2/18 | 3/18 |
| Not recovered | 12/18 | 2/18 |
| Relapsed | 0/18 | 0/18 |
| MTTR median (min-max), single-shot | 16.2 s (16.0-18.3, n=7) | 23.4 s (20.6-28.1, n=13) |
| Verified MTTR median (min-max), single-shot | 106.2 s (106.0-106.3, n=4) | 110.7 s (23.4-260.8, n=13) |
| False-alarm actions | 1/22 | 0/22 |
| Disturbed healthy system | 1/22 | 0/22 |

MTTR is fault to remediation written (the LF Edge `mttr_seconds`). Verified MTTR is fault to the start of the final run of passing service-level checks (`mttr_verified_seconds`), for recovered-by-action runs only. Recovery without action is never counted as the engine's MTTR.

What each engine answered on the scored poll, and what happened:

| Scenario | Jev (component, action: outcome) | Opus (component, action: outcome) |
|---|---|---|
| K001 scale_zero | cart, `scale_up` x2: recovered by action; frontend, `scale_up` x1: not recovered | cart, `scale_up` x3: recovered by action |
| K003 memory_limit | cart, `investigate` x3: not recovered | cart, `rollback_change` x3: recovered by action |
| K005 readiness_probe_fail | cart, `investigate` x3: not recovered | cart, `rollback_change` x1: recovered by action; cart, `investigate` x2: not recovered |
| K007 dependency_removal | valkey-cart, `scale_up` x1: recovered by action; frontend-proxy, `investigate` x2: not recovered | valkey-cart, `scale_up` x3: recovered by action |
| K008 replica_overload | cart, `scale_up` x1: recovered by action; frontend, `scale_up` x2: not recovered | cart, `scale_up` x3: recovered by action |
| K009 node_taint | frontend, `investigate` x3: 2 recovered without action, 1 not recovered | the tainted worker, `investigate` x3: recovered without action |

Verified MTTR by scenario, Opus (single-shot): K001 174.8 s (127.7-260.8), K003 26.7 s (23.4-27.2), K005 28.1 s (n=1), K007 110.6 s (99.0-142.6), K008 115.0 s (110.7-155.7). The rollbacks on K003 and K005 recover almost at once, because `rollout undo` brings back the old template, and the service-level checks passed from the first check after the undo and kept passing. Scaling from zero (K001, K007, K008) waits for a new pod to start and pass its checks. Jev's four recoveries were all `scale_up` from zero, at 106 s each. On the same kind of fix, Opus's times are close to Jev's but spread more widely.

## What this means

- The gap is in the component and the action, not in detection. Both engines detected every fault. Opus named the failing component every time, where Jev often named the caller that logged the errors. On the three `scale_up` scenarios (K001, K007, K008), that is the whole difference: 9/9 recovered by action for Opus against 4/9 for Jev.
- Opus used `rollback_change`, and Jev never did. On K003 and K005, where a template change caused the fault, both engines had the right component and diagnosis every time. Jev answered `investigate` on all six runs. Opus answered `rollback_change` on four of the six and fixed each of those. Both engines answered `investigate` on every K009 run, and neither ever chose `restart_service` or `escalate`.
- Jev is faster to answer. Jev took about 0.4 s per call, Opus through the CLI about 5 s. That shows up as 16 s against 23 s to the first action. With a 15-second poll interval it does not change which poll is scored. A poll starts 15 s after the previous one started. If an answer takes longer than that, the next poll starts at once. This happened on 2 of the 250 Opus polls, both on control runs (15.4 s each).
- Opus raised more false alarms, and none led to an action. All 7 Opus false-positive runs answered `investigate`, so nothing was done. The harness carried out Jev's one false-alarm `scale_up` (K010).

Three runs per fault are too few to give rates. Read the results as a direction: on this testbed, with these questions and a single action, Opus fixed 13 of 18 incidents and Jev fixed 4.

## Cost

The two costs come from different sources and must not be compared as like for like.

- Jev reports no cost. With the user-set prices (`JEV_PRICE_INPUT_PER_MTOK=0.042`, `JEV_PRICE_OUTPUT_PER_MTOK=0`, USD per million tokens, not a published price), the 280 calls (1,176,975 input and 108,231 output tokens) cost an estimated USD 0.049.
- Opus: the `claude` CLI reported USD 12.83 for the 250 calls (`reported:claude-cli`, 1,637,118 input and 67,248 output tokens). This is the CLI's own figure. On a subscription plan it is notional, not a bill. The CLI caches the prompt, so its reported cost runs about 1.8 times what the same call would cost through the API, and its token counts include the Claude Code harness's own context ([llm-baseline-notes.md](llm-baseline-notes.md)). No API run was made, and no price is built into the code.

## Known limits

- Everything in [results-2026-10-19-tier4-jev.md](results-2026-10-19-tier4-jev.md#known-limits) applies: one laptop, one cluster, one demo version, one load level, three runs per fault, and our own judgement for the expected answers and accepted actions.
- Opus ran through the CLI, not the API, so latency and cost include the CLI harness.
- The two batches ran one after the other on the same cluster, not interleaved. The cluster was the same, but it drifts over a day: pod placement was rebalanced between runs when needed (in both batches).
- One Opus control run (`K000-5.json`) ended `not_recovered`, with no engine action: about once a minute (every fourth 15-second check) the recovery check found "no service telemetry in the window", which is the known span-metrics gap on kind. That is measurement noise and does not change any verdict or MTTR figure.
- The run files are not in the repository, because `results/` is git-ignored.

## How to repeat

You do not need the Jev batch to run the Opus batch. It is only needed for the comparison.

1. Start the cluster: `./setup-k8s.sh`, then `./health-check-k8s.sh`.
2. Choose the transport. Without `ANTHROPIC_API_KEY`, the engine uses the `claude` CLI, as here. You must install the CLI and log in to it. With the key in `.env`, it uses the Anthropic API, which gives different latency and cost figures.
3. Run the fault batch: `./run-all-remediation-experiments K001 K002 K003 K004 K005 K006 K007 K008 K009 K010 --repeats 3 --engine claude-opus` (about 5.5 hours here).
4. Run the control: `./run-all-remediation-experiments K000 --repeats 10 --engine claude-opus` (about 2 hours here).
5. Run `python -m tier4 scorecard` and `export-lfedge` on the two run directories.
6. If a run is interrupted, run `python -m tier4 reset-faults` before anything else.
