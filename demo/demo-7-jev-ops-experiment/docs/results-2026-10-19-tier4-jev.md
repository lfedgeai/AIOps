# Results: Jev on Tier 4, closed-loop remediation on kind (2026-10-19)

This page summarizes the first full live Tier 4 batch. Tier 4 runs Tier 3's eleven scenarios on the same kind cluster. After the first poll that reports an incident, it carries out Jev's recommended action on Jev's named component, once. Then it watches for 360 seconds to see whether the system recovers. The Tier 3 summary, which uses the same faults and the same questions but never acts, is in [results-2026-10-18-tier3-jev.md](results-2026-10-18-tier3-jev.md).

Every figure on this page comes from the 40 run records listed below (`evaluation`, `polls`, `timing`, `usage` and `remediation` fields), through `python -m tier4 scorecard` and `export-lfedge`.

## What ran

Each fault scenario ran three times, and the control (K000) ran ten times: 40 runs and 280 Jev calls. All 40 runs finished with status `ok`. No run was `contaminated`, no Jev call failed and no reply was invalid.

| | |
|---|---|
| Jev model | `jev-1.13.0` (every poll) |
| Cluster | kind v0.29.0, Kubernetes v1.33.1, kindnet, four nodes, one laptop. OpenTelemetry Demo 3.1.0, Helm chart 0.42.4, as in Tier 3 |
| Load | Locust, 10 simulated users |
| Code | commit `f9960e6` (fault runs) and `8e14998` (control runs). That commit only changes `tasks.md`. |
| Fault runs | `results/tier4/20261018-232405-5ada/`, `./run-all-remediation-experiments K001 ... K010 --repeats 3` (2026-10-19 01:24 to 07:00) |
| Control runs | `results/tier4/20261019-050044-6cc4/`, `./run-all-remediation-experiments K000 --repeats 10` (07:00 to 09:05) |
| Exports | `results/tier4/6.2-combined-scorecard.md`, `6.2-combined-lfedge.jsonl` (both run directories) |

In the single pass from task 6.1 (`results/tier4/20261018-210528-2703/` and `20261018-212756-f595/`), we read the requests by hand and checked every action's record and restore. It is not in the figures below. We also searched every stored request of the 40 runs for the ten fault names, `jev-fault`, `flagd`, `feature_flag`, the scenario ID, `ground_truth`, `expected`, `remediation`, `recovered` and the expected diagnosis and action labels (other than `none`, `unknown` and `no_action`). We found none.

A run took 601 to 813 seconds (median 662). That is Tier 3's run plus the 360-second recovery watch.

## How a run differs from Tier 3

1. Polling is Tier 3's (every 15 s, first poll after 15 s), but every scenario stops at the first poll that reports an incident.
2. The harness carries out that poll's action on that poll's component once, from a fixed allowlist: `scale_up` (replicas to baseline if below it, else one more, never above 3), `restart_service` (rollout restart), `rollback_change` (`kubectl rollout undo`). `investigate`, `escalate` and `no_action` do nothing. The harness acts only on Deployments in the demo namespace.
3. The harness saves the target's state to the ledger before the action and reads it back after.
4. The watch starts when the action is done, or right after the scored poll when the harness carries out nothing. The fault stays injected during the watch, so only the action can fix it. Recovered means every Deployment is at baseline availability with its rollout complete, every application pod is Ready and no service deviates from baseline, for at least the last 60 seconds of the 360-second watch.
5. Reset undoes the action first, then the fault.

Jev is never asked again after the action, so nothing about the action or the recovery can reach a request.

**This is single-shot recommendation MTTR**: one answer, carried out once, with no feedback or retry. It is not operational MTTR.

## Results: the four verdicts

The four verdicts are scored exactly as in Tier 3.

| Scenario | Runs | Detect | RCA | Diagnosis | Remediation correct | MTTD median (min-max) s | FP runs | FP polls |
|---|---|---|---|---|---|---|---|---|
| K000 healthy control | 10 | | | | | | 0/10 | 0/150 |
| K001 scale_zero | 3 | 3/3 | 2/3 | 1/3 | 3/3 | 16.2 (16.1-16.3) | | |
| K002 kill_pod | 3 | | | | | | 3/3 | 3/3 |
| K003 memory_limit | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 16.8 (16.7-16.9) | | |
| K004 network_partition | 3 | | | | | | 0/3 | 0/45 |
| K005 readiness_probe_fail | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 16.8 (16.7-17.0) | | |
| K006 config_corruption | 3 | | | | | | 0/3 | 0/45 |
| K007 dependency_removal | 3 | 3/3 | 1/3 | 3/3 | 3/3 | 30.7 (16.1-31.9) | | |
| K008 replica_overload | 3 | 3/3 | 1/3 | 0/3 | 3/3 | 16.1 (15.9-18.2) | | |
| K009 node_taint | 3 | 3/3 | 0/3 | 0/3 | 3/3 | 16.6 (16.2-17.6) | | |
| K010 pvc_full | 3 | | | | | | 2/3 | 2/17 |
| **All** | 40 | 18/18 | 10/18 | 10/18 | 18/18 | 16.7 (15.9-31.9) | 5/22 | 5/260 |

Detection precision is 18/23. FP polls are fewer than in Tier 3 because a run stops polling at its first incident report. These verdicts are close to Tier 3's (18/18, 8/18, 10/18, 18/18, 4/22 FP runs). They are new answers from new runs, not a copy.

## Results: remediation

| Scenario | Fault runs | Executed | Not executable | Recovered by action | Recovered without action | Not recovered | MTTR median (min-max) s | Verified MTTR median (min-max) s |
|---|---|---|---|---|---|---|---|---|
| K001 scale_zero | 3 | 3/3 | 0/3 | 2/3 | 0/3 | 1/3 | 16.3 (16.2-16.3) | 106.3 (106.3-106.3, n=2) |
| K003 memory_limit | 3 | 0/3 | 3/3 | 0/3 | 0/3 | 3/3 | n/a | n/a |
| K005 readiness_probe_fail | 3 | 0/3 | 3/3 | 0/3 | 0/3 | 3/3 | n/a | n/a |
| K007 dependency_removal | 3 | 1/3 | 2/3 | 1/3 | 0/3 | 2/3 | 16.1 (n=1) | 106.1 (n=1) |
| K008 replica_overload | 3 | 3/3 | 0/3 | 1/3 | 0/3 | 2/3 | 16.2 (16.0-18.3) | 106.0 (n=1) |
| K009 node_taint | 3 | 0/3 | 3/3 | 0/3 | 2/3 | 1/3 | n/a | n/a |
| **All incident runs** | 18 | 7/18 | 11/18 | **4/18** | 2/18 | 12/18 | 16.2 (16.0-18.3, n=7) | **106.2 (106.0-106.3, n=4)** |

No run relapsed. MTTR is fault to remediation written (the LF Edge `mttr_seconds`). Verified MTTR is fault to the start of the final run of passing checks (`mttr_verified_seconds`), for recovered-by-action runs only. The two K009 runs that recovered without action do not count as Jev's MTTR. The evicted pods were rescheduled on the other worker with no action (self-recovery 16.2 and 16.6 s).

What Jev recommended on the scored poll, and what happened:

| Scenario | Runs | Jev answered (component, action) | Harness did | Outcome |
|---|---|---|---|---|
| K001 | 2 of 3 | cart, scale_up | scale cart 0 -> 1 | recovered by action |
| K001 | 1 of 3 | frontend, scale_up | scale frontend 1 -> 2 | not recovered |
| K003 | 3 of 3 | cart, investigate | nothing | not recovered |
| K005 | 3 of 3 | cart, investigate | nothing | not recovered |
| K007 | 1 of 3 | valkey-cart, scale_up | scale valkey-cart 0 -> 1 | recovered by action |
| K007 | 2 of 3 | frontend-proxy, investigate | nothing | not recovered |
| K008 | 1 of 3 | cart, scale_up | scale cart 0 -> 1 | recovered by action |
| K008 | 2 of 3 | frontend, scale_up | scale frontend 1 -> 2 | not recovered |
| K009 | 3 of 3 | frontend, investigate | nothing | 2 recovered without action, 1 not recovered |

On the no-incident scenarios and the control, Jev's scored answer led to one action. In K010 run 1 (`K010.json`), Jev reported an incident on cart (`oom_killed`, `scale_up`), and the harness scaled cart 1 -> 2. That run is `action_on_false_alarm` and `disturbed_healthy_system`: one check right after the action failed, and the next passed. On K010, a failing check can come from the fault's own short error burst as well as from the action (see [k8s-testbed-notes.md](k8s-testbed-notes.md)), so this does not show that the action caused harm. The K002 false alarms and K010 run 2 recommended `investigate`, so the harness did nothing. The control (0/150 polls) and K004 and K006 (0/45 each) had no false alarms and no action.

## What this means

"Remediation correct" was 18/18, but only 4 of 18 incident runs were fixed by the action. Three things explain the gap. All three are visible in the records:

- `investigate` is accepted but does nothing. It is in the accepted list for every incident scenario, and Jev chose it in 11 of 18 runs: every K003, K005 and K009 run and two K007 runs. On K003 and K005, Jev named the right component and diagnosis every time, but it never chose `rollback_change`, the action that undoes the template change.
- The right action on the wrong component fixes nothing. In 3 runs (K001 once, K008 twice) Jev chose `scale_up` but named frontend, the caller that logged the errors, instead of cart, which had no pods. The harness scaled frontend 1 -> 2 and cart stayed at zero.
- When component and action are both right, the fix works and is fast. All 4 runs with the right component and `scale_up` recovered. Verified MTTR was 106 seconds in each: about 16 seconds to the first poll and the action, then about 90 seconds until the service-level checks passed and kept passing.

Jev never chose `restart_service`, `rollback_change` or `escalate` on any of the 280 polls. Three runs for each fault are too few to give rates. Read the results as a direction.

## What is and is not comparable with the group's results

| | LF Edge harness | Tier 4 |
|---|---|---|
| Faults | ten Kubernetes API faults | the same ten, same names, exact restore |
| Agent | tool-calling loop that gathers evidence, writes its own fix and can retry | Jev answers four fixed questions. The harness carries out the action once, from a three-operation allowlist |
| `mttr_seconds` | fault to remediation written | the same |
| `mttr_verified_seconds` | fault to the injected target's spec and readiness check passing, per fault type, with no hold, and never for `kill_pod`, `pvc_full`, `node_taint` | fault to the start of the final run of service-level checks (all Deployments, pods and services back to baseline), held for at least 60 s to the end of a 360 s watch |
| Healthy control | skip-injection mode, false alarms not scored | K000, every poll scored, actions on false alarms recorded |

Compare `mttr_seconds` only as "time to first action". Our verified MTTR is stricter than theirs and is single-shot. Four of our scenarios (K002, K004, K006, K010) expect no incident on this cluster. It is possible that the group's agents were scored on those faults as incidents. In the LF Edge export, K010 run 1 carries an `mttr_seconds` and `mttr_verified_seconds` because an action was written and the checks passed. It is a false-alarm action, and the scorecard does not count it toward Jev's MTTR.

## Cost

Jev reports no cost. With the user-set prices (`JEV_PRICE_INPUT_PER_MTOK=0.042`, `JEV_PRICE_OUTPUT_PER_MTOK=0`, USD per million tokens, as in Tier 3 and not a published price), the 280 calls (1,176,975 input and 108,231 output tokens) cost an estimated USD 0.049. Jev answered in 360 to 1,011 ms (median 429 ms) and returned no reasoning text.

## Known limits

- The runs used one laptop, one cluster, one demo version and one load level, with three runs per fault.
- The recovery check has known noise on kind (span metrics every 60 s, a 120-second window). Self-healing faults (K002, K006, K009, K010) do not measure remediation (see the testbed notes).
- The expected answers, the no-incident scenarios and the accepted-action lists are our judgment, written before the runs. Accepting `investigate` for incident scenarios is generous for the verdict, and it is why "remediation correct" and "recovered by action" differ so much.
- The cluster was created before Jaeger had its own host port, so these runs set `TIER3_JAEGER_URL` to the control plane's IP.
- The run files are not in the repository, because `results/` is git-ignored.

## How to repeat

Run `./setup-k8s.sh` and `./health-check-k8s.sh`. Then run `./run-all-remediation-experiments K001 K002 K003 K004 K005 K006 K007 K008 K009 K010 --repeats 3` and `./run-all-remediation-experiments K000 --repeats 10`, and `python -m tier4 scorecard` and `export-lfedge` on the two run directories. Jev's answers can change between runs, and the live cluster is never the same twice.
