# Results: Jev on Tier 3, LF Edge Kubernetes faults on kind (2026-10-18)

This page summarizes the first full live Tier 3 batch. Tier 3 injects the LF Edge AIOps group's ten Kubernetes fault types into the OpenTelemetry Demo on a local kind cluster. The harness then polls Jev every 15 seconds and scores each answer against a hidden expected answer. Only Jev ran on Tier 3. The Tier 2 summary is in [results-2026-10-12-tier2-jev.md](results-2026-10-12-tier2-jev.md).

## What ran

There are eleven scenarios: one healthy control (K000) and one for each of the ten fault types. Each fault scenario ran three times, and the control ran ten times, which gives 40 runs and 351 Jev calls. All 40 runs finished with status `ok`. No run was `contaminated` (the cluster did not come back to baseline), no Jev call failed, and no reply was invalid.

| | |
|---|---|
| Jev model | `jev-1.13.0` |
| Cluster | kind v0.29.0, Kubernetes v1.33.1, kindnet, four nodes (control plane, one observability worker, two application workers), one laptop (30 GiB memory) |
| Demo | OpenTelemetry Demo 3.1.0, Helm chart 0.42.4, trimmed (no Kafka, accounting, fraud-detection, Grafana) |
| Load | Locust, 10 simulated users |
| Code | commit `ca997c2` (fault runs) and `2be68ad` (control runs). That commit only changes the Dockerfile. |
| Fault runs | `results/tier3/20261018-070851-60ee/`, `./run-all-k8s-experiments K001 ... K010 --repeats 3` |
| Control runs | `results/tier3/20261018-100756-41e8/`, `./run-all-k8s-experiments K000 --repeats 10` |

We used the single-pass runs from task 10.2 (2026-10-14 and the morning of 2026-10-18) to check the requests for leaks by hand. They are not in the figures below.

## How a run works

1. The harness checks the testbed: four Ready nodes, all application Deployments available, application pods on both application workers, Prometheus, span metrics, Jaeger, OpenSearch, Locust running, and no harness objects left over.
2. It waits 180 seconds for the load to settle. These 180 seconds are the baseline.
3. It saves the state it is about to change to a ledger file, injects the fault through the Kubernetes API, and reads the change back. K000 injects nothing.
4. It polls every 15 seconds for up to 240 seconds. Each poll builds an observation from the last 180 seconds of telemetry and cluster state and sends it to Jev. The first poll that reports an incident is the scored poll. If no poll reports an incident, the last poll is scored. The harness stores every poll.
5. In a `finally` block it restores the saved state, reads it back, and waits for the cluster to return to baseline.
6. If a run (node_taint) left one worker empty, the batch spreads the application pods over both workers again before the next run.

The harness only records Jev's answer. It never acts on it. A run took 256 to 468 seconds (median 406).

## The scenarios

We wrote the expected answers by hand from measurements on the cluster, before these runs. The reasons are in `tier3/scenarios/K*.yaml`, and the measurements are in [k8s-testbed-notes.md](k8s-testbed-notes.md).

| ID | Fault type | What the harness does | Incident? | Expected component | Expected diagnosis | Accepted actions |
|---|---|---|---|---|---|---|
| K000 | none | nothing (healthy control) | no | none | none | no_action |
| K001 | scale_zero | cart scaled to zero replicas | yes | cart | service_unavailable | scale_up, rollback_change, investigate, escalate |
| K002 | kill_pod | cart pod deleted once | no | cart | pod_restart | no_action, investigate |
| K003 | memory_limit | cart memory limit set to 10Mi | yes | cart | oom_killed | rollback_change, investigate, escalate |
| K004 | network_partition | all ingress to cart blocked by a NetworkPolicy | no | none | none | no_action, investigate |
| K005 | readiness_probe_fail | cart readiness probe always fails | yes | cart | readiness_failure | rollback_change, investigate, escalate |
| K006 | config_corruption | cart given an invalid `DATABASE_HOST`, a variable it does not read | no | none | none | no_action, investigate |
| K007 | dependency_removal | valkey-cart scaled to zero replicas | yes | valkey-cart | service_unavailable | scale_up, rollback_change, investigate, escalate |
| K008 | replica_overload | cart scaled to zero while the load triples | yes | cart | service_unavailable | scale_up, rollback_change, investigate, escalate |
| K009 | node_taint | application worker jev-worker2 tainted NoExecute | yes | jev-worker2 | node_eviction | investigate, escalate |
| K010 | pvc_full | 500 MB written to `/tmp` in a cart pod. The write is OOMKilled and the pod restarts | no | cart | oom_killed | no_action, investigate |

Four fault scenarios expect no incident. On this cluster their effect is too small or too short for anyone to act on. K002 and K010 heal on their own (cart serves again within about 30 seconds, with a short frontend error burst afterwards). K004 leaves existing connections working, and K006 is a rolling restart and nothing else. We still score their component and diagnosis against what happened, so a quiet "none / none" answer fails localization and diagnosis on K002 and K010. That is a judgment call in the expected answers, not in Jev.

## What Jev sees

Each observation is a JSON document of 6,207 to 12,699 bytes (3,834 to 6,471 input tokens). It has:

- The time window and baseline window
- Deployment rows (desired, ready, available, rollout state, baseline values)
- Pod rows (node, phase, ready, terminating, new since baseline, restarts, waiting and last termination reasons)
- Node rows (ready, pressure, schedulable, taint effects, pod count)
- Warning events
- Service rates, error rates and latencies against the baseline
- Calls between services
- Error logs
- Up to 5 failing traces

The observation shows effects only. It contains no pod specs, environment variables, resource limits, NetworkPolicy objects or taint keys. A node taint shows up as its effect (`NoExecute`), never its key. The leak test in `tests/tier3/test_runner.py` runs every scenario and every poll through the whole path. In task 10.2 we also read the requests of one run of each scenario by hand and searched every stored request for the ten fault names, `jev-fault`, `harness`, `scenario`, `ground_truth` and `inject`. We found none.

Jev answers four questions with fixed answer sets: incident (probability, 0.5 or more is yes), component (`none`, 15 services or 4 nodes), diagnosis (12 values) and action (6 values: `no_action`, `investigate`, `scale_up`, `restart_service`, `rollback_change`, `escalate`). Jev returned no reasoning text in any of the 351 polls, so the reasoning sheet (`python -m tier3 export-reasoning`) has nothing to hand-score.

## How we score

Detection, localization (the group's RCA), diagnosis and action are each PASS, FAIL or UNKNOWN, scored on the scored poll. Localization and diagnosis pass only on an exact match. Action passes if the action is in the accepted list. No run got UNKNOWN. We do not combine the four verdicts.

The scorecard follows the group's terms. `detect`, `RCA`, `diagnosis` and `remediation correct` count the 18 runs of the six incident scenarios. `FP runs` counts runs of the five no-incident scenarios (K000, K002, K004, K006, K010) in which any poll reported an incident. `FP polls` counts those polls. MTTD is the time from injection to the first poll that reports an incident.

## Results

From `python -m tier3 scorecard` over both run directories:

| Scenario | Runs | Detect | RCA | Diagnosis | Remediation correct | MTTD median (min-max) s | FP runs | FP polls |
|---|---|---|---|---|---|---|---|---|
| K000 healthy control | 10 | | | | | | 0/10 | 0/150 |
| K001 scale_zero | 3 | 3/3 | 1/3 | 1/3 | 3/3 | 16.5 (16.5-16.9) | | |
| K002 kill_pod | 3 | | | | | | 3/3 | 36/45 |
| K003 memory_limit | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 16.6 (15.9-16.9) | | |
| K004 network_partition | 3 | | | | | | 0/3 | 0/45 |
| K005 readiness_probe_fail | 3 | 3/3 | 3/3 | 3/3 | 3/3 | 16.8 (15.9-16.8) | | |
| K006 config_corruption | 3 | | | | | | 0/3 | 0/45 |
| K007 dependency_removal | 3 | 3/3 | 1/3 | 3/3 | 3/3 | 30.6 (30.6-30.6) | | |
| K008 replica_overload | 3 | 3/3 | 0/3 | 0/3 | 3/3 | 16.2 (15.8-16.8) | | |
| K009 node_taint | 3 | 3/3 | 0/3 | 0/3 | 3/3 | 16.4 (16.4-16.9) | | |
| K010 pvc_full | 3 | | | | | | 1/3 | 11/45 |
| **All** | 40 | 18/18 | 8/18 | 10/18 | 18/18 | 16.7 (15.8-30.6) | 4/22 | 47/330 |

Detection precision (detected fault runs over detected fault runs plus false-positive runs) is 18/22. Remediation executed, MTTR and recovery verified are not measured (recommendation only).

The wrong answers on the scored poll were these:

| Scenario | Runs | Jev answered | Expected |
|---|---|---|---|
| K001 | 2 of 3 | frontend, application_failure | cart, service_unavailable |
| K002 | 3 of 3 | incident; frontend, application_failure | no incident; cart, pod_restart |
| K007 | 2 of 3 | frontend-proxy (diagnosis right) | valkey-cart |
| K008 | 3 of 3 | frontend, application_failure | cart, service_unavailable |
| K009 | 3 of 3 | frontend, frontend-proxy or recommendation; application_failure | jev-worker2, node_eviction |
| K010 | 1 of 3 | incident; frontend, oom_killed | no incident; cart, oom_killed |
| K010 | 1 of 3 | no incident; none, none | no incident; cart, oom_killed |

Jev took 361 to 796 ms to answer (median 433 ms) and used 386 to 394 output tokens per answer. Jev reports no cost. With the user-set price of USD 0.042 per million input tokens and no charge for output, the 351 calls (1,508,831 input tokens) cost an estimated USD 0.063.

## What this means

Jev detected every incident. In 15 of the 18 runs it did so on the first poll after the harness saw the fault, 1 to 2 seconds later. In the three K007 runs it reported the incident one poll (about 15 seconds) later. On the healthy cluster it never raised an alarm: 0 of 150 control polls. This is the opposite of Tier 2, where Jev reported an incident in every healthy run. The observations differ, though. Tier 3 leaves the collector, Jaeger and opamp-server out of the observation (`INFRA_SERVICES` in `tier3/collect.py`). Its control observations contained no error log lines. The Tier 2 observations each carried 59 to 132 lines of collector dropped-data errors, the likely cause there. So the result supports the Tier 2 explanation, but it is not a like-for-like retest of Tier 2.

Jev was also quiet on two of the four faults that the expected answers call "no incident" (K004 and K006). On the two that heal themselves (K002, and K010 once), it reported an incident for as long as the short burst of cart errors stayed in the 180-second observation window, then stopped. In all three K002 runs, polls 1 to 12 reported an incident and polls 13 to 15 did not. Jev is reacting to real errors there. Whether a 15-second outage that already healed is an incident is the judgment in the expected answer.

Jev named the right component and diagnosis when the failing pod itself showed the cause: K003 (OOMKilled, back-off events) and K005 (readiness probe failures) were right in every run. It named the wrong component when the failing service had no pods left to show anything (K001, K008) and when the cause was a node (K009). In those cases it named the caller that logged the errors, mostly frontend, and called it an application failure. Nodes are in the answer set, and the node rows showed `NoExecute` on jev-worker2, but Jev never chose a node.

MTTD is about 16 seconds because the first poll is at 15 seconds and Jev answers in under a second. It measures the harness's poll interval more than Jev. K007 took 30 seconds because Jev first reported the incident at the second poll.

Three runs for each fault are too few to give rates. Read the results as a direction.

## Example result

This is run 1 of K008 (`results/tier3/20261018-070851-60ee/K008.json`), scored on poll 1, 16 seconds after injection. The cart Deployment showed `desired 0, available 0` against a baseline of 1, and frontend logged "Checkout failed to place order". The cart service row in the same observation still showed exactly its baseline request rate (1.74 per second) and no errors. The errors were on frontend's spans. We have not yet checked why the cart row did not drop. Jev answered:

| Question | Answer | Probabilities | Verdict |
|---|---|---|---|
| Incident | yes | 0.92 | PASS |
| Component | frontend | frontend 0.64, cart 0.29, frontend-proxy 0.05 | FAIL |
| Diagnosis | application_failure | application_failure 0.53, service_unavailable 0.45 | FAIL |
| Action | scale_up | scale_up 0.73, rollback_change 0.10, investigate 0.07 | PASS |

Jev recommended `scale_up`, the right action for a Deployment at zero replicas, but it named the wrong component.

## What is and is not comparable with the group's results

| | LF Edge harness | Tier 3 |
|---|---|---|
| Faults | ten Kubernetes API faults | the same ten, same names, exact restore, no self-naming markers |
| Agent | tool-calling loop that gathers evidence and can act | Jev answers four fixed questions from an observation the harness builds |
| Polling | every 15 s, first poll after 15 s | the same |
| Detect, RCA, remediation correct, MTTD | measured | measured (RCA = localization, remediation correct = action) |
| Remediation executed, MTTR, recovery verified | measured | not measured: the harness never acts |
| Healthy control | skip-injection mode, false alarms not scored | K000, every poll scored for false alarms |

Compare only detect, RCA, remediation correct and MTTD, and say that Jev did not gather its own evidence. Four of our scenarios expect no incident because of how the faults behave on this demo version and cluster (see the table above). It is possible that the group's agents were scored on those faults as incidents.

## Known limits

- Only Jev ran on Tier 3. There is no baseline engine to compare against yet.
- The runs used one laptop, one cluster, one demo version and one load level, with three runs per fault.
- The four "no incident" fault scenarios and the scoring of their component and diagnosis are our judgment. K002 and K010 in particular depend on what counts as an incident.
- When cart had no pods (K008), its service row still showed its baseline request rate and no errors. The errors were on the callers, so the service rows point away from the failing service. The Deployment row did show zero replicas. We have not yet checked why the cart row did not drop. A collector fix would be a change to the experiment.
- The cluster was created before Jaeger had its own host port, so these runs set `TIER3_JAEGER_URL` to the control plane's IP (see the testbed notes).
- The run files are not in the repository, because `results/` is git-ignored.

## How to repeat

Run `./setup-k8s.sh`, then `./health-check-k8s.sh`. Then run `./run-all-k8s-experiments K001 K002 K003 K004 K005 K006 K007 K008 K009 K010 --repeats 3` and `./run-all-k8s-experiments K000 --repeats 10`, and `python -m tier3 scorecard`, `export-lfedge` and `export-reasoning` on the two run directories. Jev's answers can change between runs, and the live cluster is never the same twice. The figures on this page come from the `evaluation`, `polls`, `timing` and `usage` fields of the 40 result files listed above.
