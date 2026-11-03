## Context

Tier 2 (change `otel-demo-tier2`) runs the OpenTelemetry Demo 3.1.0 on Docker Compose and injects faults with flagd flags. The live runs worked, but the LF Edge AIOps group did not accept them. Their harness (`lfedgeai/AIOps`, `research/agentic_aiops_architectures`, read at commit `a408708`, 2026-10-12) differs from ours in four ways that matter:

1. The demo runs on Kubernetes (OpenShift). Faults are Kubernetes API changes (`inject_k8s_fault` in `code/harness/run_harness.py`); flagd is optional and called legacy.
2. The harness polls the agent after injection until it reports a fault, so MTTD is measured.
3. Agents are tool-calling loops that query telemetry and the Kubernetes API themselves, and execute fixes. Remediation executed, MTTR and recovery verified are headline metrics.
4. Matrices are large (288 to 336 runs). Their latest uses `scale_zero`, `config_corruption` and `kill_pod` on `cart`. Ground truth is the fault's target component (`config/fault_ground_truth.yaml`), and RCA is scored by finding the component name in the agent's text.

Their harness can run without injection (`--skip-fault-injection`), but it does not score false alarms, so it reports no false-positive rate or detection precision, which their own RFP scorecard (`test_harness/README.md`: MTTD precision vs recall, RCA accuracy, action fidelity, cost-to-resolve) asks for.

Their fault code has details we deliberately do not copy: `config_corruption` also sets `HARNESS_INJECTED_FAULT=config_corruption` on the container, which names the fault; the NetworkPolicy and taint are named `harness-*`; recovery sets replicas to 1 rather than the previous count; `node_taint` taints the first listed node, which can be the control plane.

Jev is a classifier with `choice`, `score` and `noul` questions (`docs/jev-api-notes.md`). It cannot call tools or act. It fits the group's "Reasoning" slot only.

Host facts (2026-10-13): 30 GB RAM, 16 cores, Docker 29.8.2. `kind`, `kubectl` and `helm` are not installed.

Read from their harness on 2026-10-14 (`docs/k8s-testbed-notes.md`): poll every 15 s after a 15 s wait, fault held 90 s by default, MTTD from injection to the first alert, per-run fields in `HarnessRun`. Their `memory_limit`, `readiness_probe_fail` and `config_corruption` change every container; `pvc_full` ignores a failed write (`|| true`); `node_taint` replaces the node's whole taint list.

Facts about kind, kindnet, the demo chart and the fault behaviour below are not yet verified against pinned versions. Task group 1 verifies each before code depends on it.

## Goals / Non-Goals

**Goals:**
- Run Jev against the group's Kubernetes fault catalogue, with their fault names and target, so results are comparable with their tables.
- Measure MTTD by polling, and false positives and detection precision with a healthy control.
- Report in the group's scorecard terms and export per-run records in their field names.
- Keep hidden ground truth, the scrubber and leak test, four independent verdicts and reset in `finally`.
- No change to Tier 1 or Tier 2 behaviour or tests.

**Non-Goals:**
- Acting on Jev's recommendation. Remediation executed, MTTR and recovery verified are reported as "not measured".
- Tool-calling or multi-agent topologies (the group's scenarios 0, b, c). Jev has no tool interface.
- Real node faults on virtual machines. Planned as a follow-up change on the same harness.
- OpenShift, ClickHouse or MLflow. We read the demo's own Prometheus, Jaeger and OpenSearch, and export JSON that can be loaded into their tracking later.
- A composite score.

## Decisions

**D1. kind with four nodes.** A kind cluster named `jev`: `jev-control-plane`, `jev-worker` (label `jev-role=observability`), `jev-worker2` and `jev-worker3` (label `jev-role=app`). kind, its node image and every chart are pinned. Alternatives: a single-node cluster (then `node_taint` evicts everything, including the telemetry stack, and measures the harness instead of the app), k3s on VMs (needed only for real node faults, which are out of scope here), the group's OpenShift (not available to us locally). kind is in the group's own bootstrapping list (`README.md`: "Kind | MicroShift | Fedora").

**D2. Observability and load are kept off the fault targets.** Chart values put Prometheus, Jaeger, OpenSearch, Grafana, the OpenTelemetry Collector, flagd and the load generator on `jev-worker`, and the application services on `jev-worker2` and `jev-worker3`. `node_taint` targets an application worker only. Otherwise a fault could break the measurement or the load.

**D3. Demo from the official Helm chart, values only.** The chart release whose `appVersion` is 3.1.0 is pinned. Values set node selectors and NodePort services with fixed ports. kind `extraPortMappings` on the control-plane expose those NodePorts on the host. The chart is never forked.

**D4. NetworkPolicy must be enforced.** `network_partition` only works if the CNI enforces NetworkPolicy. If the pinned kind's default CNI does, it is used; otherwise the cluster is created with `disableDefaultCNI` and a pinned Calico. Task 1 proves a deny-all ingress policy blocks traffic before the scenario is kept.

**D5. The ten fault types, with exact restore.** The injector supports the group's types and names. Before inject it saves the exact prior state; reset restores that state, not a fixed value.

| Fault | Inject | Saved and restored |
|---|---|---|
| `scale_zero` | target replicas to 0 | previous replica count |
| `kill_pod` | delete the target's pods | the deleted pod names; recovery waits for new pods |
| `memory_limit` | set every container's memory limit to `10Mi` (and a memory request, if set, to the same, or the API server rejects it) | the deployment's pod template |
| `network_partition` | NetworkPolicy `jev-fault-np` denying all ingress to the target pods | delete the policy |
| `readiness_probe_fail` | readiness probe `exec: false`, period 5 s, threshold 1 | the pod template |
| `config_corruption` | add `DATABASE_HOST=invalid-host-that-does-not-exist.local` | the pod template |
| `dependency_removal` | a backing service's replicas to 0 | previous replica count |
| `replica_overload` | target replicas to 0 and load generator replicas to 3 | both previous counts |
| `node_taint` | taint an application worker `jev-fault=true:NoExecute` | remove that taint only |
| `pvc_full` | write 500 MB to `/tmp/jev-fault-fill` in one target pod (via `kubectl exec`; a failed write fails the inject) | remove the file |

Unlike the group's code, no fault adds a label or variable that names itself. Harness objects use the `jev-fault` prefix, which the scrubber removes. The Kubernetes client is thin HTTP with `requests`, not the `kubernetes` package. Pod exec needs a streaming protocol, so `pvc_full` uses `kubectl exec` with the same kubeconfig.

Host NodePorts are our own Services (`tier3/testbed/nodeports.yaml`) selecting the chart's pods, so the chart's service values are not needed. A pod is listed in the observation when it is new since the baseline as well as when it is unhealthy or restarted, so a replaced pod (kill_pod, node_taint) is visible as an effect.

**D6. Revert without the harness.** Before inject, the harness writes the saved state to `results/tier3/.active-fault.json`, and arms a host timer (`systemd-run --user --on-active=<duration + 120 s> python -m tier3 reset-faults`). If the harness dies, the timer restores the cluster from the ledger. `python -m tier3 reset-faults` is idempotent: it restores from the ledger if present, deletes every `jev-fault-*` NetworkPolicy, removes every `jev-fault` taint, and reports what it changed. The health check refuses to start while a ledger or harness object exists.

**D7. Scenarios.** YAML, one per experiment, in the Tier 2 shape: `experiment_id`, `application`, `description`, `rationale`, `fault {type, target}`, `target_component`, `duration_s`, `detection_window_s`, `poll_interval_s`, `workload`, `observation_window_s`, `baseline_window_s`, `expected {incident, affected_component, diagnosis, acceptable_actions}`. Target is `cart` as in the group's latest matrix, so results line up. Expected outcomes are hand-written and provisional until task 1 has observed each fault.

| ID | Fault | Target | Expected component | Expected diagnosis | Acceptable actions |
|---|---|---|---|---|---|
| K000 | none (healthy control) | none | none | none | no_action |
| K001 | scale_zero | cart | cart | service_unavailable | scale_up, rollback_change, investigate, escalate |
| K002 | kill_pod | cart | cart | pod_restart | no_action, investigate |
| K003 | memory_limit | cart | cart | oom_killed | rollback_change, investigate, escalate |
| K004 | network_partition | cart | cart | network_blocked | rollback_change, investigate, escalate |
| K005 | readiness_probe_fail | cart | cart | readiness_failure | rollback_change, investigate, escalate |
| K006 | config_corruption | cart | cart | misconfiguration | rollback_change, investigate, escalate |
| K007 | dependency_removal | valkey-cart | valkey-cart | service_unavailable | scale_up, rollback_change, investigate, escalate |
| K008 | replica_overload | cart | cart | service_unavailable | scale_up, rollback_change, investigate, escalate |
| K009 | node_taint | jev-worker2 | jev-worker2 | node_eviction | investigate, escalate |
| K010 | pvc_full | cart | cart | disk_full | investigate, escalate |

Expected behaviour that task 1 must check, because it decides whether a scenario has a symptom at all:
- `memory_limit`, `readiness_probe_fail` and `config_corruption` change the pod template, which starts a rolling update. With one replica and the default strategy, the old pod keeps serving until the new one is Ready, so users may see nothing; the only symptom may be a stuck rollout and a crash-looping or unready new pod. The ground truth for "incident" in those cases is a judgement call, written in the rationale.
- The cart service reads its store address from `VALKEY_ADDR`, so `DATABASE_HOST` may be ignored entirely, making `config_corruption` on `cart` a no-op apart from the pod restart. If so, the scenario is dropped or retargeted at a service that reads `DATABASE_HOST`, and the group is told, since their matrix uses it.
- `kill_pod` self-heals in seconds; with a 30 s poll interval the symptom may be gone before the first poll.
- `pvc_full` writes to the container's writable layer, which on kind sits on the host disk; it may show nothing.
A scenario with no observable symptom is dropped, or kept with `incident: false` if the rationale argues the system is genuinely healthy, and the decision is recorded in `docs/k8s-testbed-notes.md`.

**D8. Tier 3 vocabularies.**
- Components: `none`, the demo's application and data services (including `valkey-cart`), and the four kind node names. The same list in every run.
- Diagnosis: `none`, `service_unavailable`, `pod_restart`, `oom_killed`, `network_blocked`, `readiness_failure`, `misconfiguration`, `node_eviction`, `disk_full`, `overload`, `application_failure`, `unknown`.
- Action: `no_action`, `investigate`, `restart_service`, `scale_up`, `rollback_change`, `escalate`.
`unknown` and `no_action` are always available.

**D9. Observation shape.** Windows and byte budget as in Tier 2, plus Kubernetes state read from the API:
- `deployments`: per demo deployment, desired, ready, available and updated replicas, the same at the baseline, and whether a rollout is in progress or stalled.
- `pods`: only pods not Running and Ready, or with restarts or OOM kills in the window: service, node, phase, restart delta, waiting or last termination reason.
- `nodes`: per node, Ready, pressure conditions, schedulable, the effects of any taints (keys removed), pod count.
- `events`: Warning events in the window, deduplicated with counts.
And the Tier 2 sections: `services`, `dependencies`, `error_logs`, `failing_traces`. No pod specs, environment variables, resource limits or NetworkPolicy objects are sent: they would show the change itself rather than its effect. Node CPU and memory metrics are not collected on kind, because every kind node shares the host kernel and would report host-wide figures.

**D10. Scrubber and leak test.** The builder takes only collector output and scrub terms, as in Tier 2. Additions: drop the `kube-system`, `local-path-storage` and any CNI namespace from pods and events; replace `jev-fault` and taint keys in all text; scrub the experiment ID. Evidence is kept: event reasons (`OOMKilling`, `BackOff`, `Unhealthy`, `FailedScheduling`, `TaintManagerEviction`), replica counts, node names. The leak test runs every scenario through the whole path with fixtures that contain every scrub term, including events captured from the real cluster in task 1, and asserts nothing leaks and the question set is identical across scenarios.

**D11. Polling detection.** In the default `poll` mode the runner waits `telemetry_lag_s` (default 15 s, as in the group's harness) after inject, then every `poll_interval_s` (15 s, the group's default `--poll-interval`; read 2026-10-14) builds a fresh observation (window ending now) and asks Jev. The first poll with `incident_detected = true` sets `t_detect`, and that poll's decision is the one scored. If no poll detects before `detection_window_s` ends, the last poll's decision is scored and `t_detect` is `null`. For `K000`, polling runs for the whole window; every poll that reports an incident is a false-positive poll, and the run is a false-positive run if any poll does. `single` mode keeps the Tier 2 behaviour (one call after the duration) for comparison. The number of Jev calls, and tokens and cost where reported, are recorded per run.

**D12. Scorecard and export.** `python -m tier3 scorecard <results dir>` prints, per scenario and overall: detect (fault runs detected / fault runs), RCA (localization PASS), diagnosis PASS, remediation correct (action PASS), MTTD median and range over detected runs, false-positive runs and polls on `K000`, detection precision (detected fault runs / (detected fault runs + false-positive control runs)), agent errors, Jev calls, tokens and cost. Remediation executed, MTTR and recovery verified are printed as "not measured (recommendation only)". `python -m tier3 export-lfedge <results dir>` writes JSONL using the group's per-run field names (`detected`, `rca_correct`, `remediation_correct`, `mttd_seconds`, fault flag and variant, agent, `agent_error`, null for the unmeasured fields). Field names are checked against their harness code in task 1. Our four verdicts remain the primary result; the export is a mapping, and a count over small N is never shown as a bare percentage.

**D13. Engine side, as in Tier 2.** `tier3/engine.py` defines `Decision` (with `affected_component` and `reasoning`) and `AIOpsEngine`. `tier3/jev_adapter.py` holds the four questions and is the only Tier 3 module importing `jev.client`. The `reasoning` text is stored when present and can be exported for hand scoring, outside the verdicts.

**D14. Runner and recovery.** The Tier 2 lifecycle, with polling in place of the single wait. Recovery after reset requires every demo deployment back at its saved replica count and fully available, no harness object or ledger left, every node Ready and untainted by us, and service stats back to baseline. Otherwise the run is `contaminated` and the batch stops.

**D15. Results.** `results/tier3/<run-id>/<experiment_id>.json` holds the Tier 2 fields plus every poll (time, request size, decision), `t_detect`, the saved and restored state, pod placement, versions and git commit. `results/*` stays git-ignored.

**D16. Offline CI.** Fake Kubernetes API, fake collectors, fake engine, virtual clock. `./run-all.sh --check` runs Tier 3 tests. No test starts kind or Docker, or calls Jev.

**D17. Policy.** `CLAUDE.md`: "Kubernetes (a local kind cluster) is allowed only for the Tier 3 testbed." The "never act on the recommendation" rule is unchanged.

## Risks / Trade-offs

- [Rolling updates hide `memory_limit`, `readiness_probe_fail` and `config_corruption` from users] → observe in task 1; the observation includes deployments, pods and events, so a stuck rollout is still visible; ground truth decided per scenario with a rationale.
- [`config_corruption` is a no-op on `cart`] → verify; drop or retarget, and tell the group.
- [Polling multiplies Jev calls and cost] → record calls per run; poll interval and window are scenario parameters.
- [Polling scores the first-detect decision, which may be made on thin evidence] → also record every poll's decision, so a later analysis can score the final poll too.
- [kind's CNI does not enforce NetworkPolicy] → verify; switch to pinned Calico.
- [kind nodes share host memory, so memory pressure from the demo affects the host] → pre-check, Tier 2 must be stopped, measured usage recorded.
- [Comparisons with the group are not like-for-like: their agents gather their own evidence and act, Jev gets a built observation and does not act] → state this in every report; compare only detect, RCA, remediation correct and MTTD.
- [Leak through Kubernetes text we have not seen] → fixtures from real captured events.
- [Small N] → `--repeats`, extra `K000` runs, counts not percentages.

## Migration Plan

1. Install `kind`, `kubectl` and `helm`; verify every external fact and every fault in task group 1; record results in `docs/k8s-testbed-notes.md`.
2. Add `tier3/` and `tests/tier3/`. No Tier 1 or Tier 2 behaviour changes.
3. Update docs and `CLAUDE.md`.
4. Roll back by deleting `tier3/`, the wrappers and docs changes, and `kind delete cluster --name jev`.

## Open Questions

- Should the harness ever carry out Jev's chosen remediation (fixed mapping, Kubernetes faults only) so MTTR and recovery can be reported? Not in this change; it needs a policy decision.
- Does the group want our results in their MLflow? Ask at their weekly sync; the JSONL export keeps that option open.
- Which scenarios survive task 1 (D7), and what ground truth do the rolling-update faults get?
- Poll interval: settled 2026-10-14. The group polls every 15 s after a 15 s wait, so Tier 3 does the same; the detection window is 240 s, not their 1500 s, because Jev answers in about a second.
