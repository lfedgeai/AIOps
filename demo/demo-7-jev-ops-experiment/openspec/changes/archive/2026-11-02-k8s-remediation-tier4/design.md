## Context

Tier 3 (change `k8s-faults-tier3`) injects the LF Edge AIOps group's ten Kubernetes faults into the OpenTelemetry Demo 3.1.0 on a local kind cluster, polls Jev every 15 s, and scores detection, localization (RCA), diagnosis and action. It never acts, so `remediation executed`, `MTTR` and `recovery verified` are "not measured (recommendation only)". The group's harness measures them (`HarnessRun` fields `remediation_executed`, `mttr_seconds`, `mttr_verified_seconds`, `recovery_verified`; `docs/k8s-testbed-notes.md`), because their agents call tools and execute fixes.

Jev cannot act. It answers four fixed questions, one of which is the action (`no_action`, `investigate`, `restart_service`, `scale_up`, `rollback_change`, `escalate`), and another the component (15 demo services or 4 nodes). A closed loop therefore needs the harness to carry out that action for Jev, on the named component, and to measure the result.

The first full Tier 3 batch (2026-10-18, `docs/results-2026-10-18-tier3-jev.md`) gives the expected shape: detection 18/18, action acceptable 18/18, but the component was wrong in 10 of 18 runs, often `frontend`. A closed loop will show how often a right action on a wrong component fails to recover the system.

Constraints carried over: no Kubernetes Python client (API through `requests`, `kubectl` only where the API has no single call); ground truth never reaches the engine; four independent verdicts, no composite; reset in `finally`; save before change, restore exactly; tests offline. Policy today: `CLAUDE.md` forbids any code that acts on Jev's recommendation. This change narrows that rule to Tier 4 only.

## Goals / Non-Goals

**Goals:**
- Carry out Jev's recommended action once per run, on Jev's named component, through a fixed allowlist of reversible Deployment operations.
- Measure `remediation_executed`, `recovery_verified`, `mttr_seconds` and `mttr_verified_seconds` on the Tier 3 faults and scenarios, and report them in the group's terms.
- Tell apart "recovered because of the action", "recovered on its own" and "not recovered".
- Record actions taken on false alarms, and whether they disturbed a healthy system.
- Keep every Tier 3 guarantee: hidden ground truth, scrubber and leak test, exact restore, dead-man timer, contamination stop.
- No change to Tier 1, 2 or 3 behaviour or tests.

**Non-Goals:**
- Letting Jev choose arbitrary commands, parameters, namespaces or resource kinds. The allowlist is the whole action space.
- Acting on nodes, Services, NetworkPolicies, StatefulSets, ConfigMaps, the load generator or the observability stack.
- More than one action per run, or asking Jev again after acting (a multi-step loop is a later change).
- Real clusters or VMs. Tier 4 runs only on the local kind testbed.
- A composite score.

## Decisions

**D1. Tier 4 reuses Tier 3 by import.** New package `tier4/` with `executor.py`, `recovery.py`, `runner.py`, `report.py`, `cli.py`. It imports Tier 3's scenarios (`K000` to `K010`), fault injector, collectors, observation builder, engine and Jev adapter unchanged. The same `poll` loop finds the first detecting poll; that poll is scored exactly as in Tier 3, so Tier 3 and Tier 4 verdicts are comparable. Alternative: copy Tier 3 into Tier 4 and edit. Rejected: two diverging copies of the leak-tested path.

**D2. The allowlist.** The executor maps Jev's action and component to one operation:

| Action | Operation | Executable only if |
|---|---|---|
| `scale_up` | replicas := baseline replicas if current is below baseline, else current + 1; never above 3 | component is a Deployment in the demo namespace, not infrastructure |
| `restart_service` | rollout restart (pod template annotation `kubectl.kubernetes.io/restartedAt`, as `kubectl rollout restart` writes it) | same |
| `rollback_change` | `kubectl rollout undo deployment/<name>` (previous revision from the cluster's ReplicaSet history) | same, and the Deployment has a previous revision |
| `investigate`, `escalate`, `no_action` | nothing | n/a |

A node, `none`, an infrastructure name (`is_infra` from Tier 3, which covers the load generator and observability) or a name that is not a Deployment in the namespace is "not executable": nothing is changed and the record says why. The baseline replica count comes from the harness's own pre-fault snapshot of the cluster (the same values Jev sees as `baseline_desired`), not from the fault ledger. The cap of 3 bounds memory on the kind nodes. Alternative: let `scale_up` always add one replica. Rejected: on `scale_zero` that gives 1, the same as baseline, but on a healthy Deployment it should still add one, so the rule above covers both.

**D3. Rollback from rollout history, through kubectl.** `apps/v1` has no rollback subresource; `kubectl rollout undo` finds the ReplicaSet with the previous `deployment.kubernetes.io/revision` and copies its pod template into the Deployment. Using `kubectl` (already pinned for pod exec) gives the exact behaviour an operator gets. It never reads the harness's ledger or saved state, so it can only undo what the cluster's history holds: it fixes template faults (`memory_limit`, `readiness_probe_fail`, `config_corruption`) on the right Deployment, and does nothing useful for replica faults, NetworkPolicies or taints, because those are not in a ReplicaSet template. Alternative: reimplement undo with API calls. Rejected: more code that could differ from what operators see. Tier 3's `rebalance.sh` and earlier faults also create revisions; "previous revision" is whatever the cluster holds, and the record stores which revision was restored.

**D4. Save before act, restore in reverse order.** Before acting, the executor writes the target Deployment's replicas and pod template to the ledger (`results/tier3/.active-fault.json`, a new `action` entry next to the fault entry), then acts, then reads the change back. Reset in `finally` restores the action's saved state first, then the fault's saved state (reverse order of application), reading each back. `python -m tier4 reset-faults` and the dead-man timer restore both from the ledger alone. Alternative: a separate ledger file. Rejected: two files can disagree after a crash; one ledger is one source of truth.

**D5. Recovery is service-level, held for a period.** Tier 3's `manifestation()` treats replaced or restarted pods as "visible", which is always true after a restart or rollback. Tier 4's `recovery.py` defines recovered as: every Deployment at or above baseline availability with rollout complete; every application pod Running and Ready; and no service deviation from baseline (`any_deviation` from Tier 2), measured from span metrics over a 120 s window. The demo's span metrics arrive every 60 s and `increase()` needs two samples, so 120 s is the shortest window that returns data and it reflects about the last minute (measured 2026-10-18: windows of 15 to 90 s returned no series, and the first probes ran with a 30 s window whose service part was therefore blind). Missing service telemetry fails the check. Replaced pods and restart counts are ignored. Checks run every 15 s for the full 360 s after the action completes, and recovery counts only when the final run of passing checks reaches the end of the window and lasts at least 60 s (five checks, spanning a fresh metrics sample). A run that met the hold and later failed is a relapse: on kind, a restart under `network_partition` passed for over a minute before checkout errors began about 2 minutes after the action, so stopping at the first hold would have called it recovered. Checks start (or after the detecting poll when nothing was executed). The fault stays injected for the whole window; only the action can undo it. Results go to the harness and the record only, never to the engine.

**D6. Timing, aligned with the group's fields.** Read from their `run_harness.py` and `fault_verification.py` at `a408708` (task 1.1): `mttr_seconds` is fault injection to the agent's first successful remediation write ("legacy" in their code), and `mttr_verified_seconds` is fault injection to the first moment a per-fault-type check on the injected target passes (replicas at least 1 and ready, the 10Mi limit or the bad variable gone, the probe no longer `false`), counted only if a write targeted that target, polled every 5 s for up to 120 s with no hold. Their fault stays injected until `fault_duration`, then the harness reverts it; verification happens before that. We use the same meanings: `t_action_done` is when our write is accepted (and, for restart and rollback, the rollout is complete or 120 s pass); `mttr_seconds` is `t_inject` to `t_action_done`; `t_recovered` is the first check of a passing hold; `mttr_verified_seconds` is `t_inject` to `t_recovered`; `t_recovery_verified` (the end of the hold) is recorded too. `time_to_remediate_seconds` is `t_detect` to `t_action_done`. Each is `null` when it cannot be measured. Our check differs from theirs in what it checks: it is service-level and fault-agnostic (D5) and must hold for 60 s, so it can verify `kill_pod`, `pvc_full` and `node_taint`, which theirs cannot, and it can fail when the spec is fixed but the service is still erroring. The notes record this.

**D7. Three recovery outcomes.** `recovered_by_action` (an action was executed and recovery followed), `recovered_without_action` (no action was executed, or it was not executable, and the system recovered on its own, as K002 and K010 do), and `not_recovered`. As in the group's harness, `recovery_verified` is true only for `recovered_by_action` (theirs requires a write on the target); a self-healed run has `recovery_verified: false`, `self_recovered: true` and its own `self_recovery_seconds`, so self-healing faults do not inflate Jev's MTTR. A recovery that starts before the action completes is `recovered_without_action` with the action still recorded.

**D8. False-alarm actions.** On the no-incident scenarios and the K000 control, a detecting poll with an executable action means the harness acts on a healthy (or self-healed) system. The record keeps `action_on_false_alarm`, and the recovery check after it says whether the action disturbed the system (`disturbed_healthy_system` when the check fails at least once after the action). The scorecard reports both counts.

**D9. Isolation of the executor.** Only `tier4/executor.py` performs remediation, and only `tier4/runner.py` and `tier4/cli.py` import it. A test fails if any module outside `tier4/` imports it, or if `tier1`, `tier2` or `tier3` code contains a Deployment write that is not the Tier 3 fault injector. The Tier 3 rule "only the adapter touches Jev" carries over: no Tier 4 module imports `jev.client`.

**D10. Scorecard.** Tier 3's columns plus `remediation executed`, `not executable`, `recovered by action`, `recovered without action`, `not recovered`, MTTR median (min-max) over recovered-by-action runs, verified MTTR, and false-alarm actions. The LF Edge export fills `remediation_executed`, `recovery_verified`, `mttr_seconds` and `mttr_verified_seconds` instead of `null`.

**D11. Policy text.** `CLAUDE.md` changes from "Never add code that acts on Jev's recommendation" to: only Tier 4 acts on an engine's recommendation, only on the local kind testbed, only through `tier4/executor.py`'s allowlist, one action per run, with save-before-act and exact restore. Tiers 1 to 3 stay recommendation-only.

**D12. What Tier 4's MTTR is, and the Opus baseline.** Tier 4 measures **single-shot recommendation MTTR**: one answer, carried out once by the harness, no feedback and no retry. It is not operational MTTR (an engineer or agent who investigates, tries, checks and retries), and every report labels it so. To compare engines fairly, a Claude Opus engine (`claude-opus-5-5`) answers the same four questions from the same observation through the same `AIOpsEngine` interface, and its answer goes through the same executor and recovery check (task group 7). An agentic mode, where a model calls tools, takes several steps and verifies its own fix, would measure something closer to operational MTTR and to the LF Edge group's agents; Jev cannot run in it, so it would be a separate change, reported separately as "agentic MTTR" and never set against Jev's single-shot figures.

## Risks / Trade-offs

- [An action on the wrong component disturbs a healthy service, e.g. a restart of `frontend` during a cart fault] → Bounded to one reversible operation, saved and restored in `finally`; the disturbance is recorded and is part of what Tier 4 measures.
- [`rollout undo` rolls back to a stale or unrelated revision (from `rebalance.sh` or an earlier run)] → The record stores the revision restored; reset restores the saved template exactly; the testbed notes record revision history behaviour seen on the cluster.
- [Recovery check passes on noise and declares an early recovery] → Hold period of 60 s, a fail on missing telemetry, and the same deviation thresholds as Tier 3's manifestation check; probes in task 1 measure each fault's recovery with the right action before any Jev run.
- [Self-healing faults (K002, K010) recover before the action and inflate MTTR] → D7 separates `recovered_without_action`; MTTR for Jev uses `recovered_by_action` only.
- [`scale_up` exhausts memory on kind] → Cap of 3 replicas; the Tier 3 memory pre-check still runs.
- [A crash between act and reset leaves the action applied] → Ledger written before acting; dead-man timer and `reset-faults` restore the action first, then the fault.
- [Policy creep: other tiers start acting] → D9's import test and the narrowed `CLAUDE.md` rule.
- [MTTR mostly measures rollout and poll timings, not Jev] → Reported alongside `time_to_remediate_seconds` and the poll interval, as Tier 3 does for MTTD.

## Migration Plan

No migration. Tier 4 is additive and uses the existing cluster. Rollback is deleting `tier4/` and restoring the `CLAUDE.md` rule.

## Open Questions

- Whether a later change should ask Jev again after the action (a second step for `investigate`), which would need a new question set.
