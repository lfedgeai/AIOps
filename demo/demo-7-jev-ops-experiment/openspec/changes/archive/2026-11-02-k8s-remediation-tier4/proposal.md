## Why

Tier 3 reports MTTD, detection, RCA and "remediation correct" in the LF Edge AIOps group's terms, but `remediation executed`, `MTTR` and `recovery verified` are "not measured (recommendation only)", because no harness in this repo acts on Jev's answer. Those three columns are where the group's agents differ most, and the first Tier 3 batch (2026-10-18) suggests Jev's recommended action would often hit the wrong target: it named `frontend` in K001, K008 and K009 while recommending `scale_up` or `investigate`. Only a closed loop on the same faults shows whether Jev's recommendation, carried out as given, brings the system back, and how fast.

## What Changes

- Add **Tier 4**: closed-loop remediation on the existing Tier 3 kind cluster, with the same ten fault types, scenarios `K000` to `K010`, observation, polling and Jev adapter. After the first poll that reports an incident, the harness carries out Jev's recommended action on Jev's named component, then measures whether and when the system recovers.
- Add a **remediation executor** with a fixed allowlist mapped to reversible Kubernetes operations on Deployments in the demo namespace only:
  - `scale_up`: set the named Deployment's replicas to its baseline count if it is below it, otherwise one more than now.
  - `restart_service`: rollout restart of the named Deployment.
  - `rollback_change`: undo the named Deployment to its previous revision from the cluster's own rollout history (ReplicaSet revisions). It never reads the harness's fault ledger or saved state.
  - `investigate`, `escalate`, `no_action`: no action. The run is "not auto-remediated" and MTTR is `null`.
- **Guardrails:** at most one action per run; never nodes, Services, NetworkPolicies, StatefulSets or anything outside the demo namespace; an action on a component that is not a demo Deployment is "not executable" and nothing is done; the action's target state is saved to the ledger before acting and read back after; the full reset in `finally` restores both the fault and the action's changes; the dead-man timer and `reset-faults` cover both from the ledger alone.
- Add **recovery measurement**: after the action (or, with no action, after the detecting poll), the harness watches Deployments and service telemetry until they are back at baseline for a hold period or a recovery window ends. It reports `remediation_executed`, `recovery_verified`, `mttr_seconds` and `mttr_verified_seconds`, and separately "recovered without remediation" for faults that heal on their own.
- Extend the scorecard and the LF Edge export with these fields, measured instead of "not measured". Tier 3's own scorecard stays as it is.
- **BREAKING (policy only):** `CLAUDE.md` says "Never add code that acts on Jev's recommendation" and Tier 3 says "Still recommendation-only". This change narrows the rule: only Tier 4 acts, only on the local kind testbed, only through the allowlist. Tiers 1, 2 and 3 stay recommendation-only, and a test enforces that no other package imports the executor.

## Capabilities

### New Capabilities
- `tier4-remediation-executor`: the action allowlist, mapping to Kubernetes operations, target checks, save before act, read-back, exact restore, ledger, dead-man and `reset-faults` coverage.
- `tier4-recovery`: recovery checks, hold period, recovery window, MTTR and verified MTTR, recovered without remediation.
- `tier4-runner`: closed-loop lifecycle reusing Tier 3 (inject, poll, act once, watch recovery, reset), records, scorecard, LF Edge export with measured remediation fields, batch runs, contamination stop.
- `tier4-reproducibility`: entry scripts, README section, `CLAUDE.md` rule change, notes in `docs/k8s-testbed-notes.md`, offline CI.

### Modified Capabilities
<!-- None. openspec/specs/ is empty; earlier changes are not archived. Tier 3 behaviour does not change: Tier 4 reuses its modules by import. -->

## Impact

- **Code:** new `tier4/` package and `tests/tier4/`. Tier 4 imports Tier 3's scenarios, faults, collectors, observation, engine and Jev adapter without changing their behaviour. The executor lives only in `tier4/`.
- **Cluster:** the same kind cluster and demo install as Tier 3. Rollout history must be kept (`revisionHistoryLimit` at least 2 on the demo Deployments; the Kubernetes default is 10). No new components.
- **Safety:** the harness now changes cluster state on an engine's say-so. It stays bounded to reversible Deployment operations in one namespace on a local cluster, and every change is restored.
- **Jev usage:** polling stops at the first detecting poll, as in Tier 3, so calls per run do not grow.
- **Docs and policy:** `README.md`, `CLAUDE.md`, `.env.example` if new settings appear, `docs/k8s-testbed-notes.md`.
- **CI:** stays offline. Fake Kubernetes, fake engine, virtual clock; no cluster, Docker or Jev in tests.
