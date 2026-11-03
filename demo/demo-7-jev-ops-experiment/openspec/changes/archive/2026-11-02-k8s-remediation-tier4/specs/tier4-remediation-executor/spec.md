## ADDED Requirements

### Requirement: Fixed action allowlist
The executor SHALL map the engine's `recommended_action` and `affected_component` to exactly one operation: `scale_up` sets the named Deployment's replicas to its baseline count when current replicas are below it, otherwise to current plus one, never above 3; `restart_service` performs a rollout restart of the named Deployment; `rollback_change` undoes the named Deployment to its previous revision with `kubectl rollout undo`; `investigate`, `escalate` and `no_action` perform nothing. No other operation SHALL be possible.

#### Scenario: Scale up a Deployment at zero
- **WHEN** the decision is `scale_up` on `cart`, cart has 0 replicas and its baseline is 1
- **THEN** cart's replicas are set to 1 and the record shows `remediation_executed: true`

#### Scenario: Scale up a healthy Deployment
- **WHEN** the decision is `scale_up` on `frontend`, which has 1 replica and a baseline of 1
- **THEN** frontend's replicas are set to 2

#### Scenario: Scale up at the cap
- **WHEN** the decision is `scale_up` on a Deployment that already has 3 replicas
- **THEN** nothing is changed and the record shows `not_executable` with the reason "replica cap"

#### Scenario: Investigate
- **WHEN** the decision is `investigate`
- **THEN** nothing is changed, `remediation_executed` is false and the outcome is "not auto-remediated"

### Requirement: Target restrictions
The executor SHALL act only on a Deployment in the demo namespace whose name is not infrastructure (Tier 3 `is_infra`). A node, `none`, an infrastructure name or a name that is not a Deployment SHALL be `not_executable` with a reason, and nothing SHALL be changed.

#### Scenario: Node named
- **WHEN** the decision is `restart_service` on `jev-worker2`
- **THEN** nothing is changed and the record shows `not_executable` with the reason "not a Deployment"

#### Scenario: Load generator named
- **WHEN** the decision is `scale_up` on `load-generator`
- **THEN** nothing is changed and the reason is "infrastructure"

### Requirement: Rollback from the cluster's history only
`rollback_change` SHALL use only the cluster's own rollout history. It SHALL NOT read the fault ledger, the harness's saved state or the scenario. When the Deployment has no previous revision, the action SHALL be `not_executable`. The record SHALL store the revision before and after the undo.

#### Scenario: Template fault on the right Deployment
- **WHEN** `memory_limit` set cart's memory limit to 10Mi (a new revision) and the decision is `rollback_change` on `cart`
- **THEN** cart's pod template returns to the previous revision's template and the record stores both revision numbers

#### Scenario: Replica fault
- **WHEN** `scale_zero` set cart to 0 replicas (no new revision) and the decision is `rollback_change` on `cart`
- **THEN** the undo runs against the previous revision, cart's replicas stay 0, and the outcome is decided by the recovery check

#### Scenario: No history
- **WHEN** the named Deployment has only one revision
- **THEN** nothing is changed and the reason is "no previous revision"

### Requirement: Save before act, read back, restore exactly
Before any operation the executor SHALL write the target Deployment's replicas and pod template to an `action` entry in the ledger, then act, then read the change back. Reset SHALL restore the action's saved state first and the fault's saved state second, reading each back. A read-back mismatch SHALL be an error, and reset SHALL still run.

#### Scenario: Action and fault on the same Deployment
- **WHEN** the fault changed cart's template and the action rolled cart back
- **THEN** reset restores cart to the action's saved state, then to the fault's saved state, and cart ends equal to its pre-fault spec

#### Scenario: Crash after acting
- **WHEN** the harness dies after the action and before reset
- **THEN** `python -m tier4 reset-faults` or the dead-man timer restores the action and then the fault from the ledger alone

### Requirement: One action per run
The executor SHALL be called at most once per run, on the scored poll's decision. Later polls, if any, SHALL NOT act.

#### Scenario: Second call refused
- **WHEN** the runner calls the executor a second time in the same run
- **THEN** it raises and nothing is changed
