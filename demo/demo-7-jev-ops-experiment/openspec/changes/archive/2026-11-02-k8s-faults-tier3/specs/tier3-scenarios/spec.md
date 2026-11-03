## ADDED Requirements

### Requirement: One YAML file per experiment
Tier 3 scenarios SHALL live in `tier3/scenarios/<experiment_id>.yaml` with fields `experiment_id`, `application`, `description`, `rationale`, `fault {type, target}`, `target_component`, `duration_s`, `detection_window_s`, `poll_interval_s`, `workload`, `observation_window_s`, `baseline_window_s` and `expected {incident, affected_component, diagnosis, acceptable_actions}`. The loader SHALL validate every enum against the Tier 3 vocabularies and fail on unknown fields.

#### Scenario: Diagnosis outside the vocabulary
- **WHEN** a scenario's expected diagnosis is `resource_exhaustion`
- **THEN** loading fails with an error naming the value and the allowed set

### Requirement: Hand-written ground truth with rationale
Every expected outcome SHALL be written by hand with a `rationale` based on the symptoms measured in task group 1. It SHALL NOT be derived from any engine or LLM, and changing it SHALL be treated as a change to the experiment.

#### Scenario: Missing rationale
- **WHEN** a scenario has no `rationale`
- **THEN** loading fails

### Requirement: Tier 3 vocabularies
`tier3/vocab.py` SHALL define components (`none`, the demo services including `valkey-cart`, and the four kind node names), diagnoses (`none`, `service_unavailable`, `pod_restart`, `oom_killed`, `network_blocked`, `readiness_failure`, `misconfiguration`, `node_eviction`, `disk_full`, `overload`, `application_failure`, `unknown`) and actions (`no_action`, `investigate`, `restart_service`, `scale_up`, `rollback_change`, `escalate`), separate from Tier 1 and Tier 2.

#### Scenario: Same options every run
- **WHEN** two different scenarios run
- **THEN** the engine receives identical component, diagnosis and action lists

### Requirement: Catalogue aligned with the LF Edge harness
The catalogue SHALL contain the healthy control `K000` and `K001` to `K010`, one per fault type of the group's harness, targeting `cart` (or `valkey-cart` for `dependency_removal` and an application worker for `node_taint`), as in design D7. A scenario whose symptom does not appear in task group 1 SHALL be dropped or given `incident: false` with a rationale, and the decision recorded in `docs/k8s-testbed-notes.md`.

#### Scenario: Healthy control
- **WHEN** `K000` is loaded
- **THEN** it has no fault, expects no incident, component `none`, diagnosis `none`, and accepts only `no_action`

#### Scenario: Fault names match the group
- **WHEN** the catalogue is loaded
- **THEN** every fault type is one of the group's ten names, spelled as in their harness
