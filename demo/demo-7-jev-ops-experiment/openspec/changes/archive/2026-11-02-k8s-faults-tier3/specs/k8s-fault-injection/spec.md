## ADDED Requirements

### Requirement: The group's ten fault types
The injector SHALL support exactly `scale_zero`, `kill_pod`, `memory_limit`, `network_partition`, `readiness_probe_fail`, `config_corruption`, `dependency_removal`, `replica_overload`, `node_taint` and `pvc_full`, with the effects listed in design D5. Faults SHALL be Kubernetes API changes only. The loader SHALL reject unknown fault types, targets that are not demo deployments, and for `node_taint` any node other than `jev-worker2` or `jev-worker3`.

#### Scenario: Unknown fault type
- **WHEN** a scenario names `cpu_hog`
- **THEN** loading fails with an error naming the type

#### Scenario: Taint on the observability node
- **WHEN** a `node_taint` scenario targets `jev-worker`
- **THEN** loading fails

### Requirement: No self-describing fault markers
An injected fault SHALL NOT add any label, annotation, environment variable or object name that names the fault type or experiment. Harness-created objects and taints SHALL use the `jev-fault` prefix, which the scrubber removes.

#### Scenario: config_corruption
- **WHEN** `config_corruption` is injected
- **THEN** the only change to the pod template is the added `DATABASE_HOST` variable

### Requirement: Save, inject, read back
Before inject, the injector SHALL save the exact state it will change (replica counts, pod template, taints) to the fault ledger `results/tier3/.active-fault.json`. After inject it SHALL read the state back and confirm the change. A mismatch SHALL raise, stop the run, and still trigger reset.

#### Scenario: Read-back mismatch
- **WHEN** the replica count read back after `scale_zero` is not 0
- **THEN** inject raises, the run is recorded as an error, and reset runs

### Requirement: Exact reset
`reset` SHALL restore the saved state exactly (previous replica counts, previous pod template), delete harness NetworkPolicies, remove only the `jev-fault` taint, and delete the fill file, then read back and remove the ledger. A failed reset SHALL mark the run `contaminated`.

#### Scenario: Deployment had two replicas
- **WHEN** `scale_zero` is injected on a deployment with 2 replicas and then reset
- **THEN** the deployment has 2 replicas again

#### Scenario: Other taints are kept
- **WHEN** a node has another taint besides `jev-fault` at reset
- **THEN** only the `jev-fault` taint is removed

### Requirement: Revert without the harness
On inject, the harness SHALL arm a host timer that runs `python -m tier3 reset-faults` at the scenario duration plus 120 seconds. `reset-faults` SHALL restore from the ledger when present, delete every `jev-fault-*` NetworkPolicy, remove every `jev-fault` taint, be idempotent, and report what it changed.

#### Scenario: Harness killed during a run
- **WHEN** the harness process is killed after inject
- **THEN** the cluster is restored from the ledger no later than the duration plus 120 seconds

#### Scenario: Clean cluster
- **WHEN** `reset-faults` runs with no ledger and no harness objects
- **THEN** it succeeds and reports no changes

### Requirement: Manifestation is observed, not assumed
The harness SHALL poll for manifestation with a timeout: a service-row deviation as in Tier 2, a deployment below its saved available replicas, a new unhealthy pod, or a node condition or taint change. The result SHALL go only to the harness and result file. When not seen, `t_manifest` SHALL be `null`.

#### Scenario: scale_zero
- **WHEN** cart's available replicas drop to 0
- **THEN** `t_manifest` is set to the first poll that saw it
