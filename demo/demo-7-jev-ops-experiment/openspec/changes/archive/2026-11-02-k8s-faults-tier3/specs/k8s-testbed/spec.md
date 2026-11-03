## ADDED Requirements

### Requirement: Pre-check before the cluster starts
The testbed scripts SHALL refuse to create or start the cluster unless Docker is running and at least 12 GB of memory is available, and SHALL refuse while Tier 2 demo containers are running.

#### Scenario: Not enough memory
- **WHEN** available memory is below 12 GB
- **THEN** the script exits non-zero and prints the available and required amounts

#### Scenario: Tier 2 still running
- **WHEN** containers of the Tier 2 Compose stack are running
- **THEN** the script exits non-zero and says to stop Tier 2 first

### Requirement: Four-node kind cluster
`tier3/testbed/create.sh` SHALL create a kind cluster named `jev` from `tier3/testbed/kind.yaml`, with a pinned kind version and node image: `jev-control-plane`, `jev-worker` labelled `jev-role=observability`, and `jev-worker2` and `jev-worker3` labelled `jev-role=app`. Fixed NodePorts SHALL be mapped to the host through `extraPortMappings`.

#### Scenario: Fresh create
- **WHEN** `create.sh` runs with no `jev` cluster present
- **THEN** four nodes are Ready and each worker carries its role label

#### Scenario: Cluster already exists
- **WHEN** `create.sh` runs and the `jev` cluster exists
- **THEN** it does not recreate it and exits with a message naming `delete.sh`

### Requirement: NetworkPolicy is enforced
The cluster SHALL enforce NetworkPolicy. If the default CNI of the pinned kind does not, the cluster SHALL be created without it and a pinned Calico SHALL be installed.

#### Scenario: Deny-all policy blocks traffic
- **WHEN** a deny-all ingress NetworkPolicy selects a pod
- **THEN** connections to that pod from other pods fail

### Requirement: Demo installed from the pinned chart by values only
The testbed SHALL install the OpenTelemetry Demo Helm chart release whose app version is 3.1.0, configured only by files in `tier3/testbed/values/`. Prometheus, Jaeger, OpenSearch, Grafana, the collector, flagd and the load generator SHALL run on `jev-worker`; application services SHALL run on `jev-worker2` and `jev-worker3`.

#### Scenario: Placement after install
- **WHEN** the install finishes
- **THEN** no application pod runs on `jev-worker` or `jev-control-plane`, and no observability pod or load generator runs on an application worker

#### Scenario: No chart is patched
- **WHEN** the install scripts are inspected
- **THEN** they reference upstream charts by name and pinned version and pass only values files and `--set` flags

### Requirement: Versions are recorded
The testbed SHALL write `tier3/testbed/VERSION` with the kind, node image, Kubernetes, chart, demo app and CNI versions read from the running cluster.

#### Scenario: Version file after install
- **WHEN** the install finishes
- **THEN** `VERSION` lists each component with the version reported by the cluster

### Requirement: Health check
`tier3/testbed/health-check.sh` SHALL check: all nodes Ready, every demo deployment at its desired replicas and available, Prometheus ready with span metrics present, Jaeger listing services, OpenSearch healthy, the load generator running, no `jev-fault` taint, NetworkPolicy or file, and no fault ledger. It SHALL exit non-zero and name the first failing check.

#### Scenario: Leftover fault
- **WHEN** a `jev-fault-*` NetworkPolicy exists
- **THEN** the script exits non-zero, names it, and suggests `python -m tier3 reset-faults`

### Requirement: Stop, start and delete
The testbed SHALL provide `stop.sh` and `start.sh` (stop and start the kind node containers) and `delete.sh` (delete the cluster after confirmation, or with `--yes`).

#### Scenario: Delete without confirmation
- **WHEN** `delete.sh` runs without `--yes` and the answer is not "yes"
- **THEN** nothing is deleted
