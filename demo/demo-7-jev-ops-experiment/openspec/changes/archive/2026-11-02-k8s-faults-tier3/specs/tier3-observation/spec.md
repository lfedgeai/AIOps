## ADDED Requirements

### Requirement: Builder sees collector output only
`tier3/observation.py` SHALL build the observation from collector output and scrub terms only. It MUST NOT import or accept `Scenario`, `FaultSpec` or `GroundTruth`.

#### Scenario: Import guard
- **WHEN** a test inspects the module's imports and the builder's signature
- **THEN** no scenario or ground-truth type appears

### Requirement: Fixed observation shape
Every observation SHALL have `window`, `baseline_window`, `deployments`, `pods`, `nodes`, `events`, `services`, `dependencies`, `error_logs` and `failing_traces`, as described in design D9. It SHALL NOT contain pod specs, environment variables, resource limits or NetworkPolicy objects. A value that cannot be measured SHALL be `null`.

#### Scenario: scale_zero is visible as an effect
- **WHEN** cart has been scaled to 0
- **THEN** the cart deployment row shows 0 desired and 0 available against a baseline of 1, and no field shows a spec change

#### Scenario: Healthy pods omitted
- **WHEN** every pod is Running and Ready with no restarts
- **THEN** `pods` is an empty list

### Requirement: Caps and byte budget
Every list SHALL be capped by a named constant and ranked with the largest deviation first. The builder SHALL enforce a byte budget by trimming lowest-ranked items and SHALL record trims in the result only.

#### Scenario: Over budget
- **WHEN** the observation exceeds the byte budget
- **THEN** tail items are removed until it fits, and trim counts are in the result

### Requirement: Scrubber covers Kubernetes leak paths
The scrubber SHALL drop `kube-system`, `local-path-storage` and CNI namespaces from pods and events, replace `jev-fault` and taint keys in all text, and scrub the experiment ID. It SHALL keep event reasons, replica counts and node names.

#### Scenario: Scheduling event naming the taint
- **WHEN** the event `0/4 nodes are available: 1 node(s) had untolerated taint {jev-fault: true}` is collected
- **THEN** the observation keeps the event with the taint key removed

### Requirement: Whole-path leak test
A test SHALL run every Tier 3 scenario through collectors, builder and adapter with fixtures containing every scrub term, including events captured from the real cluster in task group 1, and SHALL assert that no scrub term, experiment ID, fault type name or expected value reaches any request, and that the question set is identical for every scenario and every poll.

#### Scenario: Contaminated fixture
- **WHEN** fixtures contain `jev-fault`, the experiment ID and a fault type name
- **THEN** none of these strings is in any request
