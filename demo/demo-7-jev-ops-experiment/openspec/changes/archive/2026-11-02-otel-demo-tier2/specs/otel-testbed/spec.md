## ADDED Requirements

### Requirement: Pinned OpenTelemetry Demo
The testbed SHALL run the official OpenTelemetry Demo at release tag 3.1.0, cloned by `setup.sh` into a git-ignored directory. The project MUST NOT fork, patch or vendor the demo.

#### Scenario: Fresh clone sets up the demo
- **WHEN** a developer runs `./setup.sh` on a machine with Docker
- **THEN** the demo is cloned at tag 3.1.0 and its images are pulled
- **AND** the tag is written to `tier2/testbed/VERSION`

#### Scenario: Demo files are unmodified at setup
- **WHEN** `git status` is run inside the demo checkout straight after setup
- **THEN** it reports no local changes

#### Scenario: Flag file after experiments
- **WHEN** experiments have run and every fault has been reset
- **THEN** the only difference in `src/flagd/demo.flagd.json` from the pinned file is formatting and the dropped `$schema` key, because flagd-ui rewrites the file on every toggle, and every flag value is identical to the pinned file

### Requirement: Actual versions are recorded
Setup SHALL record the demo tag and the image versions that actually run, because the demo's `.env` can disagree with the tag. Every result SHALL include them.

#### Scenario: Versions in results
- **WHEN** an experiment result is written
- **THEN** it contains the demo tag and the image versions recorded at setup

### Requirement: Start and stop through Compose
The testbed SHALL start and stop the demo with Docker Compose using the demo's own compose files. It MUST NOT require Kubernetes.

#### Scenario: Start
- **WHEN** the start command is run
- **THEN** the demo, its collector and its observability stack are started detached

#### Scenario: Stop
- **WHEN** the stop command is run
- **THEN** the stack is removed, including volumes, so the next start is clean

### Requirement: Resource pre-check
Setup SHALL check available memory and Docker before starting the demo, and SHALL report a clear message if the host cannot run the chosen profile. It SHALL offer the minimal profile.

#### Scenario: Not enough memory
- **WHEN** the host has less memory than the chosen profile needs
- **THEN** setup prints the shortfall and the minimal-profile option and does not start the stack

### Requirement: Health check
`health-check.sh` SHALL verify, in order: the frontend through Envoy, Prometheus readiness, the presence of span metrics for the demo services, the Jaeger service list, OpenSearch health, flagd flag state, and Locust running. It SHALL exit non-zero if any check fails and name the failing check.

#### Scenario: Healthy testbed
- **WHEN** all components are up and traffic is flowing
- **THEN** every check passes and the script exits 0

#### Scenario: Component down
- **WHEN** Prometheus is not reachable
- **THEN** the script reports the Prometheus check as failed and exits non-zero

#### Scenario: No traffic
- **WHEN** the demo is up but Locust is not generating load
- **THEN** the Locust and span-metrics checks fail with a message saying so

### Requirement: Faults are off at rest
Health check and setup SHALL verify that every flag is in its `off` variant, and SHALL fail if any fault is active.

#### Scenario: Leftover fault
- **WHEN** a flag is left in a non-off variant
- **THEN** the health check fails and names the flag
