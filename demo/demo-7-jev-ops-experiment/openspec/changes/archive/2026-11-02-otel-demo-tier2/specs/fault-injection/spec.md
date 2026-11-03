## ADDED Requirements

### Requirement: Faults use the demo's flagd flags
Faults SHALL be injected only by setting a flagd flag variant. Scenario files SHALL name a flag and variant that exist in the pinned `demo.flagd.json`. The injector MUST NOT invent other fault mechanisms.

#### Scenario: Unknown flag is rejected
- **WHEN** a scenario names a flag that is not in the pinned flag file
- **THEN** loading the scenario fails with an error naming the flag

#### Scenario: Unknown variant is rejected
- **WHEN** a scenario names a variant the flag does not define
- **THEN** loading the scenario fails with an error naming the variant

### Requirement: Inject and reset
`FaultInjector` SHALL provide `inject(flag, variant)` and `reset()`. `inject` SHALL record the previous value so `reset` restores exactly it, and SHALL confirm the new state by reading it back.

#### Scenario: Inject then reset
- **WHEN** a flag is injected and then reset
- **THEN** its variant after reset equals its variant before inject

#### Scenario: Read-back mismatch
- **WHEN** the flag state read back after inject differs from the requested one
- **THEN** inject raises an error and the experiment does not continue

### Requirement: Reset is idempotent and global
The injector SHALL offer a `reset-faults` operation that returns every flag to `off`, and calling it on a clean system SHALL succeed and change nothing.

#### Scenario: Clean system
- **WHEN** `reset-faults` is run with all flags already off
- **THEN** it succeeds and reports no changes

#### Scenario: Recovery after a crash
- **WHEN** a previous run left a flag active
- **THEN** `reset-faults` turns it off

### Requirement: Manifestation is observed, not assumed
The harness SHALL detect manifestation by polling Prometheus for a deviation from the baseline window, with a timeout. The result SHALL be reported only to the harness and the results file, never to Jev.

#### Scenario: Fault manifests
- **WHEN** the error rate of the target service exceeds the baseline threshold before the timeout
- **THEN** the manifestation time is recorded

#### Scenario: Fault does not manifest
- **WHEN** no deviation is seen before the timeout
- **THEN** the manifestation time is `null` and the run continues

### Requirement: Workload control
The harness SHALL verify that the load generator is running before injecting a fault, and SHALL be able to set the user count from the scenario.

#### Scenario: No load
- **WHEN** the load generator is not running at the start of an experiment
- **THEN** the experiment stops before injecting and reports why

### Requirement: Return to baseline
After reset, the harness SHALL wait until the target service's error rate and latency are back within a configured band of the baseline window, or until a timeout.

#### Scenario: Recovered
- **WHEN** metrics return to the baseline band
- **THEN** the run is marked clean

#### Scenario: Not recovered
- **WHEN** metrics have not returned to the band by the timeout
- **THEN** the run is marked contaminated and no further experiment starts
