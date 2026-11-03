## ADDED Requirements

### Requirement: Declarative scenario files
Each experiment SHALL be one YAML file under `tier2/scenarios/` with `experiment_id`, `application`, `fault` (flag and variant), `target_service`, `fault_type`, `duration_s`, `workload`, `observation_window_s`, `baseline_window_s` and `expected` (`incident`, `affected_service`, `diagnosis`, `acceptable_actions`). Loading SHALL validate the file and fail with a clear error if it is invalid.

#### Scenario: Valid file loads
- **WHEN** a complete scenario file is loaded
- **THEN** it yields a frozen scenario object with the same field values

#### Scenario: Missing field
- **WHEN** a scenario file lacks `expected.diagnosis`
- **THEN** loading fails and names the missing field

#### Scenario: Value outside the vocabulary
- **WHEN** `expected.diagnosis` or an acceptable action is not in the Tier 2 vocabulary
- **THEN** loading fails and names the value

#### Scenario: Unknown service
- **WHEN** `target_service` or `expected.affected_service` is not a known demo service
- **THEN** loading fails and names the value

### Requirement: Hand-written ground truth
Expected outcomes SHALL be written by hand in the scenario file with a rationale. They MUST NOT be derived from Jev, from any LLM, or from telemetry collected in the run.

#### Scenario: Rationale present
- **WHEN** a scenario file is loaded
- **THEN** it has a non-empty rationale

### Requirement: Healthy control
The catalogue SHALL include `F000`, a control with no fault, where `expected.incident` is false and `expected.diagnosis` is `none`.

#### Scenario: Control has no fault
- **WHEN** `F000` is loaded
- **THEN** it has no fault flag and expects no incident

### Requirement: Fault catalogue
The catalogue SHALL contain these fault scenarios, each backed by a flagd flag that was verified on a live stack to produce a measurable effect in the pinned demo version: `F001` productCatalogFailure (direct failure of one product), `F003` paymentFailure at 100% (transaction failure), `F004` paymentUnreachable (misleading symptoms: errors appear in checkout while payment is healthy), `F005` imageSlowLoad at 10 seconds (latency without errors), `F006` productCatalogLockContention (database-related degradation). A scenario whose flag fails verification SHALL be removed or replaced, and the reason documented. `cartFailure` (former F002) and `intlShippingSlowdown` (former F005) were removed this way and their IDs are not reused.

#### Scenario: Catalogue size and coverage
- **WHEN** the catalogue is listed
- **THEN** it contains the control and the verified fault scenarios, covering direct failure, latency, downstream dependency failure, transaction failure, database degradation and misleading symptoms

#### Scenario: Unverified fault is not shipped
- **WHEN** a flag does not produce a measurable effect in the pinned version
- **THEN** no scenario for it is in the catalogue and `docs/otel-demo-notes.md` records why

### Requirement: Scenario listing
A command SHALL list the available experiment IDs and descriptions without starting the testbed.

#### Scenario: List
- **WHEN** the list command is run
- **THEN** every experiment ID in `tier2/scenarios/` is printed
