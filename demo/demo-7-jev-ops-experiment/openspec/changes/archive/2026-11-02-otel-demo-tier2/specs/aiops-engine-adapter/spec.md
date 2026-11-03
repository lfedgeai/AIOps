## ADDED Requirements

### Requirement: Engine interface
The system SHALL define `AIOpsEngine` with `decide(Observation) -> Decision`. The runner, evaluator and collectors MUST depend only on this interface and not on Jev classes.

#### Scenario: Other engine
- **WHEN** a fake engine implementing `decide` is passed to the runner
- **THEN** the experiment runs without any Jev code or key

#### Scenario: No Jev import outside the adapter
- **WHEN** the Tier 2 modules other than the adapter are inspected
- **THEN** none imports `jev.client`

### Requirement: Jev adapter
`JevAdapter` SHALL implement `AIOpsEngine` by calling `JevClient` with Tier 2 questions: `incident_detected`, `affected_service` (a choice over the full fixed list of demo services), `diagnosis` and `recommended_action`. The questions SHALL be the same in every run.

#### Scenario: Request shape
- **WHEN** the adapter builds a request for an observation
- **THEN** it contains the observation as state and the four Tier 2 questions

### Requirement: Backward-compatible client change
`JevClient.build_request` and `decide` SHALL accept an optional `questions` argument. With no argument they SHALL produce exactly the Tier 1 request and behaviour.

#### Scenario: Tier 1 unchanged
- **WHEN** the existing Tier 1 tests are run after the change
- **THEN** they pass without modification

#### Scenario: Custom questions
- **WHEN** `build_request` is called with a different question list
- **THEN** the request carries that list in place of the Tier 1 questions

### Requirement: Normalized decision
The adapter SHALL return a `Decision` with `incident_detected`, `affected_service`, `diagnosis`, `recommended_action`, `confidence`, `reasoning`, `status` and `raw_response`. `confidence` and `reasoning` SHALL be `null` when Jev does not provide them. The raw response MUST always be kept.

#### Scenario: Complete answer
- **WHEN** Jev returns a valid answer to all questions
- **THEN** the decision has status `ok` and the raw response is stored unchanged

#### Scenario: No confidence available
- **WHEN** Jev returns no probability for an answer
- **THEN** `confidence` is `null`

### Requirement: Strict parsing
An answer outside the allowed set SHALL give status `invalid` and MUST NOT be coerced. A failed call SHALL give status `error`. The raw response is kept in both cases when one exists.

#### Scenario: Unknown service name
- **WHEN** Jev names a service that is not in the allowed list
- **THEN** the decision status is `invalid`

#### Scenario: Transport failure
- **WHEN** the call fails after retries
- **THEN** the status is `error` and the error text is recorded

### Requirement: Tier 2 vocabulary
Diagnosis SHALL be one of `application_failure`, `latency_degradation`, `dependency_unreachable`, `database_contention`, `resource_exhaustion`, `unknown`, `none`. Action SHALL be one of `investigate`, `restart_service`, `rollback_change`, `scale_up`, `escalate`, `no_action`. These lists MUST be separate from the Tier 1 lists.

#### Scenario: Always an exit
- **WHEN** the question list is built
- **THEN** `unknown`, `none` and `no_action` are available as answers

### Requirement: Recommendation is only recorded
Nothing in Tier 2 SHALL act on the recommended action. The only state-changing operations are the harness's own fault injection and reset.

#### Scenario: Restart recommended
- **WHEN** Jev recommends `restart_service`
- **THEN** the decision is stored and no service is restarted
