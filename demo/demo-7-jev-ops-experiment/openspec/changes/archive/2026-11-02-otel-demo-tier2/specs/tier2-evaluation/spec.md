## ADDED Requirements

### Requirement: Four independent verdicts
The evaluator SHALL produce a separate verdict for detection, localization, diagnosis and action. Each verdict SHALL be PASS, FAIL or UNKNOWN. The evaluator MUST NOT produce a single combined score.

#### Scenario: Mixed result
- **WHEN** Jev finds the incident and the right service but gives the wrong diagnosis and an unacceptable action
- **THEN** detection and localization are PASS and diagnosis and action are FAIL

#### Scenario: No composite
- **WHEN** a result record is written
- **THEN** it has no overall score field

### Requirement: Detection
Detection SHALL be PASS when `incident_detected` equals `expected.incident`, otherwise FAIL.

#### Scenario: Incident found
- **WHEN** a fault scenario expects an incident and Jev reports one
- **THEN** detection is PASS

#### Scenario: Incident missed
- **WHEN** a fault scenario expects an incident and Jev reports none
- **THEN** detection is FAIL

### Requirement: Localization
Localization SHALL be PASS when `affected_service` equals `expected.affected_service`, otherwise FAIL. For the healthy control, localization is PASS when Jev names no affected service.

#### Scenario: Wrong service
- **WHEN** Jev names `checkout` and the expected service is `payment`
- **THEN** localization is FAIL

### Requirement: Diagnosis
Diagnosis SHALL be PASS when `diagnosis` equals `expected.diagnosis`, otherwise FAIL.

#### Scenario: Matching diagnosis
- **WHEN** Jev answers `dependency_unreachable` and that is expected
- **THEN** diagnosis is PASS

### Requirement: Action
Action SHALL be PASS when `recommended_action` is in `expected.acceptable_actions`, otherwise FAIL.

#### Scenario: Acceptable action
- **WHEN** the recommended action is in the acceptable list
- **THEN** action is PASS

#### Scenario: Unacceptable action
- **WHEN** the recommended action is not in the list
- **THEN** action is FAIL

### Requirement: Unknown verdicts
A dimension SHALL be UNKNOWN only when the decision status is `invalid` or `error`, or when that field is missing. UNKNOWN MUST NOT count as PASS or FAIL in any aggregate.

#### Scenario: Invalid decision
- **WHEN** the decision status is `invalid`
- **THEN** all four verdicts are UNKNOWN

#### Scenario: Dimensions are scored independently
- **WHEN** Jev reports no incident but names a service and a diagnosis
- **THEN** detection is FAIL and the other dimensions are still compared

### Requirement: Independence from the observation
The evaluator SHALL be a pure function of the stored decision and the ground truth. It MUST run after the decision is stored and MUST NOT receive the observation builder or the Jev client.

#### Scenario: Evaluation after storage
- **WHEN** a run completes
- **THEN** the raw Jev response is already stored before any verdict is computed

### Requirement: False positives
A healthy-control run in which Jev reports an incident SHALL be recorded as a false positive. The aggregate SHALL report the false-positive count and rate over control runs.

#### Scenario: False positive
- **WHEN** `F000` is run and Jev reports an incident
- **THEN** detection is FAIL and the run is counted as a false positive

#### Scenario: Rate
- **WHEN** a batch has 5 control runs and 1 false positive
- **THEN** the aggregate shows 1 of 5

### Requirement: Aggregates per dimension
The batch aggregate SHALL report PASS, FAIL and UNKNOWN counts for each dimension, per scenario and overall, plus decision latency and token statistics when available.

#### Scenario: Counts exclude unknown from rates
- **WHEN** a dimension has 4 PASS, 1 FAIL and 1 UNKNOWN
- **THEN** its pass rate is computed over 5 and the UNKNOWN count of 1 is shown beside it

### Requirement: Timing block
Each result SHALL contain `started_at`, `t_inject`, `t_manifest`, `t_observation`, `t_decision`, `detection_seconds`, `diagnosis_seconds` and `total_seconds`. A value that cannot be measured SHALL be `null`.

#### Scenario: Manifestation not measured
- **WHEN** `t_manifest` is `null`
- **THEN** `detection_seconds` is `null`

#### Scenario: Measured
- **WHEN** all timestamps are present
- **THEN** `total_seconds` equals the end time minus `started_at`
