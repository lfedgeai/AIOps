## ADDED Requirements

### Requirement: Bounded probable-cause diagnosis
The adapter SHALL ask Jev for one probable cause from `none`, `resource_exhaustion`, `dependency_failure`, `application_bug`, `configuration`, `network`, `unknown`, in the same request as the operational decision. An out-of-set or missing cause SHALL make the decision invalid.

#### Scenario: Cause returned
- **WHEN** Jev replies with an allowed cause
- **THEN** the decision records it with its confidence and probabilities

#### Scenario: Invalid cause
- **WHEN** Jev replies with a cause outside the allowed set
- **THEN** the decision status is invalid and no value is coerced

### Requirement: Hand-written expected cause
Every scenario SHALL define an expected probable cause by hand, with a short reason next to the definition, and it MUST NOT come from any LLM.

#### Scenario: All scenarios have a cause
- **WHEN** the catalogue is validated
- **THEN** every scenario's expected cause belongs to the allowed set

### Requirement: Cause scored separately from the decision
Probable cause SHALL be reported as its own accuracy figure. It MUST NOT change whether a run is PASS or FAIL, and it MUST NOT be included in the overall confidence figure.

#### Scenario: Wrong cause, right decision
- **WHEN** severity, action and human review are correct but the cause is wrong
- **THEN** the run is PASS and the summary counts the cause as incorrect

#### Scenario: Summary shows cause accuracy
- **WHEN** a batch finishes
- **THEN** the summary lists probable_cause accuracy, marked as not part of PASS/FAIL
