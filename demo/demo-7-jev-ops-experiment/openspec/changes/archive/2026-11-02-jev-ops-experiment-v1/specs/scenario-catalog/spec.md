## ADDED Requirements

### Requirement: Required scenarios exist
The catalogue SHALL define at least the scenarios `healthy`, `degraded`, `critical`, `ambiguous` and `contradictory`, addressable by name.

#### Scenario: All required names present
- **WHEN** the catalogue is loaded
- **THEN** each of the five required scenario names resolves to a scenario definition

#### Scenario: Unknown name rejected
- **WHEN** a scenario is requested by a name not in the catalogue
- **THEN** the lookup fails with an error that lists the valid names

### Requirement: Independent expected outcome per scenario
Each scenario SHALL carry an expected outcome consisting of severity (`normal`, `degraded`, `high`, `critical`), action (`observe`, `investigate`, `restart`, `escalate`) and human-review-required (`yes` or `no`). The expected outcome MUST be defined statically in code and MUST NOT be produced by Jev or any other LLM.

#### Scenario: Expected outcome is static
- **WHEN** the same scenario is loaded twice, with different seeds
- **THEN** its expected outcome is identical both times

#### Scenario: Values are bounded
- **WHEN** the catalogue is validated
- **THEN** every expected severity, action and human-review value belongs to its allowed set

### Requirement: Rationale documented for expected outcomes
Each scenario SHALL record a short rationale for its expected outcome, and the `ambiguous` and `contradictory` scenarios SHALL state explicitly why that outcome was chosen.

#### Scenario: Rationale present
- **WHEN** the catalogue is validated
- **THEN** every scenario has a non-empty rationale

### Requirement: Scenario metric profiles
Each scenario SHALL define the metric profile the generator uses (value ranges and correlation behaviour), kept separate from its expected outcome.

#### Scenario: Profile covers all metrics
- **WHEN** the catalogue is validated
- **THEN** every scenario profile defines all required observation fields
