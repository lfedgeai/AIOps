## ADDED Requirements

### Requirement: Tier 3 question set
`tier3/jev_adapter.py` SHALL send one request per run with four questions: `incident_detected` (`noul`, incident if the probability is at least 0.5), `affected_component` (`choice` over the Tier 3 component list), `diagnosis` (`choice` over the Tier 3 diagnoses) and `recommended_action` (`choice` over the Tier 3 actions). It SHALL be the only Tier 3 module that imports `jev.client`, and SHALL send through `JevClient.send`.

#### Scenario: Request contents
- **WHEN** a request is built
- **THEN** it holds the observation as state and exactly the four questions, with the same wording and options in every run

### Requirement: Strict parsing
An answer outside its option set, a missing answer, or a probability outside 0 to 1 SHALL make the decision `invalid`. A failed call SHALL be `error`. Neither SHALL be coerced or guessed, and the raw response SHALL always be kept.

#### Scenario: Out-of-set component
- **WHEN** Jev answers `affected_component` with `node-3`
- **THEN** the decision is `invalid` and the raw response is stored

### Requirement: Reasoning text recorded, not scored automatically
When the reply carries a top-level `reasoning` string, the adapter SHALL store it on the decision. It SHALL NOT affect any verdict.

#### Scenario: Reply with reasoning
- **WHEN** the reply includes `reasoning`
- **THEN** the result file contains the text and the four verdicts are computed from the choice answers only

### Requirement: Engine-neutral seam
The runner SHALL call engines only through `AIOpsEngine.decide(observation) -> Decision` defined in `tier3/engine.py`, so another engine can be added without changing the runner.

#### Scenario: Fake engine in tests
- **WHEN** tests pass a fake engine to the runner
- **THEN** the full lifecycle runs without importing `jev.client`
