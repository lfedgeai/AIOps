## ADDED Requirements

### Requirement: Decision comparison
The evaluator SHALL compare Jev's severity, action and human-review decision against the expected outcome, report a per-field result, and report an overall match only when all three fields are correct.

#### Scenario: All fields match
- **WHEN** all three decision fields equal the expected values
- **THEN** the overall result is correct and every field result is correct

#### Scenario: Partial match
- **WHEN** severity and action match but human-review differs
- **THEN** the overall result is incorrect, severity and action are marked correct, and human-review is marked incorrect

### Requirement: Aggregate summary computed from results
The evaluator SHALL compute, from the actual run records only, the total runs, correct and incorrect counts, accuracy by decision type (severity, action, human-review), accuracy by scenario, average confidence, average confidence for correct decisions, average confidence for incorrect decisions, and latency statistics. No figure SHALL be hard-coded or estimated.

#### Scenario: Counts and accuracy
- **WHEN** aggregating a known set of records with 3 correct and 1 incorrect
- **THEN** total is 4, correct is 3, incorrect is 1 and overall accuracy is 0.75

#### Scenario: Accuracy by scenario
- **WHEN** records span several scenarios
- **THEN** the summary reports correct/total for each scenario present

#### Scenario: Confidence split
- **WHEN** records include confidence values
- **THEN** the summary reports separate mean confidence for correct and incorrect decisions

#### Scenario: No confidence available
- **WHEN** no record has a confidence value
- **THEN** confidence metrics are reported as not available rather than zero

#### Scenario: Empty input
- **WHEN** aggregating zero records
- **THEN** the evaluator returns zero counts without raising a division error

### Requirement: Invalid and errored runs reported separately
Runs whose decision was invalid or whose Jev call failed SHALL be counted and reported in their own categories. Invalid decisions SHALL count as incorrect. Errored runs SHALL be excluded from accuracy denominators and shown explicitly.

#### Scenario: Categories visible
- **WHEN** a batch contains one invalid decision and one errored call
- **THEN** the summary shows one invalid and one errored, the invalid counts as incorrect, and the errored run is excluded from accuracy

### Requirement: Human-readable summary
The program SHALL print the summary in text at the end of a batch, with overall and by-scenario sections.

#### Scenario: Summary layout
- **WHEN** a batch completes
- **THEN** the output shows `Overall` (runs, correct, incorrect) and `By scenario` (correct/total per scenario) sections

### Requirement: Machine-readable result persistence
Each run SHALL be written as one JSON line containing scenario, seed, observations, expected outcome, decision, raw Jev response, confidence, latency, per-field and overall match flags and status; a summary JSON SHALL be written alongside. Files SHALL go under `results/` with a timestamped name.

#### Scenario: JSONL written
- **WHEN** a batch of N runs completes
- **THEN** a JSONL file with N valid JSON lines exists under `results/`

#### Scenario: Summary file written
- **WHEN** a batch completes
- **THEN** a summary JSON exists whose figures equal those printed

#### Scenario: Results are not committed
- **WHEN** results files are produced
- **THEN** git ignores them, except `results/.gitkeep`
