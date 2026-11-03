## ADDED Requirements

### Requirement: Command-line entry point
The project SHALL provide `python run.py` supporting `--scenario <name|random>`, `--seed <int>` and `--runs <n>`, with a single run as the default.

#### Scenario: Default run
- **WHEN** the user runs `python run.py` with no options
- **THEN** one run executes for a randomly chosen scenario and its seed is printed

#### Scenario: Named scenario
- **WHEN** the user runs `python run.py --scenario degraded`
- **THEN** one run executes for the `degraded` scenario

#### Scenario: Seeded run
- **WHEN** the user runs `python run.py --scenario ambiguous --seed 1234`
- **THEN** the generated observations are identical to every other run with those arguments

#### Scenario: Invalid scenario
- **WHEN** the user passes an unknown scenario name
- **THEN** the program exits non-zero with a message listing valid names

### Requirement: Clear per-run output
For an individual run the program SHALL print the scenario, seed, generated server state, expected outcome, Jev decision, Jev confidence where available, and a PASS or FAIL result.

#### Scenario: Matching decision
- **WHEN** Jev's severity, action and human-review all equal the expected outcome
- **THEN** the output ends with `Result: PASS`

#### Scenario: Mismatching decision
- **WHEN** any of the three fields differs from the expected outcome
- **THEN** the output ends with `Result: FAIL` and identifies which fields differed

#### Scenario: Invalid or errored decision
- **WHEN** the adapter returns an invalid or error result
- **THEN** the output shows that status instead of a decision and does not report PASS

### Requirement: Batch execution
`--runs N` SHALL execute N runs, cycling through scenarios evenly when `--scenario` is not given, and SHALL print a summary at the end. Each run's seed SHALL be derived from a base seed and recorded so it can be replayed individually.

#### Scenario: Balanced scenarios
- **WHEN** the user runs `python run.py --runs 100`
- **THEN** each of the five scenarios is executed 20 times

#### Scenario: Replay one run from a batch
- **WHEN** the user takes a scenario and seed from a batch result record and runs them singly
- **THEN** the generated observations equal those in the batch record

#### Scenario: One failure does not stop the batch
- **WHEN** one run's Jev call fails permanently
- **THEN** the batch records the error and continues with the remaining runs

### Requirement: Read-only experiment
The runner SHALL only generate data, query Jev and record results; it MUST NOT execute any remediation action on any system.

#### Scenario: Recommended action is not executed
- **WHEN** Jev recommends `restart`
- **THEN** nothing is restarted and the action is only recorded and compared
