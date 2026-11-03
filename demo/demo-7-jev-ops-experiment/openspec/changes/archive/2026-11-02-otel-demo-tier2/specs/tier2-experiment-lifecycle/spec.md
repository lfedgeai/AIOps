## ADDED Requirements

### Requirement: Lifecycle order
The runner SHALL perform, in order: verify the testbed is healthy; establish baseline telemetry; verify the workload; inject the fault; wait for manifestation; collect telemetry; build the observation; ask the engine; store the raw response; normalize; evaluate; reset the fault; verify return to baseline; persist the result.

#### Scenario: Order of steps
- **WHEN** an experiment runs with fake components that record calls
- **THEN** the calls occur in the order above

#### Scenario: Unhealthy testbed
- **WHEN** the health check fails at the start
- **THEN** no fault is injected and the run reports the failing check

### Requirement: Reset always runs
The fault reset SHALL run in a `finally` block, so it runs when the engine raises, a collector fails, or the run is interrupted.

#### Scenario: Engine raises
- **WHEN** the engine raises an exception after injection
- **THEN** the fault is reset before the error is reported

#### Scenario: Collector fails
- **WHEN** a collector fails after injection
- **THEN** the fault is reset and the run is recorded as an error

### Requirement: Contamination guard
If the system does not return to baseline after reset, the run SHALL be marked contaminated and the batch SHALL stop.

#### Scenario: Batch stops
- **WHEN** an experiment ends contaminated
- **THEN** no further experiment in the batch starts and the summary says why

### Requirement: Healthy control runs the same path
The control SHALL use the same lifecycle with the injection step skipped, so any difference in results comes from the fault.

#### Scenario: Control
- **WHEN** `F000` is run
- **THEN** observation, engine call and evaluation run as for a fault and only injection and manifestation are skipped

### Requirement: Result file
Each experiment SHALL write `results/tier2/<run-id>/<experiment_id>.json` containing the experiment ID, fault, ground truth, normalized engine output with the raw response, the four verdicts, the timing block, trim notes, demo and tool versions, git commit and run status (`ok`, `error` or `contaminated`).

#### Scenario: Contents
- **WHEN** a result file is read
- **THEN** it contains every field above and parses as JSON

#### Scenario: Ground truth stays out of the request
- **WHEN** a result is written
- **THEN** ground truth appears in the result file and does not appear in the stored request payload

### Requirement: Raw response never dropped
The raw engine response SHALL be stored in the result even when parsing fails.

#### Scenario: Invalid reply
- **WHEN** the reply cannot be parsed
- **THEN** the result has status `invalid` and the raw text

### Requirement: Batch runs
`run-all-experiments` SHALL run the whole catalogue, with `--repeats N` for repeated runs, and write a batch summary with per-dimension aggregates and the false-positive rate.

#### Scenario: Repeats
- **WHEN** `--repeats 3` is given
- **THEN** every scenario runs three times and each run has its own result file

#### Scenario: Summary
- **WHEN** a batch finishes
- **THEN** a summary file with per-dimension counts is written beside the results

### Requirement: Single experiment command
`run-experiment <ID>` SHALL run one experiment and print the four verdicts, the timing and the path to the result file.

#### Scenario: Unknown ID
- **WHEN** an ID with no scenario file is given
- **THEN** the command exits non-zero and lists the valid IDs

### Requirement: Secrets stay out
No result file or log SHALL contain `JEV_API_KEY`.

#### Scenario: Key not stored
- **WHEN** a result file is searched for the key value in a test with a known fake key
- **THEN** it is not found
