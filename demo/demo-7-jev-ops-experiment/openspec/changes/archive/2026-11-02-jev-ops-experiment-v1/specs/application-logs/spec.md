## ADDED Requirements

### Requirement: Application logs in every observation
Each generated observation SHALL include application log lines for the last 15 minutes, each with a timestamp, level (INFO, WARN, ERROR or FATAL), service and message. The lines SHALL match the scenario and SHALL be determined by the seed.

#### Scenario: Logs present and reproducible
- **WHEN** an observation is generated twice with the same scenario and seed
- **THEN** both contain the same non-empty, time-ordered log lines

#### Scenario: Logs match severity
- **WHEN** `healthy` and `critical` observations are generated
- **THEN** healthy logs contain no ERROR or FATAL lines and critical logs contain at least one

#### Scenario: Logs point at the failing service
- **WHEN** an observation has a degraded or down service
- **THEN** crash lines in its logs are attributed to that service

### Requirement: Logs do not leak the answer
Log lines MUST NOT contain scenario names or wording that tells Jev which decision to make.

#### Scenario: No leakage
- **WHEN** logs are generated for every scenario
- **THEN** no message contains a scenario name, "expected" or an instruction to restart or escalate

### Requirement: Log-driven scenarios
The catalogue SHALL include `hung_worker`, where the evidence for a restart is mainly in the logs, and `log_only_errors`, where metrics look healthy but the logs show user-facing errors.

#### Scenario: Hung worker
- **WHEN** a `hung_worker` observation is generated
- **THEN** host metrics are low, only the worker is degraded, and the worker logs show ERROR lines and no successful jobs

#### Scenario: Log-only errors
- **WHEN** a `log_only_errors` observation is generated
- **THEN** all services are healthy and metrics are normal, but the logs contain repeated ERROR lines

### Requirement: Metrics-only comparison
The runner SHALL support `--no-logs`, which sends Jev the same metrics without the logs, records that mode in every result and the summary, and marks the results file name. Adding logs MUST NOT change the metrics generated for a seed.

#### Scenario: Ablation keeps metrics identical
- **WHEN** the same scenario and seed are run with and without `--no-logs`
- **THEN** the metrics sent to Jev are identical and only the logs differ

#### Scenario: Mode is visible
- **WHEN** a `--no-logs` batch finishes
- **THEN** the results file ends in `-nologs.jsonl` and the report states that logs were not sent
