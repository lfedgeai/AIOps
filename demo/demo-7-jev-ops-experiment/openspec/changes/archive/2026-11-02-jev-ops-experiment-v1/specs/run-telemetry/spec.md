## ADDED Requirements

### Requirement: Per-run telemetry
Each run SHALL record input and output tokens, Jev's reported elapsed time, the request round trip, the total call time including retries, the number of attempts, the final HTTP status, request and response sizes, the number of log lines sent, and Jev's raw usage object. Telemetry SHALL also be recorded for invalid and errored runs where available.

#### Scenario: Successful call
- **WHEN** Jev answers with a usage object and elapsedMs
- **THEN** the run record contains the token counts, elapsed time, round trip, attempts and status

#### Scenario: Retried call
- **WHEN** a call succeeds on its second attempt
- **THEN** the record shows two attempts and total time of at least the round trip

#### Scenario: Errored call
- **WHEN** every attempt fails
- **THEN** the error record still shows the attempt count and total time

### Requirement: Cost is reported or estimated, never invented
The system SHALL use a cost reported in Jev's usage object when present. Otherwise it SHALL estimate cost only from user-supplied prices per million tokens. Otherwise cost SHALL be not available. The source of each cost SHALL be recorded, and there MUST NOT be a built-in default price.

#### Scenario: No price and no reported cost
- **WHEN** no prices are configured and Jev reports no cost
- **THEN** cost is not available and the summary says how to configure prices

#### Scenario: Estimated cost
- **WHEN** both prices are configured and Jev reports tokens but no cost
- **THEN** cost equals tokens times price and is marked estimated

#### Scenario: Reported cost
- **WHEN** Jev's usage includes a numeric cost field
- **THEN** that value is used and marked as reported

### Requirement: Batch telemetry summary
The batch summary SHALL report token totals and means, total and per-run cost, round-trip, Jev-elapsed and overhead statistics, attempts, retried runs, HTTP status counts, and per-scenario means. The summary file SHALL record start and end time, wall time, runs per minute, git commit, Python version, platform, and the Jev URL and model requested, and MUST NOT contain the API key.

#### Scenario: Summary contents
- **WHEN** a batch finishes
- **THEN** the printed summary has a Telemetry section and the summary file has the batch metadata
