## ADDED Requirements

### Requirement: Collectors for real telemetry
The harness SHALL collect metrics from Prometheus, traces from Jaeger and logs from OpenSearch over HTTP. Each collector SHALL accept an injected request function so tests run without the testbed.

#### Scenario: Offline collection
- **WHEN** a collector is given a fake request function that returns fixture data
- **THEN** it returns parsed telemetry without any network access

#### Scenario: Backend unavailable
- **WHEN** a backend does not respond
- **THEN** the collector raises an error that names the backend, and the run is recorded as an error, not a pass or fail

### Requirement: Normalized Observation
The system SHALL convert collector output into an `Observation` containing the time window with timestamps, per-service rows (request rate, error rate, p50 and p95 latency, and change versus the baseline window), dependency edges derived from traces, deduplicated error log lines per service, and failing-trace summaries.

#### Scenario: Observation fields
- **WHEN** an observation is built from fixture telemetry
- **THEN** it contains every field group above and a window start and end

#### Scenario: Same shape every run
- **WHEN** observations are built for a healthy and a faulty fixture
- **THEN** they have the same field groups and the same caps

### Requirement: Bounded size
The builder SHALL rank services by change versus baseline and apply named caps (services, log lines per service, traces) and a total byte budget, trimming the lowest-ranked items first. It SHALL record what was trimmed in the result record and MUST NOT put trim notes in the Jev request.

#### Scenario: Over budget
- **WHEN** fixture telemetry exceeds the byte budget
- **THEN** the serialized observation is within the budget and the lowest-ranked items were removed first

#### Scenario: Trim recorded
- **WHEN** items were trimmed
- **THEN** the result record lists what was removed and the Jev request does not mention it

### Requirement: Observation is built without ground truth
The observation builder SHALL accept only collector output and the window. Its signature MUST NOT include the scenario, the fault, or the ground truth.

#### Scenario: Signature
- **WHEN** the builder's parameters are inspected
- **THEN** none of them is a scenario, fault or ground-truth type

### Requirement: Scrubber
Before serialization, the system SHALL remove from all telemetry any key or value containing `feature_flag` (this drops flag evaluations and variants, which the demo emits on spans and logs), replace every flag name defined in the pinned flag file and the term `flagd` wherever they appear in text, and replace the experiment ID of the current run. Flag variants MUST NOT be scrubbed as free text, because variants such as `on` and `off` occur in ordinary words. Short terms such as the experiment ID SHALL match only at word boundaries so that hex trace IDs are not altered.

#### Scenario: Flag evaluation data on a span
- **WHEN** fixture trace data contains a `feature_flag` attribute naming the injected flag
- **THEN** the serialized observation contains neither the attribute nor the flag name

#### Scenario: Flag name in a log line
- **WHEN** a fixture log line contains a flag name
- **THEN** that text is replaced with a redaction marker

#### Scenario: Ordinary words survive
- **WHEN** a log line contains the words "on" or "off" in normal text
- **THEN** the text is unchanged

#### Scenario: Hex trace IDs survive
- **WHEN** a trace ID contains characters that look like an experiment ID
- **THEN** the trace ID is unchanged

### Requirement: Harness and test infrastructure are excluded
The collectors SHALL exclude `flagd`, `flagd-ui`, `telemetry-docs` and `load-generator` from services, dependency edges, logs and traces, because they are the fault-injection and load machinery and not part of the application the AI reasons about.

#### Scenario: flagd traffic
- **WHEN** span metrics include calls to flagd
- **THEN** the observation has no `flagd` service row and no edge to it

### Requirement: No answer in the request
The state sent to the engine for every scenario SHALL NOT contain the experiment ID, flag names, `feature_flag` data, the expected diagnosis or acceptable-action labels, or any evaluation result. The question set sent with the state SHALL be identical for every scenario, so the answer vocabulary cannot hint at the answer.

#### Scenario: Leak test over the catalogue
- **WHEN** a request is built for every scenario from fixture telemetry that deliberately contains these strings
- **THEN** none of them appears in the serialized state

#### Scenario: Same questions everywhere
- **WHEN** requests are built for different scenarios
- **THEN** their question sets are identical

### Requirement: Documented content
The README SHALL state exactly which telemetry fields Jev receives, the caps and the byte budget, and the limit that service names and error text are evidence and not treated as leakage.

#### Scenario: README lists inputs
- **WHEN** the README section on observations is read
- **THEN** it lists every field group, every cap and the budget
