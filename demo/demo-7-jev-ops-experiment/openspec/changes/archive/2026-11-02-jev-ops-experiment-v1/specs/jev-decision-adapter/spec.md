## ADDED Requirements

### Requirement: API usage follows current Jev documentation
The adapter SHALL use the SDK, endpoint, authentication and request format described in the current Jev documentation, verified before implementation. It MUST NOT use invented parameters. The documentation source and date consulted SHALL be recorded in the repository.

#### Scenario: Documentation recorded
- **WHEN** the adapter is implemented
- **THEN** the repository contains a note naming the Jev documentation consulted, the date, and any confidence/probability field found (or its absence)

### Requirement: Isolated adapter interface
All Jev communication SHALL live behind a single module exposing a decide operation that takes observations and returns a decision. No other module SHALL import Jev SDK or HTTP details.

#### Scenario: Swap with a fake
- **WHEN** the runner is given a fake adapter with the same interface
- **THEN** the experiment runs end to end without contacting Jev

### Requirement: Bounded decisions
The adapter SHALL ask Jev to choose one severity from (`normal`, `degraded`, `high`, `critical`), one action from (`observe`, `investigate`, `restart`, `escalate`), and human-review-required as `yes` or `no`.

#### Scenario: Valid reply accepted
- **WHEN** Jev replies with values from all three allowed sets
- **THEN** the adapter returns a decision containing those values

#### Scenario: Out-of-set reply rejected
- **WHEN** Jev replies with a value outside an allowed set or omits a field
- **THEN** the adapter returns an explicit invalid-decision result and does not coerce or guess a value

### Requirement: Confidence and latency capture
The adapter SHALL return Jev's confidence or probability when the API provides one, otherwise a null value, and SHALL measure request latency in milliseconds.

#### Scenario: Confidence unavailable
- **WHEN** the Jev response contains no confidence information
- **THEN** the decision's confidence is null and no value is fabricated

#### Scenario: Latency recorded
- **WHEN** a request completes
- **THEN** the decision includes the elapsed time in milliseconds

### Requirement: Raw response retention
The adapter SHALL keep the raw Jev response with each decision for later analysis and MUST NOT include credentials in anything it returns or stores.

#### Scenario: No secrets stored
- **WHEN** a decision is returned and serialised
- **THEN** it contains the raw response body but no API key or authorisation header

### Requirement: Configuration and failure handling
The adapter SHALL read credentials and endpoint settings from environment variables, fail with a clear message when required settings are missing, and retry transient failures a bounded number of times before reporting an error result.

#### Scenario: Missing API key
- **WHEN** the required credential variable is unset
- **THEN** the program exits before any request with a message naming the missing variable

#### Scenario: Persistent transport failure
- **WHEN** requests keep failing after the retry limit
- **THEN** the adapter returns an error result for that run instead of raising, so a batch can continue
