## ADDED Requirements

### Requirement: Complete server observations
The generator SHALL produce, for each run, CPU utilisation %, memory utilisation %, disk utilisation %, system load, network errors, application error rate, API latency, service statuses and recent events/restarts.

#### Scenario: All fields populated
- **WHEN** an observation is generated for any scenario
- **THEN** every listed field is present and within its valid range (percentages 0–100, non-negative counts and latency)

### Requirement: Scenario-correlated metrics
Generated metrics SHALL be correlated with the chosen scenario rather than drawn as independent random numbers.

#### Scenario: Healthy state looks healthy
- **WHEN** many `healthy` observations are generated across seeds
- **THEN** their resource use, error rate and latency stay within the healthy profile bands and all services report healthy

#### Scenario: Severity ordering holds
- **WHEN** many observations are generated for `healthy`, `degraded` and `critical`
- **THEN** the mean CPU, error rate and latency increase in that order

#### Scenario: Contradictory state breaks correlation deliberately
- **WHEN** a `contradictory` observation is generated
- **THEN** it contains signals that conflict, such as low resource use alongside a failing service or high error rate, or high resource use alongside healthy services and no errors

### Requirement: Seed-based reproducibility
The generator SHALL accept a seed and use only a private random source derived from it, so that the same scenario and seed always yield identical observations.

#### Scenario: Same seed, same output
- **WHEN** an observation is generated twice with the same scenario and seed
- **THEN** the two observations are equal

#### Scenario: Different seed, different output
- **WHEN** observations are generated with two different seeds for the same scenario
- **THEN** the observations differ

#### Scenario: No global RNG side effects
- **WHEN** the generator runs
- **THEN** it does not read or modify Python's global random state

### Requirement: Observations are serialisable
Observations SHALL convert to a plain dictionary suitable for JSON storage and for inclusion in the Jev request.

#### Scenario: Round trip to JSON
- **WHEN** an observation is converted to a dictionary and dumped as JSON
- **THEN** the dump succeeds and contains every observation field
