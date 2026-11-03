## ADDED Requirements

### Requirement: One-command workflow
The repository SHALL provide root scripts `setup.sh`, `health-check.sh`, `run-experiment` and `run-all-experiments` that wrap the Tier 2 code, so a developer can reproduce an experiment from a fresh clone.

#### Scenario: Fresh clone
- **WHEN** a developer clones the repo and follows the README
- **THEN** `./setup.sh`, `./health-check.sh` and `./run-experiment F001` run in that order without extra manual steps beyond setting the Jev key

### Requirement: README covers Tier 2
The README SHALL explain: what the project tests; why the OpenTelemetry Demo is used; the architecture; prerequisites; how to start the demo; how to verify telemetry; how faults are injected; how Jev receives observations; how experiments are run; how results are evaluated; and how to add a new experiment.

#### Scenario: Sections present
- **WHEN** the README is read
- **THEN** each topic above has its own section

#### Scenario: Add an experiment
- **WHEN** a developer follows the "add a new experiment" section
- **THEN** a new scenario file is enough for it to appear in the scenario list and run

### Requirement: Tiers are labelled
The README and docs SHALL describe Tier 1 as synthetic controlled observations testing basic decision reasoning, and Tier 2 as the OpenTelemetry Demo testing AIOps against realistic distributed-system telemetry. The two tiers SHALL NOT share implementation beyond the Jev client.

#### Scenario: Tier 1 unchanged
- **WHEN** `./run-all.sh --check` is run
- **THEN** all Tier 1 tests still pass

### Requirement: Policy documents updated
`CLAUDE.md` SHALL be updated to allow Docker Compose for the Tier 2 testbed only, keep Kubernetes out of scope, and add the Tier 2 rules: no ground truth in observations, scrubber required, four separate verdicts, no composite score, reset always runs.

#### Scenario: Compose rule
- **WHEN** `CLAUDE.md` is read
- **THEN** it no longer forbids Compose for Tier 2 and still forbids Kubernetes

### Requirement: Offline CI
CI SHALL run the Tier 1 and Tier 2 unit tests offline, without Docker Compose, the demo or a Jev key.

#### Scenario: CI run
- **WHEN** CI runs `./run-all.sh --check`
- **THEN** the Tier 2 tests run and pass with no network access to Jev

### Requirement: Notes on the demo
`docs/otel-demo-notes.md` SHALL record the verified flags, ports, query examples and gotchas for the pinned version, including which planned faults were dropped and why.

#### Scenario: Notes exist
- **WHEN** the file is read
- **THEN** it lists each flag used, how it was verified, and the pinned versions

### Requirement: Dependencies pinned
New Python dependencies SHALL be pinned in `requirements.txt`, and new environment variables SHALL be documented in `.env.example` and the README.

#### Scenario: Pinned
- **WHEN** `requirements.txt` is read
- **THEN** the YAML library has an exact version
