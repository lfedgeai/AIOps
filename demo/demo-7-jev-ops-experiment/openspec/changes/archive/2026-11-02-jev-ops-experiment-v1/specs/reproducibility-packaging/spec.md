## ADDED Requirements

### Requirement: README explains the experiment
The README SHALL describe the experiment in plain language, include the statement that Jev receives observations and selects from a bounded set of decisions which are compared against independently defined expected outcomes, and state clearly that this is an experimental benchmark and not a production autonomous remediation system.

#### Scenario: Disclaimer present
- **WHEN** a reader opens the README
- **THEN** it contains the experimental-benchmark, not-production statement

### Requirement: Reproduction instructions
The README SHALL document prerequisites, Python version, installation, Jev API configuration, environment variables, running one scenario, running a batch, reproducing a run from a seed, and where results are stored.

#### Scenario: Fresh setup works from docs
- **WHEN** a new user follows the README on a clean machine
- **THEN** they can install dependencies, set variables from `.env.example` and run a single scenario without other guidance

#### Scenario: Non-determinism noted
- **WHEN** a reader reaches the seed instructions
- **THEN** the README explains that the seed fixes the input observations but Jev's answers may still vary

### Requirement: Secrets never committed
The repository SHALL provide `.env.example` with placeholder values and git-ignore `.env`. No API key SHALL appear in any committed file.

#### Scenario: Example env file
- **WHEN** `.env.example` is read
- **THEN** it lists every required variable with placeholders and no real credentials

### Requirement: Unit tests for the experiment machinery
The project SHALL include tests for scenario generation, seed determinism, expected-outcome definitions, comparison logic and result aggregation, runnable with `pytest` without network access or Jev credentials. Tests MUST NOT assert any particular answer from real Jev.

#### Scenario: Offline test run
- **WHEN** `pytest` runs with no Jev credentials and no network
- **THEN** all tests pass

#### Scenario: No Jev-answer assertions
- **WHEN** the test suite is reviewed
- **THEN** every test that exercises the runner uses a fake adapter with scripted decisions

### Requirement: Optional minimal Docker image
The project SHALL provide a minimal Dockerfile (optional for users to use) that runs `run.py` with configuration supplied by environment variables at run time. It MUST NOT require Docker Compose or Kubernetes, and MUST NOT bake credentials into the image.

#### Scenario: Container run
- **WHEN** the image is built and run with the required environment variables
- **THEN** `run.py` executes and writes results to a mounted results directory

#### Scenario: No secrets in image
- **WHEN** the Dockerfile and build context are inspected
- **THEN** no `.env` file or key is copied into the image
