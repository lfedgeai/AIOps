## ADDED Requirements

### Requirement: Entry points
The repository SHALL provide root wrappers `setup-k8s.sh` (pre-check, create, install, health), `health-check-k8s.sh`, `run-k8s-experiment <ID>` and `run-all-k8s-experiments [--repeats N]`, and `python -m tier3 list`, `run`, `batch`, `reset-faults`, `scorecard`, `export-lfedge` and `export-reasoning`.

#### Scenario: List
- **WHEN** `run-k8s-experiment list` runs
- **THEN** it prints every Tier 3 scenario ID with its fault type and description

### Requirement: Policy and docs updated
`CLAUDE.md` SHALL allow Kubernetes (a local kind cluster) for the Tier 3 testbed only, keep the rule that nothing acts on the engine's recommendation, and add the Tier 3 rules. `README.md` SHALL describe Tier 3, its relation to the LF Edge harness and what is not comparable, host requirements, commands and the scenario table, and list new environment variables, which SHALL also be in `.env.example`. `docs/k8s-testbed-notes.md` SHALL record every verified fact, version, fault behaviour and dropped scenario.

#### Scenario: Policy text
- **WHEN** `CLAUDE.md` is read after this change
- **THEN** it limits Kubernetes to the Tier 3 kind testbed and still forbids acting on the recommendation

### Requirement: Offline tests and CI
Tier 3 tests SHALL use a fake Kubernetes API, fake collectors, a fake engine and a virtual clock. `./run-all.sh --check` SHALL run them, and CI SHALL pass on Python 3.10 to 3.13 with no cluster, Docker, network or API key.

#### Scenario: Fresh clone check
- **WHEN** `./run-all.sh --check` runs on a fresh clone with no cluster and no key
- **THEN** Tier 1, Tier 2 and Tier 3 tests pass

### Requirement: No Kubernetes Python client
Tier 3 SHALL talk to the Kubernetes API with `requests` only. Any new Python dependency SHALL be pinned in `requirements.txt`.

#### Scenario: Requirements
- **WHEN** `requirements.txt` is read
- **THEN** it does not contain the `kubernetes` package
