## ADDED Requirements

### Requirement: Entry points
The repository SHALL provide `run-remediation-experiment <ID>` and `run-all-remediation-experiments [IDs...] [--repeats N]` wrappers, and `python -m tier4 list`, `run`, `batch`, `reset-faults`, `scorecard` and `export-lfedge`. Tier 4 SHALL use the Tier 3 testbed scripts (`setup-k8s.sh`, `health-check-k8s.sh`) unchanged.

#### Scenario: Reset from the command line
- **WHEN** a ledger holds an action and a fault
- **THEN** `python -m tier4 reset-faults` restores the action, then the fault, and removes the ledger

### Requirement: Policy and documentation
`CLAUDE.md` SHALL state that only Tier 4 acts on an engine's recommendation, only on the local kind testbed, only through the allowlist, one action per run, with save-before-act and exact restore, and that Tiers 1 to 3 stay recommendation-only. `README.md` SHALL describe Tier 4, its allowlist, what MTTR means here, and what is and is not comparable with the group's agents. `docs/k8s-testbed-notes.md` SHALL record the action probes measured on the cluster.

#### Scenario: Reader checks the rule
- **WHEN** someone reads `CLAUDE.md`
- **THEN** it names Tier 4 as the only tier that acts and lists the allowlist

### Requirement: Offline tests
Tier 4 tests SHALL run offline with a fake Kubernetes API, a fake engine and a virtual clock, and SHALL be part of `./run-all.sh --check` and CI on Python 3.10 to 3.13. No test SHALL start kind or Docker, call `kubectl`, or call Jev.

#### Scenario: CI without a cluster
- **WHEN** CI runs `./run-all.sh --check`
- **THEN** the Tier 4 tests pass with no cluster, Docker or API key
