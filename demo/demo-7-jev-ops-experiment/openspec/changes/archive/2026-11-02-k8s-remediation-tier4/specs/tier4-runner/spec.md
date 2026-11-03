## ADDED Requirements

### Requirement: Closed-loop lifecycle reusing Tier 3
For each experiment the Tier 4 runner SHALL run Tier 3's lifecycle up to the first detecting poll (health, workload, leftovers check, baseline, save and inject, poll), score that poll with Tier 3's four verdicts, call the executor once on that poll's decision, run the recovery check, and in a `finally` block reset the action and the fault and check the cluster's recovery as Tier 3 does. When no poll detects, the executor SHALL NOT be called and the recovery check SHALL start at the end of the detection window.

#### Scenario: Detected and acted
- **WHEN** poll 1 reports an incident with `scale_up` on `cart`
- **THEN** no further Jev call is made, the executor runs once, recovery is checked, and reset runs

#### Scenario: Never detected
- **WHEN** no poll reports an incident on K001
- **THEN** the executor is not called, the outcome is decided by the recovery check, and reset runs

#### Scenario: Engine call fails
- **WHEN** the engine raises during a poll
- **THEN** that poll is an error as in Tier 3, and reset still runs

### Requirement: Ground truth stays hidden
The engine SHALL receive only Tier 3 observations. The executor's and the recovery check's results SHALL NOT appear in any request. The whole-path leak test SHALL cover every Tier 4 scenario.

#### Scenario: Leak test
- **WHEN** every scenario runs through the Tier 4 path with a fake engine
- **THEN** no request contains a fault name, `jev-fault`, the expected answer, an executor result or a recovery result

### Requirement: Contamination stops the batch
A run whose cluster does not return to the saved baseline after reset SHALL be `contaminated`, and the batch SHALL stop. The batch SHALL rebalance application pods between runs as Tier 3 does.

#### Scenario: Action not restored
- **WHEN** the action's saved replica count cannot be restored
- **THEN** the run is `contaminated` and the batch stops

### Requirement: Results and scorecard
Each run SHALL write `results/tier4/<run-id>/<experiment_id>.json` with everything Tier 3 records, plus the action (operation, target, reason when not executable, saved state, read-back, revisions for a rollback), every recovery check, the outcome and the timing fields. `python -m tier4 scorecard` SHALL show Tier 3's columns plus remediation executed, not executable, recovered by action, recovered without action, not recovered, MTTR median (min-max) over runs with an executed action, verified MTTR median (min-max) over `recovered_by_action` runs, self-recovery median, false-alarm actions and disturbed healthy systems. `python -m tier4 export-lfedge` SHALL fill `remediation_executed`, `recovery_verified`, `mttr_seconds` and `mttr_verified_seconds` from the records.

#### Scenario: Verified MTTR excludes self-healing
- **WHEN** K002 recovers without action in 20 s and K001 recovers by action with verified MTTR 50 s
- **THEN** the verified MTTR median is computed from the K001 run only, and K002 is counted under recovered without action with a self-recovery time of 20 s

### Requirement: Executor isolation
Only `tier4/runner.py` and `tier4/cli.py` SHALL import `tier4/executor.py`. No module outside `tier4/` SHALL import it, and no Tier 4 module SHALL import `jev.client`.

#### Scenario: Import guard
- **WHEN** a module in `tier3/` imports `tier4.executor`
- **THEN** the isolation test fails
