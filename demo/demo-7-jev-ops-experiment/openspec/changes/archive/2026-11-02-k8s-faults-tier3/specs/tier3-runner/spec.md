## ADDED Requirements

### Requirement: Lifecycle with reset in finally
For each experiment the runner SHALL check health, set the workload, refuse to start if a ledger or harness object exists, collect the baseline, record pod placement, save state and inject, detect (poll or single mode), store every decision, evaluate, and in a `finally` block reset and check recovery.

#### Scenario: Engine call fails
- **WHEN** the engine raises during a poll
- **THEN** that poll is recorded as an error, polling continues until the window ends, and reset and recovery still run

### Requirement: Polling detection
In `poll` mode the runner SHALL wait `telemetry_lag_s` after inject, then ask the engine every `poll_interval_s` with a fresh observation until it reports an incident or `detection_window_s` ends. The first detecting poll SHALL set `t_detect` and its decision SHALL be scored; if none detects, the last poll's decision SHALL be scored and `t_detect` SHALL be `null`. For `K000`, polling SHALL run the whole window and every detecting poll SHALL count as a false-positive poll. `single` mode SHALL keep the Tier 2 behaviour.

#### Scenario: Detected on the third poll
- **WHEN** polls 1 and 2 report no incident and poll 3 reports one
- **THEN** `t_detect` is poll 3's time, MTTD is `t_detect - t_inject`, and poll 3's decision is scored

#### Scenario: Healthy control with one false alarm
- **WHEN** one of eight `K000` polls reports an incident
- **THEN** the run is a false-positive run with 1 false-positive poll out of 8

### Requirement: Recovery covers the cluster
After reset the run SHALL count as recovered only when every demo deployment is back at its saved replicas and available, no harness object or ledger remains, every node is Ready and free of `jev-fault` taints, and service stats are back to baseline within the timeout. Otherwise the run SHALL be `contaminated` and the batch SHALL stop.

#### Scenario: Rollout still in progress
- **WHEN** cart's restored rollout is not complete when the recovery timeout ends
- **THEN** the run is `contaminated` and the batch stops

### Requirement: Four independent verdicts
Detection, localization, diagnosis and action SHALL each be PASS, FAIL or UNKNOWN using the Tier 2 rules on `affected_component`. No composite score. UNKNOWN only for an `invalid` or `error` decision or a missing field.

#### Scenario: Missed incident still scores other fields
- **WHEN** the scored decision reports no incident for `K001` but names `cart` and `service_unavailable`
- **THEN** detection is FAIL, localization is PASS and diagnosis is PASS

### Requirement: Results
Each run SHALL write `results/tier3/<run-id>/<experiment_id>.json` with fault, saved and restored state, ground truth, placement, every poll (time, exact request, decision, raw response, reasoning), the scored decision, verdicts, `t_inject`, `t_manifest`, `t_detect`, MTTD (`null` when not detected), Jev call count, tokens and cost where reported, trim notes, versions and git commit.

#### Scenario: Not detected
- **WHEN** no poll detects
- **THEN** `t_detect` and `mttd_seconds` are `null`

### Requirement: Scorecard in the group's terms
`python -m tier3 scorecard <results dir>` SHALL print per scenario and overall: detect, RCA (localization), diagnosis, remediation correct (action), MTTD median and range, false-positive runs and polls, detection precision, agent errors, Jev calls, tokens and cost. Remediation executed, MTTR and recovery verified SHALL be printed as "not measured (recommendation only)". Counts SHALL be shown with their denominators.

#### Scenario: Unmeasured fields
- **WHEN** the scorecard is printed
- **THEN** the remediation-executed, MTTR and recovery rows say "not measured (recommendation only)"

### Requirement: Export for the LF Edge harness
`python -m tier3 export-lfedge <results dir>` SHALL write one JSON line per run with the group's field names (checked against their harness in task group 1), `null` for fields we do not measure, and our experiment ID alongside.

#### Scenario: Export of a detected run
- **WHEN** a `K001` run detected the fault after 47 s and localized `cart`
- **THEN** its line has `detected: true`, `rca_correct: true`, `mttd_seconds: 47`, and `remediation_executed: null`

### Requirement: Reasoning export for hand scoring
`python -m tier3 export-reasoning <results dir>` SHALL write a Markdown sheet with experiment ID, observation summary, the scored poll's reasoning text, ground truth and empty score columns. Hand scores SHALL never be folded into the verdicts.

#### Scenario: No reasoning
- **WHEN** the scored decision has no reasoning
- **THEN** its row says "no reasoning returned"
