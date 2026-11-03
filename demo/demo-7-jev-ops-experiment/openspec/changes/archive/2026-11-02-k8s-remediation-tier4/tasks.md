## 1. Verify on the cluster before coding against it

- [x] 1.1 Read the LF Edge harness (`lfedgeai/AIOps`, `research/agentic_aiops_architectures/code/harness/run_harness.py`) for how it sets `remediation_executed`, `recovery_verified`, `mttr_seconds` and `mttr_verified_seconds`, and whether its fault stays injected during recovery; record it in `docs/k8s-testbed-notes.md` and adjust the D6 export mapping if needed (2026-10-18: mttr = fault to first write, mttr_verified = fault to target check passing; D6, D7 and the recovery spec aligned)
- [x] 1.2 Start the cluster (`tier3/testbed/start.sh`), confirm health, and record `revisionHistoryLimit` and the current revision of every demo Deployment (2026-10-18: healthy; every Deployment revisionHistoryLimit 10; cart at revision 38, other app Deployments 8-9)
- [x] 1.3 By hand, for each fault on its Tier 3 target, apply each allowlisted action on the right component (`scale_up`, `restart_service`, `rollback_change`) and record whether and when the service-level recovery check (D5) would pass; note which faults no allowlisted action can fix (expected: `network_partition`, `node_taint`)
- [x] 1.4 By hand, apply `restart_service` and `rollback_change` to an unrelated healthy Deployment (`frontend`) and record the disturbance and the revision `rollout undo` picks after `rebalance.sh` has run
- [x] 1.5 Confirm that `kubectl rollout undo` and a rollout restart leave no annotation or label that names the fault or the experiment, and that the harness's exact restore brings the Deployment back after each action
- [x] 1.6 Measure the recovery check's noise on the healthy cluster: run it for 360 s with no fault and record any failing checks; tune nothing that the scenarios' expected answers depend on (25 of 25 checks passed over 360 s; 75 of 75 with the two K000 frontend probes)
- [x] 1.7 Write the findings to `docs/k8s-testbed-notes.md` (a Tier 4 section: actions per fault, recovery times with the right action, history behaviour, noise)

## 2. Executor

- [x] 2.1 `tier4/executor.py`: allowlist mapping (D2), target restrictions with reasons, replica cap, one call per run
- [x] 2.2 `scale_up` and `restart_service` through the Tier 3 Kubernetes client (`patch`); `rollback_change` through pinned `kubectl rollout undo` with the cluster's kubeconfig, recording revisions before and after
- [x] 2.3 Save the target Deployment's replicas and pod template to an `action` entry in the ledger before acting; read back after acting
- [x] 2.4 Extend reset: restore the action entry first, then the fault (D4); make `python -m tier4 reset-faults` and the dead-man timer handle a ledger with an action (Tier 3 `reset_faults` restores a ledger `action` entry first; Tier 3 runner gained optional `after_scored`, `on_reset`, `stop_at_first_detection`, `extra_hold_s`, unset by default)
- [x] 2.5 Offline tests: each action, each refusal reason, the cap, the second-call refusal, ledger before act, read-back mismatch, reverse-order restore, crash recovery from the ledger alone

## 3. Recovery

- [x] 3.1 `tier4/recovery.py`: the service-level check (D5) from Kubernetes state and service stats, ignoring replaced pods and restarts
- [x] 3.2 Hold period (60 s) and window (360 s), stats over 120 s; `t_recovered`, `t_recovery_verified`, MTTR, verified MTTR, time to remediate (D6); `null` when not measurable
- [x] 3.3 Outcomes `recovered_by_action`, `recovered_without_action`, `not_recovered` (D7); `action_on_false_alarm` and `disturbed_healthy_system` (D8) (outcome uses whether the passing run began at the pre-action check, not timestamps)
- [x] 3.4 Offline tests with a virtual clock: early pass then fail, window end, self-healing, recovery before the action completes, false-alarm action that disturbs

## 4. Runner, results and reports

- [x] 4.1 `tier4/runner.py`: Tier 3 lifecycle to the first detecting poll, Tier 3 verdicts on that poll, executor once, recovery, reset in `finally`, contamination stop, rebalance between runs
- [x] 4.2 Results under `results/tier4/<run-id>/` with the action, every recovery check, outcome and timing fields
- [x] 4.3 Whole-path leak test over every scenario: no fault name, `jev-fault`, expected answer, executor result or recovery result in any request
- [x] 4.4 Isolation tests (D9): only `tier4/runner.py` and `tier4/cli.py` import the executor; nothing outside `tier4/` does; no Tier 4 module imports `jev.client`
- [x] 4.5 `tier4/report.py`: scorecard (D10) and LF Edge export with measured remediation fields; MTTR from `recovered_by_action` runs only
- [x] 4.6 `tier4/cli.py`: `list`, `run`, `batch`, `reset-faults`, `scorecard`, `export-lfedge`; root wrappers `run-remediation-experiment` and `run-all-remediation-experiments`

## 5. Policy and documentation

- [x] 5.1 `CLAUDE.md`: narrow the "never act" rule to Tier 4 (D11), add a Tier 4 rules section, layout and commands; Tier 1 to 3 rules unchanged
- [x] 5.2 `README.md`: Tier 4 section with the allowlist, what MTTR measures here, and what is and is not comparable with the group's agents
- [x] 5.3 New settings, if any, in `.env.example` and the README table (none needed: Tier 4 reuses the Tier 3 settings)
- [x] 5.4 Confirm `./run-all.sh --check` runs the Tier 4 tests and CI passes on Python 3.10 to 3.13; Tier 1 to 3 tests pass unchanged (CI run on 8abf598, 2026-10-18: all jobs pass; Tier 4 tests collected via `testpaths = tests`)

## 6. Live runs

- [x] 6.1 Run each scenario once with Jev through Tier 4; read each request by hand for leaks, and check every action's record and restore (2026-10-18: `results/tier4/20261018-210528-2703` K000, `20261018-212756-f595` K001-K010; a session reset killed the first pass after K000)
- [x] 6.2 Batch `K001` to `K010` with `--repeats 3`, plus at least 10 `K000` runs (2026-10-19: `results/tier4/20261018-232405-5ada`, `20261019-050044-6cc4`; 40/40 ok)
- [x] 6.3 Print the scorecard and the LF Edge export (`results/tier4/6.2-combined-scorecard.md`, `6.2-combined-lfedge.jsonl`)
- [x] 6.4 Write `docs/results-<date>-tier4-jev.md` from the run records only, stating what is not comparable with the group's agents; stop the cluster (`docs/results-2026-10-19-tier4-jev.md`; cluster stopped 2026-10-19 after 7.5)

## 7. Opus baseline engine, same interface (after group 6)

- [x] 7.1 Move Tier 3's question set and answer vocabularies out of `tier3/jev_adapter.py` into a module with no Jev import (`tier3/questions.py`); the Jev adapter imports it; Tier 3 tests and the "only the adapter imports `jev.client`" test still pass
- [x] 7.2 `tier3/claude_engine.py` (an `AIOpsEngine`): the same observation and the same four questions to `claude-opus-5-5` through the Anthropic API, structured output restricted to the same answer sets, strict parsing (out-of-set is `invalid`), tokens and cost from `CLAUDE_OPUS_PRICE_*` only, `n/a` otherwise; reuse `llm/claude.py` settings (temperature, effort) where they fit
- [x] 7.3 Tests: the prompt carries no scenario, fault name, expected answer or harness term (the Tier 3/4 whole-path leak tests run with this engine's request builder); parsing of valid, out-of-set and missing answers; no API call in tests
- [x] 7.4 `--engine jev|claude-opus` on `python -m tier3` and `python -m tier4` (`run`, `batch`), recorded in every run record and in the LF Edge export's `approach`
- [x] 7.5 Live: Tier 4 batch with `--engine claude-opus` on the same scenarios and repeats as 6.2 (`K001`-`K010` x3, `K000` x10) (2026-10-19: `results/tier4/20261019-070552-2d0e`, `20261019-122929-7076`; 40/40 ok, transport `claude-cli`; `7.5-combined-scorecard.md`, `7.5-combined-lfedge.jsonl`)
- [x] 7.6 Write the comparison from the run records only: Jev vs Opus on detect, RCA, action, single-shot MTTR and verified MTTR, recovered by action, false-alarm actions, cost; label every MTTR "single-shot recommendation MTTR" (`docs/results-2026-10-19-tier4-jev-vs-opus.md`)
