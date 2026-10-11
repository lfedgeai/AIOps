# Run notes — 25 September 2026 (skills + no-CHANGELOG Option 2 re-run)

## Experiment deltas vs 22–23 Sep matrix ([EVALUATION_RESULTS_23sep26.md](EVALUATION_RESULTS_23sep26.md))

| Change | Detail |
|--------|--------|
| Shared skills | `code/agents/prompts.py` injects detect/diagnose/remediate/verify for **all** agents (`AGENT_SKILLS=1`) |
| Playbook reference | `code/agents/playbooks/scenario_a_cart.md` appended after skills (not a skill) |
| RAG docs corpus | **CHANGELOG.md removed**; index `INDEX_VERSION=no-changelog-v1` |
| Agent timeout | `AGENT_TIMEOUT` **900s** (was 600; qwen36 had 32/48 timeouts) |
| Detection window | `--detection-timeout` **1200s** (was 720) |

**Invalidates** prompt/corpus-sensitive comparison as a pure model bake-off against Sep 22–23; use as A/B for skills+corpus hygiene + timeout fix.

## Preflight (2026-09-25)

All 7 lineup agents PASS `scripts/preflight_agent_apis.py` (including `maas_llama31-70b`, `maas_qwen36-27b`).  
`scripts/pre_run_health_gate.py` OK.

Prior-run “access” issues:
- **Llama 0/48 detect** — LLM invoked; not an API outage (model/behavior). Still in lineup; preflight OK.
- **Qwen36 32/48 `agent_error`** — almost all `timeout after 600s` → raised timeout/detection window.

## Kickoff

**Started:** Fri 25 Sep 2026 **14:06:25** AEST (`aiops-c-matrix-sa-c-permutations.service`)  
**Baseline archive:** `out/archive_sep22_option2/` (frozen Sep 22–23 JSON before overwrite)

```bash
systemctl --user status aiops-c-matrix-sa-c-permutations.service
tail -f out/c_matrix_scenario_a_c_permutations_run.log
# After COMPLETE:
python3.13 scripts/compare_option2_kickoffs.py \
  --baseline-kickoff "2026-09-22 00:02:00" \
  --baseline-dir out/archive_sep22_option2 \
  --new-kickoff "$(cat out/.matrix_kickoff_aest)" \
  --markdown docs/COMPARE_25sep26_vs_23sep26.md
# Also auto-updates docs/Learnings_across_runs_report.md via the matrix script.
```

Cross-run synthesis: [Learnings_across_runs_report.md](Learnings_across_runs_report.md)
