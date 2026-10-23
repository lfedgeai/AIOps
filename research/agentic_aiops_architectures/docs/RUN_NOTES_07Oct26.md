# Run notes — Option 2 (Oct 7–9 execute-gate matrix + post-run cleanup)

## Oct 7–9 matrix (completed)

| Parameter | Value |
|-----------|--------|
| Script | `scripts/run_c_matrix_scenario_a_c_permutations.sh` |
| Scenario / variant | `a` / `cart` |
| Faults | `scale_zero`, `config_corruption`, `kill_pod` |
| Contexts | 16 (C0 + all non-empty subsets of {1,2,3,4}) |
| Agents (that run) | deepseek, qwen3, qwen38-27b, qwen36-27b, granite, llama31-70b, glm-53-flash |
| Omitted at kickoff | `nemotron-cascade-2` (prelude timeout / 503) |
| Execute gate | `AGENT_EXECUTE_GATE=1` |
| Skills / RAG | `AGENT_SKILLS=1`, `FORCE_RAG=1`, `INDEX_VERSION=no-changelog-v1` |
| Timeouts | `AGENT_TIMEOUT=900`, detection 1200s |
| Result | **336/336**, report [EVALUATION_RESULTS_09oct26.md](EVALUATION_RESULTS_09oct26.md) |

## Post-run cleanup (next matrix; invalidates prompt-sensitive compare to Oct cells)

| Change | Detail |
|--------|--------|
| Generic prompts/skills | No app/service-name hints in `prompts.py` or `skills/*` |
| Playbooks | No longer injected into `BASE_SYSTEM_PROMPT` |
| Llama removed | `maas_llama31-70b` dropped from default matrix scripts (0/48 detect) |
| Next design total | 6 agents × 16 contexts × 3 faults = **288** cells |

## Next matrix (generic prompts − playbook − llama)

| Parameter | Value |
|-----------|--------|
| Agents | deepseek, qwen3, qwen38-27b, qwen36-27b, granite, glm-53-flash (**6**) |
| Design total | 6 × 16 × 3 = **288** |
| Prompts | Application-agnostic (`prompts.py` / skills); no playbook inject |
| Execute gate | `AGENT_EXECUTE_GATE=1` (unchanged) |
| Baseline for Δ | [EVALUATION_RESULTS_09oct26.md](EVALUATION_RESULTS_09oct26.md) — **not** a pure re-run |

```bash
systemctl --user start aiops-c-matrix-sa-c-permutations.service
tail -f out/c_matrix_scenario_a_c_permutations_run.log
```
