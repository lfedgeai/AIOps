# Run notes — 11 August 2026 (Option 2 re-run, expanded MaaS lineup)

## Agent lineup change (vs 10–11 Aug run)

**Replaced:**

| Was | Now |
|-----|-----|
| `nemotron-nano-3` (`nvidia/nemotron-3-nano`) | `nemotron-cascade-2` (`nemotron-cascade-2-30b`) |

**Added:**

| Approach | Model | Endpoint family |
|----------|-------|-----------------|
| `maas_qwen35-9b` | `qwen35-9b` | prelude-maas |
| `maas_qwen36-27b` | `qwen36-27b` | prelude-maas |
| `maas_kimi-k2-7` | `kimi-k2-7` | prelude-maas |

**Kept:** `maas_deepseek`, `maas_qwen3`, `maas_granite`, `maas_llama31-70b`

**Default matrix agents (8):** cascade, deepseek, qwen3, qwen35-9b, qwen36-27b, kimi-k2-7, granite, llama31-70b

**Fairness notes (this run):**
- Shared prompts via `code/agents/prompts.py` (fixes Aug Granite/Llama prompt skew)
- API preflight aborts matrix if any default agent chat endpoint fails
- Do **not** compare aggregate rates 1:1 vs Aug (5→8 agents; different Nemotron; new Qwen/Kimi)

## Suite

Option 2 — scenario **A** × all CONTEXT_C subsets (16) × **8** agents × 3 faults = **384 runs**  
Script: `scripts/run_option2_permutations_systemd.sh` → `run_c_matrix_scenario_a_c_permutations.sh`  
Log: `out/c_matrix_scenario_a_c_permutations_run.log`  
Service: `aiops-c-matrix-sa-c-permutations.service`  
ETA: **~22–26 h** (scaled from Aug ~219 s/run)  
**Started:** Tue 11 Aug 2026 **21:58:56 AEST** (`aiops-c-matrix-sa-c-permutations.service`)  
**Completed:** Wed 12 Aug 2026 **21:15:41 AEST** — **384/384**, `failures=0`  
**Report:** [EVALUATION_RESULTS_12Aug26.md](EVALUATION_RESULTS_12Aug26.md)  
Prior report: [EVALUATION_RESULTS_11Aug26.md](EVALUATION_RESULTS_11Aug26.md)

## Preflight at kickoff

- All 8 agent APIs: PASS (chat)
- Chat + native tools smoke (cascade, qwen35, qwen36, kimi): PASS
- MLflow local `:5050` up; ClickHouse PF up; `cart` 1/1

## Kickoff fix (approach naming)

First start used `--classifier` which tagged MLflow/`approach` as directory names (`nemotron_agent`). **Stopped**, switched matrix scripts to `--agents <approach>` so tags match lineup IDs (`nemotron-cascade-2`, …). Restarted clean.

## Planned analysis (post-run)

1. Harness JSON vs MLflow metrics congruence — **done** (384/384 matched, 0 mismatches)  
2. Family comparisons: Nemotron (nano Aug vs cascade now); Qwen (qwen3 vs qwen35 vs qwen36) — **done**  
3. Top CONTEXT_C × agent × fault combinations — Aug leaders vs this run — **done**  
4. Report: [EVALUATION_RESULTS_12Aug26.md](EVALUATION_RESULTS_12Aug26.md) — **done**
