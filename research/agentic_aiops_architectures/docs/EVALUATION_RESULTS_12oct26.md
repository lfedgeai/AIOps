# Evaluation Results — Option 2 matrix (6 agents)

**Generated:** 2026-10-12 06:55 AEST  
**Design:** Option 2 — scenario **A** × all CONTEXT_C subsets (16) × **6 agents** × 3 faults  
**Runs:** **288/288** completed  
**Variant:** `cart` · **Forced RAG:** on for C≠0  
**Tracking:** local MLflow (`http://127.0.0.1:5050`), experiment `agentic_aiops_mttd_mttr`  
**Primary baseline:** [EVALUATION_RESULTS_27sep26.md](EVALUATION_RESULTS_27sep26.md) (Option 2, skills + no-CHANGELOG)  
**Run notes:** [RUN_NOTES_07Oct26.md](RUN_NOTES_07Oct26.md)  

> **Lineup (this run):** `maas_deepseek`, `maas_glm-53-flash`, `maas_granite`, `maas_qwen3`, `maas_qwen36-27b`, `maas_qwen38-27b`.  
> **Comparability:** Shared prompts (`code/agents/prompts.py`) and API preflight. Primary Δ table uses Sep 25–27; older Aug-11 (different design size) appears only in legacy combo sections.

---

## What this report measures

Each of the **288** cells is one harness trial: inject a Kubernetes fault on a target workload (scenario A / cart variant), give the agent a time window with telemetry/K8s tools (and optional RAG corpora), then score the outcome.

| Axis | Values |
|------|--------|
| Scenario | A (single-service fault) |
| CONTEXT_C | 16 labels (`0` + subsets of {1,2,3,4}) |
| Agent | 6 models |
| Fault | `scale_zero`, `config_corruption`, `kill_pod` |

| Metric | Meaning |
|--------|---------|
| **Detect** | Agent recognized a fault within the detection window |
| **RCA correct** | Agent identified the right failing component / root cause |
| **Remediation correct (OR)** | Appropriate fix proposed (judged), whether or not executed |
| **Remediation executed** | Agent performed a remediation action |
| **Recovery verified** | Target workload healthy after the agent’s actions |
| **MTTD** | Seconds from fault → detection (median over *detected* runs with `mttd_seconds`) |
| **MTTR** | Seconds from fault → first successful remediation write (`mttr_seconds`) |
| **MTTR verified** | Seconds from fault → recovery verified (`mttr_verified_seconds`) |
| **`agent_error`** | Harness recorded an agent/API failure for that cell |

**How to read rates:** `successes/trials`. Per-agent rows are **48** trials. Per-context rows are **18** trials (6 agents × 3 faults) when the matrix is complete. “Full win” = detect ∧ RCA ∧ exec on the same cell. MTTR medians use only cells where the field is present (n shown in model×context tables).

---

## Executive summary

| Metric | This run (n=288) | Sep 25–27 (n=335) |
|--------|------------------|-------------------------------------|
| Detection | **247/288 (86%)** | 266/335 (79%) |
| RCA correct | **175/288 (61%)** | 196/335 (59%) |
| Remediation correct (OR) | **203/288 (70%)** | 207/335 (62%) |
| Remediation executed | **50/288 (17%)** | 83/335 (25%) |
| Recovery verified | **49/288 (17%)** | 73/335 (22%) |
| MTTD median (detected) | **117s** | 100s |
| MTTR median | **105s** | — |
| MTTR verified median | **128s** | — |
| `agent_error` | **21/288** | 21/335 |

_Primary baseline = [EVALUATION_RESULTS_27sep26.md](EVALUATION_RESULTS_27sep26.md) (same Option 2 design). Older Aug-11 (5-agent / 240-cell) figures remain in family/combo sections where noted._

### Headline takeaways

1. **Detection vs execution:** detect **86%** (247/288); remediation executed **17%** (50/288) — most models still stop at diagnosis.
2. **RCA vs execute split:** **DeepSeek** RCA **41/48** / exec **0/48**; **Granite** RCA **37/48** / exec **0/48**.
3. **Primary executors this run:** **maas_glm-53-flash** 17/48, **maas_qwen38-27b** 17/48, **maas_qwen36-27b** 16/48.
4. **qwen36-27b:** detect/RCA/exec **35/48** with **12/48** `agent_error` (when healthy, detect usually co-occurs with RCA+exec).
6. **glm-53-flash:** detect **45/48**, RCA **23/48**, exec **17/48** (MTTD med 120s).
6. **Nemotron / Kimi:** not in this batch (omit from headlines until preflight-green).

## By agent (48 runs each)

| Agent | Detect | RCA | Exec | MTTD med | MTTR med | MTTR verified med |
|---|---|---|---|---|---|---|
| `maas_deepseek` | 48/48 (100%) | 41/48 (85%) | 0/48 (0%) | 102s | — | — |
| `maas_glm-53-flash` | 45/48 (94%) | 23/48 (48%) | 17/48 (35%) | 120s | 57s (n=45) | 81s (n=17) |
| `maas_granite` | 47/48 (98%) | 37/48 (77%) | 0/48 (0%) | 72s | — | — |
| `maas_qwen3` | 48/48 (100%) | 40/48 (83%) | 0/48 (0%) | 125s | 93s (n=1) | — |
| `maas_qwen36-27b` | 35/48 (73%) | 17/48 (35%) | 16/48 (33%) | 117s | 117s (n=35) | 115s (n=16) |
| `maas_qwen38-27b` | 24/48 (50%) | 17/48 (35%) | 17/48 (35%) | 158s | 158s (n=24) | 249s (n=16) |

## Model family comparison

### Nemotron

| Lineup | Detect | RCA | Exec | Notes |
|---|---|---|---|---|
| `nemotron-nano-3` (Aug) | 38/96 (40%) | 37/96 (39%) | 34/96 (35%) | prior batch |
| `nemotron-cascade-2` (this run) | n/a | n/a | n/a | not in this batch |

### Qwen family (`qwen3-14b` vs `qwen38-27b` vs `qwen36-27b`)

| Model | Detect | RCA | Exec | MTTD med | MTTR med | MTTR verified med |
|---|---|---|---|---|---|---|
| qwen3 (RHDP) | 48/48 (100%) | 40/48 (83%) | 0/48 (0%) | 125s | 93s (n=1) | — |
| qwen38-27b (MaaS) | 24/48 (50%) | 17/48 (35%) | 17/48 (35%) | 158s | 158s (n=24) | 249s (n=16) |
| qwen36-27b (MaaS) | 35/48 (73%) | 17/48 (35%) | 16/48 (33%) | 117s | 117s (n=35) | 115s (n=16) |
| qwen3 (Aug 11 MLflow) | 26/48 (54%) | 16/48 (33%) | 7/48 (15%) | 140s | — | — |

## Top performing combinations

### This run — best cells (detect ∧ RCA ∧ exec)

**Full wins:** **50/288** (maas_glm-53-flash 17, maas_qwen38-27b 17, maas_qwen36-27b 16)

| Context | Agent | Fault | MTTD | MTTR | MTTR verified |
|---------|-------|-------|------|------|---------------|
| C2 | `maas_qwen36-27b` | scale_zero | 53s | 53s | 89s |
| C4 | `maas_glm-53-flash` | scale_zero | 54s | 37s | 54s |
| C3 | `maas_qwen36-27b` | scale_zero | 55s | 55s | 81s |
| C2 | `maas_glm-53-flash` | scale_zero | 55s | 40s | 55s |
| C1 | `maas_qwen36-27b` | scale_zero | 55s | 55s | 112s |
| C1-4 | `maas_glm-53-flash` | scale_zero | 55s | 33s | 55s |
| C2-3 | `maas_qwen36-27b` | scale_zero | 56s | 56s | 90s |
| C4 | `maas_qwen36-27b` | scale_zero | 57s | 57s | 84s |
| C2-4 | `maas_qwen36-27b` | scale_zero | 62s | 62s | 135s |
| C3 | `maas_glm-53-flash` | scale_zero | 62s | 38s | 62s |
| C1-4 | `maas_qwen36-27b` | scale_zero | 62s | 62s | 93s |
| C2-3 | `maas_glm-53-flash` | scale_zero | 63s | 33s | 63s |
| C1-2-4 | `maas_glm-53-flash` | scale_zero | 63s | 39s | 63s |
| C1-3 | `maas_qwen36-27b` | scale_zero | 65s | 65s | 104s |
| C1-2-4 | `maas_qwen36-27b` | scale_zero | 69s | 69s | 111s |
| C1-3 | `maas_glm-53-flash` | scale_zero | 70s | 35s | 70s |
| C3-4 | `maas_glm-53-flash` | scale_zero | 71s | 37s | 71s |
| C1-3-4 | `maas_qwen36-27b` | scale_zero | 77s | 77s | 117s |
| C1-2-3-4 | `maas_qwen36-27b` | scale_zero | 79s | 79s | 130s |
| C1-3-4 | `maas_glm-53-flash` | scale_zero | 81s | 37s | 81s |

### Top CONTEXT_C × agent (RCA rate, 3 faults per cell)

| Rank | Context | Agent | Detect | RCA | Exec |
|------|---------|-------|--------|-----|------|
| 1 | C4 | `maas_granite` | 3/3 | 3/3 | 0/3 |
| 2 | C4 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 3 | C3-4 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 4 | C3 | `maas_qwen3` | 3/3 | 3/3 | 0/3 |
| 5 | C3 | `maas_granite` | 3/3 | 3/3 | 0/3 |
| 6 | C2-4 | `maas_qwen3` | 3/3 | 3/3 | 0/3 |
| 7 | C2-4 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 8 | C2-3-4 | `maas_qwen3` | 3/3 | 3/3 | 0/3 |
| 9 | C2-3-4 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 10 | C2-3 | `maas_qwen3` | 3/3 | 3/3 | 0/3 |
| 11 | C2-3 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 12 | C2 | `maas_granite` | 3/3 | 3/3 | 0/3 |
| 13 | C2 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 14 | C1-4 | `maas_qwen3` | 3/3 | 3/3 | 0/3 |
| 15 | C1-4 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |

### Aug 11 top combos (from prior report — working agents only)

| Context | Best RCA agents (Aug) | RCA | Exec highlight |
|---------|----------------------|-----|----------------|
| C3 | DeepSeek, Granite | 8/15 each | Qwen only executor |
| C{1,4} | DeepSeek | 8/15 | — |
| C0 | Granite, DeepSeek | weak RCA (3/15 C0) | — |
| Full set C{1,2,3,4} | DeepSeek | 3/3 detect+RCA (Jul Option2) | 0 exec |

### Top combos: Aug 11 → this run (same context, comparable agents)

| Context | Agent | Now RCA | Now exec | Aug 11 (MLflow) |
|---|---|---|---|---|
| C0 | deepseek | 0/3 RCA | 0/3 exec | 2/3 RCA, 0/3 exec |
| C0 | qwen3 | 0/3 RCA | 0/3 exec | 0/3 RCA, 0/3 exec |
| C0 | granite | 0/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C1 | deepseek | 1/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1 | qwen3 | 3/3 RCA | 0/3 exec | 1/3 RCA, 1/3 exec |
| C1 | granite | 3/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C2 | deepseek | 3/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C2 | qwen3 | 2/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C2 | granite | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C3 | deepseek | 2/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C3 | qwen3 | 3/3 RCA | 0/3 exec | 3/3 RCA, 1/3 exec |
| C3 | granite | 3/3 RCA | 0/3 exec | 2/3 RCA, 0/3 exec |
| C4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C4 | qwen3 | 2/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C4 | granite | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1-4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1-4 | qwen3 | 3/3 RCA | 0/3 exec | 2/3 RCA, 0/3 exec |
| C1-4 | granite | 2/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-4 | deepseek | 3/3 RCA | 0/3 exec | 2/3 RCA, 0/3 exec |
| C2-4 | qwen3 | 3/3 RCA | 0/3 exec | 2/3 RCA, 1/3 exec |
| C2-4 | granite | 2/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1-2-3-4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1-2-3-4 | qwen3 | 2/3 RCA | 0/3 exec | 1/3 RCA, 1/3 exec |
| C1-2-3-4 | granite | 2/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-3 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-3 | qwen3 | 3/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C2-3 | granite | 2/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-3-4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-3-4 | qwen3 | 3/3 RCA | 0/3 exec | 0/3 RCA, 0/3 exec |
| C2-3-4 | granite | 2/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |

## MLflow validation

- Harness JSON runs (this batch): **288**
- MLflow runs in window: **288**
- Matched by `run_id`: **288**
- Metric mismatches: **0**

## By context (18 runs each = 6 agents × 3 faults)

| Context | Detect | RCA | Exec | MTTD med | MTTR med | MTTR verified med |
|---|---|---|---|---|---|---|
| C0 | 15/18 (83%) | 3/18 (17%) | 3/18 (17%) | 104s | 93s (n=7) | 128s (n=3) |
| C1 | 16/18 (89%) | 11/18 (61%) | 3/18 (17%) | 102s | 96s (n=7) | 112s (n=3) |
| C2 | 15/18 (83%) | 11/18 (61%) | 3/18 (17%) | 137s | 97s (n=6) | 89s (n=3) |
| C3 | 16/18 (89%) | 12/18 (67%) | 3/18 (17%) | 96s | 157s (n=7) | 81s (n=3) |
| C4 | 15/18 (83%) | 11/18 (61%) | 3/18 (17%) | 133s | 104s (n=6) | 84s (n=3) |
| C1-2 | 16/18 (89%) | 13/18 (72%) | 3/18 (17%) | 113s | 123s (n=7) | 136s (n=3) |
| C1-3 | 15/18 (83%) | 12/18 (67%) | 3/18 (17%) | 108s | 113s (n=6) | 104s (n=3) |
| C1-4 | 14/18 (78%) | 11/18 (61%) | 3/18 (17%) | 136s | 97s (n=6) | 93s (n=3) |
| C2-3 | 14/18 (78%) | 11/18 (61%) | 3/18 (17%) | 109s | 92s (n=5) | 90s (n=3) |
| C2-4 | 15/18 (83%) | 12/18 (67%) | 3/18 (17%) | 104s | 82s (n=6) | 135s (n=3) |
| C3-4 | 16/18 (89%) | 10/18 (56%) | 3/18 (17%) | 130s | 120s (n=7) | 203s (n=3) |
| C1-2-3 | 16/18 (89%) | 12/18 (67%) | 3/18 (17%) | 124s | 99s (n=7) | 137s (n=3) |
| C1-2-4 | 16/18 (89%) | 11/18 (61%) | 3/18 (17%) | 119s | 84s (n=7) | 111s (n=3) |
| C1-3-4 | 17/18 (94%) | 11/18 (61%) | 3/18 (17%) | 120s | 139s (n=8) | 117s (n=3) |
| C2-3-4 | 15/18 (83%) | 12/18 (67%) | 4/18 (22%) | 123s | 110s (n=6) | 227s (n=4) |
| C1-2-3-4 | 16/18 (89%) | 12/18 (67%) | 4/18 (22%) | 134s | 155s (n=7) | 130s (n=3) |

## By agent × context (3 faults each)

One row per (agent, CONTEXT_C). Rates are over the 3 faults; MTTR / MTTR verified medians use cells where the field is present.

| Agent | Context | Detect | RCA | Exec | MTTD med | MTTR med | MTTR verified med |
|---|---|---|---|---|---|---|---|
| `maas_deepseek` | C0 | 3/3 | 0/3 | 0/3 | 104s | — | — |
| `maas_deepseek` | C1 | 3/3 | 1/3 | 0/3 | 98s | — | — |
| `maas_deepseek` | C2 | 3/3 | 3/3 | 0/3 | 88s | — | — |
| `maas_deepseek` | C3 | 3/3 | 2/3 | 0/3 | 91s | — | — |
| `maas_deepseek` | C4 | 3/3 | 3/3 | 0/3 | 78s | — | — |
| `maas_deepseek` | C1-2 | 3/3 | 2/3 | 0/3 | 88s | — | — |
| `maas_deepseek` | C1-3 | 3/3 | 3/3 | 0/3 | 98s | — | — |
| `maas_deepseek` | C1-4 | 3/3 | 3/3 | 0/3 | 113s | — | — |
| `maas_deepseek` | C2-3 | 3/3 | 3/3 | 0/3 | 113s | — | — |
| `maas_deepseek` | C2-4 | 3/3 | 3/3 | 0/3 | 107s | — | — |
| `maas_deepseek` | C3-4 | 3/3 | 3/3 | 0/3 | 101s | — | — |
| `maas_deepseek` | C1-2-3 | 3/3 | 3/3 | 0/3 | 107s | — | — |
| `maas_deepseek` | C1-2-4 | 3/3 | 3/3 | 0/3 | 99s | — | — |
| `maas_deepseek` | C1-3-4 | 3/3 | 3/3 | 0/3 | 116s | — | — |
| `maas_deepseek` | C2-3-4 | 3/3 | 3/3 | 0/3 | 89s | — | — |
| `maas_deepseek` | C1-2-3-4 | 3/3 | 3/3 | 0/3 | 135s | — | — |
| `maas_glm-53-flash` | C0 | 2/3 | 1/3 | 1/3 | 106s | 65s (n=2) | 109s (n=1) |
| `maas_glm-53-flash` | C1 | 3/3 | 2/3 | 1/3 | 100s | 42s (n=3) | 85s (n=1) |
| `maas_glm-53-flash` | C2 | 2/3 | 1/3 | 1/3 | 94s | 49s (n=2) | 55s (n=1) |
| `maas_glm-53-flash` | C3 | 3/3 | 2/3 | 1/3 | 97s | 57s (n=3) | 62s (n=1) |
| `maas_glm-53-flash` | C4 | 3/3 | 1/3 | 1/3 | 177s | 86s (n=3) | 54s (n=1) |
| `maas_glm-53-flash` | C1-2 | 3/3 | 2/3 | 1/3 | 175s | 95s (n=3) | 87s (n=1) |
| `maas_glm-53-flash` | C1-3 | 3/3 | 1/3 | 1/3 | 369s | 243s (n=3) | 70s (n=1) |
| `maas_glm-53-flash` | C1-4 | 3/3 | 1/3 | 1/3 | 186s | 49s (n=3) | 55s (n=1) |
| `maas_glm-53-flash` | C2-3 | 2/3 | 1/3 | 1/3 | 133s | 63s (n=2) | 63s (n=1) |
| `maas_glm-53-flash` | C2-4 | 3/3 | 2/3 | 1/3 | 91s | 50s (n=3) | 87s (n=1) |
| `maas_glm-53-flash` | C3-4 | 3/3 | 2/3 | 1/3 | 112s | 51s (n=3) | 71s (n=1) |
| `maas_glm-53-flash` | C1-2-3 | 3/3 | 1/3 | 1/3 | 203s | 58s (n=3) | 134s (n=1) |
| `maas_glm-53-flash` | C1-2-4 | 3/3 | 1/3 | 1/3 | 120s | 68s (n=3) | 63s (n=1) |
| `maas_glm-53-flash` | C1-3-4 | 3/3 | 1/3 | 1/3 | 102s | 44s (n=3) | 81s (n=1) |
| `maas_glm-53-flash` | C2-3-4 | 3/3 | 2/3 | 2/3 | 357s | 137s (n=3) | 249s (n=2) |
| `maas_glm-53-flash` | C1-2-3-4 | 3/3 | 2/3 | 1/3 | 221s | 155s (n=3) | 94s (n=1) |
| `maas_granite` | C0 | 3/3 | 0/3 | 0/3 | 67s | — | — |
| `maas_granite` | C1 | 3/3 | 3/3 | 0/3 | 630s | — | — |
| `maas_granite` | C2 | 3/3 | 3/3 | 0/3 | 372s | — | — |
| `maas_granite` | C3 | 3/3 | 3/3 | 0/3 | 65s | — | — |
| `maas_granite` | C4 | 3/3 | 3/3 | 0/3 | 249s | — | — |
| `maas_granite` | C1-2 | 3/3 | 3/3 | 0/3 | 53s | — | — |
| `maas_granite` | C1-3 | 3/3 | 3/3 | 0/3 | 67s | — | — |
| `maas_granite` | C1-4 | 2/3 | 2/3 | 0/3 | 137s | — | — |
| `maas_granite` | C2-3 | 3/3 | 2/3 | 0/3 | 43s | — | — |
| `maas_granite` | C2-4 | 3/3 | 2/3 | 0/3 | 35s | — | — |
| `maas_granite` | C3-4 | 3/3 | 1/3 | 0/3 | 427s | — | — |
| `maas_granite` | C1-2-3 | 3/3 | 3/3 | 0/3 | 72s | — | — |
| `maas_granite` | C1-2-4 | 3/3 | 2/3 | 0/3 | 68s | — | — |
| `maas_granite` | C1-3-4 | 3/3 | 3/3 | 0/3 | 150s | — | — |
| `maas_granite` | C2-3-4 | 3/3 | 2/3 | 0/3 | 172s | — | — |
| `maas_granite` | C1-2-3-4 | 3/3 | 2/3 | 0/3 | 68s | — | — |
| `maas_qwen3` | C0 | 3/3 | 0/3 | 0/3 | 97s | 93s (n=1) | — |
| `maas_qwen3` | C1 | 3/3 | 3/3 | 0/3 | 103s | — | — |
| `maas_qwen3` | C2 | 3/3 | 2/3 | 0/3 | 142s | — | — |
| `maas_qwen3` | C3 | 3/3 | 3/3 | 0/3 | 125s | — | — |
| `maas_qwen3` | C4 | 3/3 | 2/3 | 0/3 | 87s | — | — |
| `maas_qwen3` | C1-2 | 3/3 | 3/3 | 0/3 | 118s | — | — |
| `maas_qwen3` | C1-3 | 3/3 | 3/3 | 0/3 | 130s | — | — |
| `maas_qwen3` | C1-4 | 3/3 | 3/3 | 0/3 | 139s | — | — |
| `maas_qwen3` | C2-3 | 3/3 | 3/3 | 0/3 | 129s | — | — |
| `maas_qwen3` | C2-4 | 3/3 | 3/3 | 0/3 | 122s | — | — |
| `maas_qwen3` | C3-4 | 3/3 | 2/3 | 0/3 | 140s | — | — |
| `maas_qwen3` | C1-2-3 | 3/3 | 3/3 | 0/3 | 132s | — | — |
| `maas_qwen3` | C1-2-4 | 3/3 | 3/3 | 0/3 | 151s | — | — |
| `maas_qwen3` | C1-3-4 | 3/3 | 2/3 | 0/3 | 120s | — | — |
| `maas_qwen3` | C2-3-4 | 3/3 | 3/3 | 0/3 | 124s | — | — |
| `maas_qwen3` | C1-2-3-4 | 3/3 | 2/3 | 0/3 | 150s | — | — |
| `maas_qwen36-27b` | C0 | 3/3 | 1/3 | 1/3 | 105s | 105s (n=3) | 128s (n=1) |
| `maas_qwen36-27b` | C1 | 3/3 | 1/3 | 1/3 | 316s | 316s (n=3) | 112s (n=1) |
| `maas_qwen36-27b` | C2 | 2/3 | 1/3 | 1/3 | 299s | 299s (n=2) | 89s (n=1) |
| `maas_qwen36-27b` | C3 | 2/3 | 1/3 | 1/3 | 286s | 286s (n=2) | 81s (n=1) |
| `maas_qwen36-27b` | C4 | 2/3 | 1/3 | 1/3 | 238s | 238s (n=2) | 84s (n=1) |
| `maas_qwen36-27b` | C1-2 | 3/3 | 2/3 | 1/3 | 123s | 123s (n=3) | 136s (n=1) |
| `maas_qwen36-27b` | C1-3 | 2/3 | 1/3 | 1/3 | 88s | 88s (n=2) | 104s (n=1) |
| `maas_qwen36-27b` | C1-4 | 1/3 | 1/3 | 1/3 | 62s | 62s (n=1) | 93s (n=1) |
| `maas_qwen36-27b` | C2-3 | 2/3 | 1/3 | 1/3 | 250s | 250s (n=2) | 90s (n=1) |
| `maas_qwen36-27b` | C2-4 | 2/3 | 1/3 | 1/3 | 187s | 187s (n=2) | 135s (n=1) |
| `maas_qwen36-27b` | C3-4 | 2/3 | 1/3 | 1/3 | 275s | 275s (n=2) | 203s (n=1) |
| `maas_qwen36-27b` | C1-2-3 | 3/3 | 1/3 | 1/3 | 257s | 257s (n=3) | 142s (n=1) |
| `maas_qwen36-27b` | C1-2-4 | 2/3 | 1/3 | 1/3 | 138s | 138s (n=2) | 111s (n=1) |
| `maas_qwen36-27b` | C1-3-4 | 3/3 | 1/3 | 1/3 | 215s | 215s (n=3) | 117s (n=1) |
| `maas_qwen36-27b` | C2-3-4 | 2/3 | 1/3 | 1/3 | 95s | 95s (n=2) | 164s (n=1) |
| `maas_qwen36-27b` | C1-2-3-4 | 1/3 | 1/3 | 1/3 | 79s | 79s (n=1) | 130s (n=1) |
| `maas_qwen38-27b` | C0 | 1/3 | 1/3 | 1/3 | 317s | 317s (n=1) | 380s (n=1) |
| `maas_qwen38-27b` | C1 | 1/3 | 1/3 | 1/3 | 96s | 96s (n=1) | 186s (n=1) |
| `maas_qwen38-27b` | C2 | 2/3 | 1/3 | 1/3 | 399s | 399s (n=2) | 214s (n=1) |
| `maas_qwen38-27b` | C3 | 2/3 | 1/3 | 1/3 | 236s | 236s (n=2) | 249s (n=1) |
| `maas_qwen38-27b` | C4 | 1/3 | 1/3 | 1/3 | 133s | 133s (n=1) | 285s (n=1) |
| `maas_qwen38-27b` | C1-2 | 1/3 | 1/3 | 1/3 | 159s | 159s (n=1) | 357s (n=1) |
| `maas_qwen38-27b` | C1-3 | 1/3 | 1/3 | 1/3 | 115s | 115s (n=1) | 182s (n=1) |
| `maas_qwen38-27b` | C1-4 | 2/3 | 1/3 | 1/3 | 253s | 253s (n=2) | 225s (n=1) |
| `maas_qwen38-27b` | C2-3 | 1/3 | 1/3 | 1/3 | 255s | 255s (n=1) | 339s (n=1) |
| `maas_qwen38-27b` | C2-4 | 1/3 | 1/3 | 1/3 | 102s | 102s (n=1) | 232s (n=1) |
| `maas_qwen38-27b` | C3-4 | 2/3 | 1/3 | 1/3 | 355s | 355s (n=2) | 235s (n=1) |
| `maas_qwen38-27b` | C1-2-3 | 1/3 | 1/3 | 1/3 | 99s | 99s (n=1) | 137s (n=1) |
| `maas_qwen38-27b` | C1-2-4 | 2/3 | 1/3 | 1/3 | 655s | 655s (n=2) | 260s (n=1) |
| `maas_qwen38-27b` | C1-3-4 | 2/3 | 1/3 | 1/3 | 224s | 224s (n=2) | 250s (n=1) |
| `maas_qwen38-27b` | C2-3-4 | 1/3 | 1/3 | 1/3 | 124s | 124s (n=1) | 289s (n=1) |
| `maas_qwen38-27b` | C1-2-3-4 | 3/3 | 2/3 | 2/3 | 803s | 803s (n=3) | 278s (n=1) |

## Agent errors

- `maas_qwen36-27b`: 12/48
- `maas_qwen38-27b`: 6/48
- `maas_glm-53-flash`: 3/48

---

## Recommendations

1. **Default RCA pair:** **maas_deepseek** + **maas_qwen3** for diagnosis; pair with **maas_glm-53-flash** / **maas_qwen38-27b** when execution is required.
2. **Executors this run:** **maas_glm-53-flash**, **maas_qwen38-27b**.
3. **Llama:** Dropped from default lineup after 0% detect (optional agent code remains).
4. **Context:** Prefer high-RCA contexts from the by-context / agent×context tables.
5. **Re-add omitted models** (Nemotron/Kimi) only after MaaS preflight is green.
