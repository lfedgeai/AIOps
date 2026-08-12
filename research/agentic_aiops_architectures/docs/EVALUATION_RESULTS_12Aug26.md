# Evaluation Results — 11–12 August 2026 (Option 2, expanded MaaS lineup)

**Generated:** 2026-08-12 21:48 AEST  
**Design:** Option 2 — scenario **A** × all CONTEXT_C subsets (16) × **8 agents** × 3 faults  
**Runs:** **384/384** completed (`failures=0`)  
**Variant:** `cart` · **Forced RAG:** on for C≠0  
**Tracking:** local MLflow (`http://127.0.0.1:5050`)  
**Prior run:** [EVALUATION_RESULTS_11Aug26.md](EVALUATION_RESULTS_11Aug26.md) (5 agents, 240 runs)  
**Run notes:** [RUN_NOTES_11Aug26.md](RUN_NOTES_11Aug26.md)  

> **Comparability:** Shared prompts (`code/agents/prompts.py`), preflight gate, and Nemotron cascade replace nano. Aug-11 JSON for DeepSeek/Qwen3/Granite/Llama was **overwritten** by this run — Aug comparisons for those agents use MLflow + this report's prior doc.

---

## Executive summary

| Metric | This run (n=384) | Aug 11 run (n=240) |
|--------|------------------|---------------------|
| Detection | **199/384 (52%)** | 118/240 (49%) |
| RCA correct | **170/384 (44%)** | 92/240 (38%) |
| Remediation correct (OR) | **213/384 (55%)** | 110/240 (46%) |
| Remediation executed | **82/384 (21%)** | 7/240 (3%) |
| Recovery verified | **72/384 (19%)** | 7/240 (3%) |
| MTTD median (detected) | **54s** | 59s |
| `agent_error` | **18/384** | 19/240 |

### Headline takeaways

1. **Execution gap closed:** remediation executed rose from **3% → 21%** (82/384) after unified prompts and new MaaS models — primarily **qwen36-27b** (28/48) and **nemotron-cascade-2** (18/48).
2. **RCA leaders unchanged, executor split:** **DeepSeek** (88% RCA) and **Granite** (67% RCA) still dominate diagnosis but execute **0%** remediations; **qwen36-27b** is the only model strong on both RCA and exec.
3. **Nemotron fixed:** cascade-2 replaces invalid nano-3 (0/48 Aug) with **38% detect / 42% RCA / 38% exec** — first usable Nemotron in the matrix.
4. **Qwen36 > Qwen3 > Qwen35:** MaaS **qwen36-27b** (62% RCA, 58% exec) beats RHDP **qwen3** (29% RCA, 12% exec) and **qwen35-9b** (29% each) on this cart workload.
5. **Llama still broken:** **maas_llama31-70b** 0/48 detect with 12 `agent_error` — connection/API failures persist.
6. **Best CONTEXT_C cells:** **C2**, **C2-3**, **C2-3-4** with **qwen36-27b** hit **3/3 RCA + exec** across all three faults.

## By agent (48 runs each)

| Agent | Detect | RCA | Exec | MTTD med |
|---|---|---|---|---|
| `maas_deepseek` | 46/48 (96%) | 42/48 (88%) | 0/48 (0%) | 58s |
| `maas_granite` | 48/48 (100%) | 32/48 (67%) | 0/48 (0%) | 33s |
| `maas_kimi-k2-7` | 17/48 (35%) | 18/48 (38%) | 16/48 (33%) | 27s |
| `maas_llama31-70b` | 0/48 (0%) | 0/48 (0%) | 0/48 (0%) | — |
| `maas_qwen3` | 26/48 (54%) | 14/48 (29%) | 6/48 (12%) | 157s |
| `maas_qwen35-9b` | 14/48 (29%) | 14/48 (29%) | 14/48 (29%) | 45s |
| `maas_qwen36-27b` | 30/48 (62%) | 30/48 (62%) | 28/48 (58%) | 82s |
| `nemotron-cascade-2` | 18/48 (38%) | 20/48 (42%) | 18/48 (38%) | 49s |

## Model family comparison

### Nemotron (nano Aug → cascade now)

| Lineup | Detect | RCA | Exec | Notes |
|---|---|---|---|---|
| `nemotron-nano-3` (Aug) | 0/48 (0%) | 0/48 (0%) | 0/48 (0%) | 48/48 API HTML — invalid |
| `nemotron-cascade-2` (now) | 18/48 (38%) | 20/48 (42%) | 18/48 (38%) | Healthy MaaS |

### Qwen family (`qwen3-14b` vs `qwen35-9b` vs `qwen36-27b`)

| Model | Detect | RCA | Exec | MTTD med |
|---|---|---|---|---|
| qwen3 (RHDP) | 26/48 (54%) | 14/48 (29%) | 6/48 (12%) | 157s |
| qwen35-9b (MaaS) | 14/48 (29%) | 14/48 (29%) | 14/48 (29%) | 45s |
| qwen36-27b (MaaS) | 30/48 (62%) | 30/48 (62%) | 28/48 (58%) | 82s |
| qwen3 (Aug 11 MLflow) | 26/48 (54%) | 16/48 (33%) | 7/48 (15%) | 140s |

## Top performing combinations

### This run — best cells (detect+RCA+exec, n=1 each)

**Full wins (detect ∧ RCA ∧ exec):** 82/384

| Context | Agent | Fault | MTTD |
|---------|-------|-------|------|
| C0 | `maas_kimi-k2-7` | scale_zero | 22.77264s |
| C4 | `maas_kimi-k2-7` | scale_zero | 24.013863s |
| C2 | `maas_kimi-k2-7` | scale_zero | 25.054823s |
| C3 | `maas_kimi-k2-7` | scale_zero | 25.218668s |
| C2-3 | `maas_kimi-k2-7` | scale_zero | 25.376428s |
| C2-4 | `maas_kimi-k2-7` | scale_zero | 25.433651s |
| C1-3 | `maas_kimi-k2-7` | scale_zero | 25.620678s |
| C1 | `maas_kimi-k2-7` | scale_zero | 26.138153s |
| C1-4 | `maas_kimi-k2-7` | scale_zero | 26.843142s |
| C1-2-3 | `maas_kimi-k2-7` | scale_zero | 27.057783s |
| C1-2-4 | `maas_kimi-k2-7` | scale_zero | 27.090786s |
| C2-3-4 | `maas_kimi-k2-7` | scale_zero | 27.117996s |
| C1-2 | `maas_kimi-k2-7` | scale_zero | 27.497253s |
| C1-3-4 | `maas_kimi-k2-7` | scale_zero | 27.948723s |
| C3 | `nemotron-cascade-2` | scale_zero | 28.314672s |
| C1-2-3-4 | `maas_kimi-k2-7` | scale_zero | 29.381355s |
| C2-3 | `nemotron-cascade-2` | scale_zero | 32.429856s |
| C0 | `nemotron-cascade-2` | scale_zero | 33.050102s |
| C4 | `nemotron-cascade-2` | scale_zero | 34.022942s |
| C1-2 | `nemotron-cascade-2` | scale_zero | 34.192795s |

### Top CONTEXT_C × agent (RCA rate, min 3 cells = 3 faults)

| Rank | Context | Agent | Detect | RCA | Exec |
|------|---------|-------|--------|-----|------|
| 1 | C2-3-4 | `maas_qwen36-27b` | 3/3 | 3/3 | 3/3 |
| 2 | C2-3 | `maas_qwen36-27b` | 3/3 | 3/3 | 3/3 |
| 3 | C2 | `maas_qwen36-27b` | 3/3 | 3/3 | 3/3 |
| 4 | C1-3 | `maas_qwen36-27b` | 3/3 | 3/3 | 2/3 |
| 5 | C4 | `maas_granite` | 3/3 | 3/3 | 0/3 |
| 6 | C4 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 7 | C3-4 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 8 | C3 | `maas_granite` | 3/3 | 3/3 | 0/3 |
| 9 | C3 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 10 | C2-4 | `maas_granite` | 3/3 | 3/3 | 0/3 |
| 11 | C2-4 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 12 | C2-3-4 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 13 | C2-3 | `maas_granite` | 3/3 | 3/3 | 0/3 |
| 14 | C2 | `maas_granite` | 3/3 | 3/3 | 0/3 |
| 15 | C2 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |

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
| C1 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1 | qwen3 | 1/3 RCA | 1/3 exec | 1/3 RCA, 1/3 exec |
| C1 | granite | 1/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C2 | deepseek | 3/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C2 | qwen3 | 1/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C2 | granite | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C3 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C3 | qwen3 | 1/3 RCA | 0/3 exec | 3/3 RCA, 1/3 exec |
| C3 | granite | 3/3 RCA | 0/3 exec | 2/3 RCA, 0/3 exec |
| C4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C4 | qwen3 | 1/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C4 | granite | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1-4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1-4 | qwen3 | 2/3 RCA | 1/3 exec | 2/3 RCA, 0/3 exec |
| C1-4 | granite | 2/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-4 | deepseek | 3/3 RCA | 0/3 exec | 2/3 RCA, 0/3 exec |
| C2-4 | qwen3 | 1/3 RCA | 1/3 exec | 2/3 RCA, 1/3 exec |
| C2-4 | granite | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1-2-3-4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1-2-3-4 | qwen3 | 0/3 RCA | 0/3 exec | 1/3 RCA, 1/3 exec |
| C1-2-3-4 | granite | 2/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-3 | deepseek | 2/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-3 | qwen3 | 2/3 RCA | 1/3 exec | 1/3 RCA, 0/3 exec |
| C2-3 | granite | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-3-4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-3-4 | qwen3 | 1/3 RCA | 0/3 exec | 0/3 RCA, 0/3 exec |
| C2-3-4 | granite | 1/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |

## MLflow validation

- Harness JSON runs (this batch): **384**
- MLflow runs in window: **384**
- Matched by `run_id`: **384**
- Metric mismatches: **0**

## By context (24 runs each)

| Context | Detect | RCA | Exec | MTTD med |
|---|---|---|---|---|
| C0 | 11/24 (46%) | 7/24 (29%) | 7/24 (29%) | 55s |
| C1 | 14/24 (58%) | 12/24 (50%) | 7/24 (29%) | 42s |
| C1-2 | 13/24 (54%) | 10/24 (42%) | 5/24 (21%) | 52s |
| C1-2-3 | 11/24 (46%) | 10/24 (42%) | 4/24 (17%) | 63s |
| C1-2-3-4 | 10/24 (42%) | 8/24 (33%) | 3/24 (12%) | 82s |
| C1-2-4 | 9/24 (38%) | 8/24 (33%) | 3/24 (12%) | 52s |
| C1-3 | 14/24 (58%) | 13/24 (54%) | 5/24 (21%) | 46s |
| C1-3-4 | 14/24 (58%) | 9/24 (38%) | 7/24 (29%) | 112s |
| C1-4 | 14/24 (58%) | 13/24 (54%) | 6/24 (25%) | 58s |
| C2 | 12/24 (50%) | 12/24 (50%) | 5/24 (21%) | 33s |
| C2-3 | 14/24 (58%) | 14/24 (58%) | 7/24 (29%) | 49s |
| C2-3-4 | 13/24 (54%) | 11/24 (46%) | 6/24 (25%) | 70s |
| C2-4 | 12/24 (50%) | 11/24 (46%) | 5/24 (21%) | 47s |
| C3 | 13/24 (54%) | 12/24 (50%) | 4/24 (17%) | 35s |
| C3-4 | 12/24 (50%) | 9/24 (38%) | 4/24 (17%) | 86s |
| C4 | 13/24 (54%) | 11/24 (46%) | 4/24 (17%) | 36s |

## Agent errors

- `maas_llama31-70b`: 12/48
- `maas_qwen36-27b`: 4/48
- `maas_qwen3`: 2/48

---

## Recommendations

1. **Default RCA pair:** DeepSeek + Granite for diagnosis; add **qwen36-27b** when remediation execution is required.
2. **Executor:** **qwen36-27b** and **nemotron-cascade-2** are the primary executors; Kimi is fast but less consistent on RCA.
3. **Nemotron:** Keep **cascade-2** in lineup — first valid Nemotron result (38% exec).
4. **Llama:** Remove or fix **maas_llama31-70b** endpoint before next matrix (0/48).
5. **Context:** Prioritize **C2** and **C2-3-4** bundles for qwen36; DeepSeek/Granite still peak on **C3** and **C1-4** for RCA-only.
