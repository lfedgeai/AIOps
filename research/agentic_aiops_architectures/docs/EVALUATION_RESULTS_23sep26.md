# Evaluation Results — 22–23 September 2026 (Option 2, 7-agent MaaS lineup)

**Generated:** 2026-09-23 05:32 AEST (prose corrected 2026-09-23)  
**Design:** Option 2 — scenario **A** × all CONTEXT_C subsets (16) × **7 agents** × 3 faults  
**Runs:** **336/336** completed (`failures=0`)  
**Variant:** `cart` · **Forced RAG:** on for C≠0  
**Tracking:** local MLflow (`http://127.0.0.1:5050`), experiment `agentic_aiops_mttd_mttr`  
**Kickoff:** 2026-09-22 00:02 AEST · **Complete:** 2026-09-23 05:32 AEST  
**Prior run:** [EVALUATION_RESULTS_11Aug26.md](EVALUATION_RESULTS_11Aug26.md) (5 agents, 240 runs)  
**Run notes:** [RUN_NOTES_14Sep26.md](RUN_NOTES_14Sep26.md)  
**Across-runs synthesis (exec gap, context, next tests):** [Learnings_across_runs_report.md](Learnings_across_runs_report.md)

> **Lineup (this run):** `maas_deepseek`, `maas_qwen3`, `maas_qwen38-27b`, `maas_qwen36-27b`, `maas_granite`, `maas_llama31-70b`, `maas_glm-53-flash`.  
> **Omitted:** Nemotron (MaaS 503) and Kimi (MaaS 404).  
> **Comparability:** Shared prompts (`code/agents/prompts.py`) and API preflight. Aug-11 harness JSON for overlapping agents was overwritten by this batch — Aug comparisons below use MLflow + the prior report.

---

## What this report measures

Each of the **336** cells is one harness trial: inject a Kubernetes fault on the OpenTelemetry Demo **cart** service, give the agent a time window to investigate with telemetry/K8s tools (and optional RAG corpora), then score the outcome.

| Axis | Values | Count |
|------|--------|------:|
| Scenario | A (single-service cart fault) | 1 |
| CONTEXT_C | `0` (no corpora) + all non-empty subsets of {1,2,3,4} | 16 |
| Agent | 7 MaaS/RHDP models | 7 |
| Fault | `scale_zero`, `config_corruption`, `kill_pod` | 3 |
| **Total** | 1 × 16 × 7 × 3 | **336** |

**Outcome metrics** (boolean unless noted), logged in harness JSON and MLflow:

| Metric | Meaning |
|--------|---------|
| **Detect** | Agent recognized that something was wrong within the detection window |
| **RCA correct** | Agent identified the right failing component / root cause (cart) |
| **Remediation correct (OR)** | Agent proposed an appropriate fix (judged), whether or not it ran it |
| **Remediation executed** | Agent actually performed a remediation action (e.g. scale/restart) |
| **Recovery verified** | Cluster returned to a healthy cart baseline after the agent’s actions |
| **MTTD** | Seconds from fault injection to detection (median over *detected* runs only) |
| **`agent_error`** | Harness recorded an agent/API failure for that cell (soft-fail path) |

**How to read the tables:** rates are `successes/trials`. Per-agent rows are **48** trials (16 contexts × 3 faults). Per-context rows are **21** trials (7 agents × 3 faults). “Full win” means detect ∧ RCA ∧ exec on the same cell. Aug 11 columns are the previous 5-agent / 240-run Option 2 baseline for trend only — different lineup size, so overall % are not a perfect like-for-like.

---

## Executive summary

| Metric | This run (n=336) | Aug 11 run (n=240) |
|--------|------------------|---------------------|
| Detection | **246/336 (73%)** | 118/240 (49%) |
| RCA correct | **115/336 (34%)** | 92/240 (38%) |
| Remediation correct (OR) | **139/336 (41%)** | 110/240 (46%) |
| Remediation executed | **43/336 (13%)** | 7/240 (3%) |
| Recovery verified | **42/336 (12%)** | 7/240 (3%) |
| MTTD median (detected) | **99s** | 59s |
| `agent_error` | **41/336** | 19/240 |

### Headline takeaways

1. **Detection up, execution still scarce:** detect rose **49% → 73%** vs Aug 11, and remediation executed rose **3% → 13%** (7 → 43 cells). Absolute exec remains low — most models still stop at diagnosis.
2. **RCA vs execute split:** **DeepSeek** leads RCA (**41/48, 85%**) but **executes 0**. **Granite** is next on RCA (**21/48, 44%**) with **0** exec. The models that actually remediate are **qwen36-27b** and **glm-53-flash** (**16/48 exec each**), plus a few Qwen3/35 wins.
3. **GLM is the new reliable detector/executor:** **100% detect**, **38% RCA**, **33% exec**, but slow (**MTTD med 199s**). Together with qwen36 it accounts for **32/43** full wins.
4. **Qwen36 is brittle but complete when healthy:** only **16/48 detect** — and **32/48 `agent_error`** — yet every successful detection also got RCA + exec (**16/16**). High upside, poor reliability this run.
5. **Qwen family this run:** **qwen3** best detect (**100%**) / modest RCA (**25%**) / low exec (**8%**); **qwen38-27b** similar detect, weaker RCA (**15%**), matching exec (**15%**); **qwen36-27b** lowest detect, highest conditional exec.
6. **Llama still unusable:** **maas_llama31-70b** **0/48** detect with **7/48 `agent_error`** — keep out of the default lineup until the endpoint is fixed.

## By agent (48 runs each)

| Agent | Detect | RCA | Exec | MTTD med |
|---|---|---|---|---|
| `maas_deepseek` | 45/48 (94%) | 41/48 (85%) | 0/48 (0%) | 62s |
| `maas_glm-53-flash` | 48/48 (100%) | 18/48 (38%) | 16/48 (33%) | 199s |
| `maas_granite` | 42/48 (88%) | 21/48 (44%) | 0/48 (0%) | 38s |
| `maas_llama31-70b` | 0/48 (0%) | 0/48 (0%) | 0/48 (0%) | — |
| `maas_qwen3` | 48/48 (100%) | 12/48 (25%) | 4/48 (8%) | 96s |
| `maas_qwen38-27b` | 47/48 (98%) | 7/48 (15%) | 7/48 (15%) | 119s |
| `maas_qwen36-27b` | 16/48 (33%) | 16/48 (33%) | 16/48 (33%) | 74s |

## Model family comparison

### Qwen family (`qwen3-14b` vs `qwen38-27b` vs `qwen36-27b`)

| Model | Detect | RCA | Exec | MTTD med |
|---|---|---|---|---|
| qwen3 (RHDP) | 48/48 (100%) | 12/48 (25%) | 4/48 (8%) | 96s |
| qwen38-27b (MaaS) | 47/48 (98%) | 7/48 (15%) | 7/48 (15%) | 119s |
| qwen36-27b (MaaS) | 16/48 (33%) | 16/48 (33%) | 16/48 (33%) | 74s |
| qwen3 (Aug 11 MLflow) | 26/48 (54%) | 16/48 (33%) | 7/48 (15%) | 140s |

Nemotron was **not** in this matrix (endpoint unavailable). Prior Aug nano / cascade comparisons live in [EVALUATION_RESULTS_11Aug26.md](EVALUATION_RESULTS_11Aug26.md) / [EVALUATION_RESULTS_12Aug26.md](EVALUATION_RESULTS_12Aug26.md).

## Top performing combinations

### This run — best cells (detect ∧ RCA ∧ exec)

**Full wins:** **43/336** (qwen36-27b 16, glm-53-flash 16, qwen38-27b 7, qwen3 4)

Fastest 20 full-win cells (all `scale_zero` in this slice):

| Context | Agent | Fault | MTTD |
|---------|-------|-------|------|
| C1-2 | `maas_glm-53-flash` | scale_zero | 51.759985s |
| C4 | `maas_qwen36-27b` | scale_zero | 52.748909s |
| C1 | `maas_qwen36-27b` | scale_zero | 53.688386s |
| C1 | `maas_glm-53-flash` | scale_zero | 57.798764s |
| C1-3-4 | `maas_qwen36-27b` | scale_zero | 59.517124s |
| C1-4 | `maas_qwen36-27b` | scale_zero | 62.39339s |
| C4 | `maas_glm-53-flash` | scale_zero | 62.394259s |
| C3 | `maas_glm-53-flash` | scale_zero | 62.432623s |
| C2-3 | `maas_qwen36-27b` | scale_zero | 63.423709s |
| C1-2-3 | `maas_glm-53-flash` | scale_zero | 65.960534s |
| C1-2 | `maas_qwen36-27b` | scale_zero | 67.011358s |
| C2 | `maas_glm-53-flash` | scale_zero | 67.794386s |
| C1-3 | `maas_glm-53-flash` | scale_zero | 68.010131s |
| C3-4 | `maas_qwen36-27b` | scale_zero | 70.351697s |
| C2-3-4 | `maas_glm-53-flash` | scale_zero | 71.123175s |
| C2-4 | `maas_glm-53-flash` | scale_zero | 71.539466s |
| C3 | `maas_qwen36-27b` | scale_zero | 73.322375s |
| C2-4 | `maas_qwen36-27b` | scale_zero | 74.516464s |
| C1-2-4 | `maas_qwen36-27b` | scale_zero | 76.126741s |
| C1-4 | `maas_glm-53-flash` | scale_zero | 76.479639s |

### Top CONTEXT_C × agent (RCA rate, 3 faults per cell)

| Rank | Context | Agent | Detect | RCA | Exec |
|------|---------|-------|--------|-----|------|
| 1 | C4 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 2 | C3 | `maas_granite` | 3/3 | 3/3 | 0/3 |
| 3 | C3 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 4 | C2-4 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 5 | C2-3-4 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 6 | C2-3 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 7 | C2 | `maas_granite` | 3/3 | 3/3 | 0/3 |
| 8 | C2 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 9 | C1-4 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 10 | C1-3 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 11 | C1-2-4 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 12 | C1-2-3-4 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 13 | C1-2-3 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 14 | C1-2 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 15 | C1 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |

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
| C1 | qwen3 | 1/3 RCA | 0/3 exec | 1/3 RCA, 1/3 exec |
| C1 | granite | 0/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C2 | deepseek | 3/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C2 | qwen3 | 1/3 RCA | 1/3 exec | 1/3 RCA, 0/3 exec |
| C2 | granite | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C3 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C3 | qwen3 | 0/3 RCA | 0/3 exec | 3/3 RCA, 1/3 exec |
| C3 | granite | 3/3 RCA | 0/3 exec | 2/3 RCA, 0/3 exec |
| C4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C4 | qwen3 | 2/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C4 | granite | 2/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1-4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1-4 | qwen3 | 1/3 RCA | 1/3 exec | 2/3 RCA, 0/3 exec |
| C1-4 | granite | 1/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-4 | deepseek | 3/3 RCA | 0/3 exec | 2/3 RCA, 0/3 exec |
| C2-4 | qwen3 | 1/3 RCA | 1/3 exec | 2/3 RCA, 1/3 exec |
| C2-4 | granite | 2/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1-2-3-4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1-2-3-4 | qwen3 | 0/3 RCA | 0/3 exec | 1/3 RCA, 1/3 exec |
| C1-2-3-4 | granite | 2/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-3 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-3 | qwen3 | 1/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C2-3 | granite | 2/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-3-4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-3-4 | qwen3 | 1/3 RCA | 0/3 exec | 0/3 RCA, 0/3 exec |
| C2-3-4 | granite | 0/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |

## MLflow validation

- Harness JSON runs (this batch): **336**
- MLflow runs in window: **336**
- Matched by `run_id`: **336**
- Metric mismatches: **0**

(Outcome metrics agree. `agent_error` is present on harness JSON for 41 runs but is not logged as an MLflow metric/param.)

## By context (21 runs each = 7 agents × 3 faults)

| Context | Detect | RCA | Exec | MTTD med |
|---|---|---|---|---|
| C0 | 16/21 (76%) | 4/21 (19%) | 3/21 (14%) | 137s |
| C1 | 16/21 (76%) | 6/21 (29%) | 2/21 (10%) | 66s |
| C1-2 | 16/21 (76%) | 7/21 (33%) | 2/21 (10%) | 78s |
| C1-2-3 | 16/21 (76%) | 8/21 (38%) | 3/21 (14%) | 113s |
| C1-2-3-4 | 15/21 (71%) | 7/21 (33%) | 2/21 (10%) | 137s |
| C1-2-4 | 16/21 (76%) | 8/21 (38%) | 3/21 (14%) | 120s |
| C1-3 | 16/21 (76%) | 7/21 (33%) | 3/21 (14%) | 106s |
| C1-3-4 | 12/21 (57%) | 3/21 (14%) | 2/21 (10%) | 118s |
| C1-4 | 15/21 (71%) | 7/21 (33%) | 3/21 (14%) | 102s |
| C2 | 16/21 (76%) | 10/21 (48%) | 3/21 (14%) | 95s |
| C2-3 | 16/21 (76%) | 9/21 (43%) | 3/21 (14%) | 86s |
| C2-3-4 | 13/21 (62%) | 7/21 (33%) | 3/21 (14%) | 123s |
| C2-4 | 16/21 (76%) | 8/21 (38%) | 3/21 (14%) | 82s |
| C3 | 16/21 (76%) | 9/21 (43%) | 3/21 (14%) | 61s |
| C3-4 | 15/21 (71%) | 6/21 (29%) | 3/21 (14%) | 98s |
| C4 | 16/21 (76%) | 9/21 (43%) | 2/21 (10%) | 63s |

## Agent errors

- `maas_qwen36-27b`: 32/48
- `maas_llama31-70b`: 7/48
- `maas_qwen38-27b`: 1/48
- `maas_granite`: 1/48

---

## Recommendations

1. **Default RCA pair:** DeepSeek + Granite for diagnosis; pair with **glm-53-flash** or **qwen36-27b** when remediation execution is required.
2. **Executors this run:** **glm-53-flash** (reliable detect + 33% exec) and **qwen36-27b** (high conditional exec, but 32/48 agent errors — investigate API stability before relying on it).
3. **Llama:** Remove or fix **maas_llama31-70b** before the next matrix (0/48 detect).
4. **Context:** Highest RCA density on **C2** / **C2-3** / **C3** / **C4**; DeepSeek still peaks on many single- and multi-corpus cells for RCA-only.
5. **Nemotron / Kimi:** Re-add only after MaaS endpoints pass preflight; they were correctly excluded from this batch.

For why agents don’t execute, how CONTEXT_C is delivered, chunking vs corpus hygiene, and a prioritized next-test list, see **[Learnings_across_runs_report.md](Learnings_across_runs_report.md)**. Shared skill packs: [`code/agents/skills/`](../code/agents/skills/).
