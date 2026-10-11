# Evaluation Results — Option 2 matrix (7 agents)

**Generated:** 2026-10-10 20:26 AEST  
**Design:** Option 2 — scenario **A** × all CONTEXT_C subsets (16) × **7 agents** × 3 faults  
**Runs:** **336/336** completed  
**Variant:** `cart` · **Forced RAG:** on for C≠0  
**Tracking:** local MLflow (`http://127.0.0.1:5050`), experiment `agentic_aiops_mttd_mttr`  
**Primary baseline:** [EVALUATION_RESULTS_27sep26.md](EVALUATION_RESULTS_27sep26.md) (Option 2, skills + no-CHANGELOG)  
**Run notes:** [RUN_NOTES_07Oct26.md](RUN_NOTES_07Oct26.md)  

> **Lineup (this run):** `maas_deepseek`, `maas_glm-53-flash`, `maas_granite`, `maas_llama31-70b`, `maas_qwen3`, `maas_qwen36-27b`, `maas_qwen38-27b`.  
> **Comparability:** Shared prompts (`code/agents/prompts.py`) and API preflight. Primary Δ table uses Sep 25–27; older Aug-11 (different design size) appears only in legacy combo sections.  
> **Nemotron:** omitted (prelude-maas timeout / HTTP 503 at kickoff preflight).

---

## Changes before this run (vs Sep 25–27)

| Change | Detail |
|--------|--------|
| **Execute gate** | `AGENT_EXECUTE_GATE=1` — block final JSON when `detected=true` until a rem write (or cannot-remediate) |
| **Remediation flatten** | `flatten_suggested_remediations` (fix nested list-of-lists TypeError) |
| **Qwen slot** | `maas_qwen38-27b` on maas-rhdp (replaces dead prelude `qwen35-9b`) |
| **Carried forward** | `AGENT_SKILLS=1`, `INDEX_VERSION=no-changelog-v1`, timeouts 900/1200 |

**Post-run (does not affect these 336 cells):** generic prompts/skills; playbooks removed; llama dropped from default lineup. See [RUN_NOTES_07Oct26.md](RUN_NOTES_07Oct26.md).

---

## What this report measures

Each of the **336** cells is one harness trial: inject a Kubernetes fault on a target workload (scenario A / cart variant), give the agent a time window with telemetry/K8s tools (and optional RAG corpora), then score the outcome.

| Axis | Values |
|------|--------|
| Scenario | A (single-service fault) |
| CONTEXT_C | 16 labels (`0` + subsets of {1,2,3,4}) |
| Agent | 7 models |
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

**How to read rates:** `successes/trials`. Per-agent rows are **48** trials. Per-context rows are **21** trials (7 agents × 3 faults) when the matrix is complete. “Full win” = detect ∧ RCA ∧ exec on the same cell. MTTR medians use only cells where the field is present (n shown in model×context tables).

---

## Executive summary

| Metric | This run (n=336) | Sep 25–27 (n=335) |
|--------|------------------|-------------------------------------|
| Detection | **234/336 (70%)** | 266/335 (79%) |
| RCA correct | **191/336 (57%)** | 196/335 (59%) |
| Remediation correct (OR) | **205/336 (61%)** | 207/335 (62%) |
| Remediation executed | **105/336 (31%)** | 83/335 (25%) |
| Recovery verified | **87/336 (26%)** | 73/335 (22%) |
| MTTD median (detected) | **126s** | 100s |
| MTTR median | **112s** | — |
| MTTR verified median | **137s** | — |
| `agent_error` | **20/336** | 21/335 |

_Primary baseline = [EVALUATION_RESULTS_27sep26.md](EVALUATION_RESULTS_27sep26.md) (same Option 2 design). Older Aug-11 (5-agent / 240-cell) figures remain in family/combo sections where noted._

### Headline takeaways

1. **Detection vs execution:** detect **70%** (234/336); remediation executed **31%** (105/336) — most models still stop at diagnosis.
2. **RCA vs execute split:** **DeepSeek** RCA **45/48** / exec **0/48**; **Granite** RCA **32/48** / exec **3/48**.
3. **Primary executors this run:** **maas_qwen36-27b** 40/48, **maas_glm-53-flash** 27/48, **maas_qwen3** 19/48, **maas_qwen38-27b** 16/48.
4. **qwen36-27b:** detect/RCA/exec **43/48** with **5/48** `agent_error` (when healthy, detect usually co-occurs with RCA+exec).
5. **Llama:** **maas_llama31-70b** detect **0/48**, `agent_error` **5/48** — keep out of default lineup until fixed.
6. **glm-53-flash:** detect **48/48**, RCA **35/48**, exec **27/48** (MTTD med 134s).
7. **Nemotron / Kimi:** not in this batch (omit from headlines until preflight-green).
8. **Execute gate:** raised overall exec vs Sep 25–27 (**25% → 31%**) but **did not** convert DeepSeek (still 0% exec); see below.

## Execute gate (diagnose → remediate) — what was configured and what happened

### Mechanism (`AGENT_EXECUTE_GATE=1`)

Shared in `code/agents/execute_gate.py` for every agent:

1. If the model emits final JSON with `detected=true` **before** calling a rem write tool (`scale_deployment` / `restart_deployment` / `delete_pod` / `patch_deployment`), the harness **rejects** that finish.
2. It injects an `[execute-gate]` nudge (up to **3** times) requiring a write call, or an explicit “cannot remediate” statement.
3. Scoring still counts **`remediation_executed` only on a successful write-tool call** — prose like “Restarted cartservice…” does **not** count.

### Outcomes this run (48 cells / agent)

| Agent | Cells with gate nudge | Hit nudge=3 | After nudge → **exec** | After nudge → final `detected=false` | Exec **without** any gate nudge |
|-------|----------------------:|------------:|-----------------------:|-------------------------------------:|--------------------------------:|
| `maas_deepseek` | **48** | 46 | **0** | 0 | 0 |
| `maas_granite` | **39** | 28 | **3** | **3** | 0 |
| `maas_qwen3` | **36** | 0 | **17** | 0 | 2 |
| `maas_qwen36-27b` | 0 | 0 | — | — | **40** |
| `maas_glm-53-flash` | 0 | 0 | — | — | **27** |
| `maas_qwen38-27b` | 0 | 0 | — | — | **16** |
| `maas_llama31-70b` | 0 | 0 | — | — | 0 (never detected) |

No cell used the “cannot remediate” escape hatch in text.

### What agents actually did in response

| Agent | Response pattern |
|-------|------------------|
| **DeepSeek** | Detect + strong RCA, then tries to finish with JSON. Gate blocks repeatedly (usually 3×). Final JSON still **`detected=true`**, `suggested_remediations` written in past tense as if a fix ran (e.g. “Restarted the cartservice deployment…”), but **zero write tools** → `remediation_executed=false`. |
| **Granite** | Same diagnose→suggest pattern on most cells; after 3 nudges usually still **prose-only** remediations (“Restart cartservice deployment”). **3/39** nudged cells eventually called a write tool. **3** nudged cells ended with **`detected=false`** (abandoned the fault claim under gate pressure). |
| **Qwen3** | Often needs a nudge; **17/36** then called a real write (commonly `scale_deployment` on scale_zero) and verified. Remaining nudged cells stayed suggestion-only or weak RCA. |
| **Qwen36 / GLM / Qwen38** | When they detect, they typically **call write tools before final JSON**, so the gate never fires. Example (qwen36, C0/`scale_zero`): voluntary `scale_deployment` cart → 1, then verify Running — no `[execute-gate]` lines in the transcript. |

**Takeaway:** the gate is a **harness policy**, not model training. It increases exec for models that will act when pushed (Qwen3) and for natural executors it is a no-op. It does **not** force non-executors (DeepSeek) to call tools; they can still emit misleading “I restarted…” prose after max nudges.

## By agent (48 runs each)

| Agent | Detect | RCA | Exec | MTTD med | MTTR med | MTTR verified med |
|---|---|---|---|---|---|---|
| `maas_deepseek` | 48/48 (100%) | 45/48 (94%) | 0/48 (0%) | 101s | 269s (n=2) | — |
| `maas_glm-53-flash` | 48/48 (100%) | 35/48 (73%) | 27/48 (56%) | 134s | 88s (n=48) | 79s (n=26) |
| `maas_granite` | 37/48 (77%) | 32/48 (67%) | 3/48 (6%) | 174s | 58s (n=8) | 142s (n=1) |
| `maas_llama31-70b` | 0/48 (0%) | 0/48 (0%) | 0/48 (0%) | — | — | — |
| `maas_qwen3` | 42/48 (88%) | 21/48 (44%) | 19/48 (40%) | 142s | 120s (n=42) | 141s (n=13) |
| `maas_qwen36-27b` | 43/48 (90%) | 41/48 (85%) | 40/48 (83%) | 181s | 181s (n=43) | 202s (n=31) |
| `maas_qwen38-27b` | 16/48 (33%) | 17/48 (35%) | 16/48 (33%) | 99s | 99s (n=16) | 143s (n=16) |

## Model family comparison

### Nemotron

| Lineup | Detect | RCA | Exec | Notes |
|---|---|---|---|---|
| `nemotron-nano-3` (Aug) | 38/96 (40%) | 37/96 (39%) | 34/96 (35%) | prior batch |
| `nemotron-cascade-2` (this run) | n/a | n/a | n/a | not in this batch |

### Qwen family (`qwen3-14b` vs `qwen38-27b` vs `qwen36-27b`)

| Model | Detect | RCA | Exec | MTTD med | MTTR med | MTTR verified med |
|---|---|---|---|---|---|---|
| qwen3 (RHDP) | 42/48 (88%) | 21/48 (44%) | 19/48 (40%) | 142s | 120s (n=42) | 141s (n=13) |
| qwen38-27b (MaaS) | 16/48 (33%) | 17/48 (35%) | 16/48 (33%) | 99s | 99s (n=16) | 143s (n=16) |
| qwen36-27b (MaaS) | 43/48 (90%) | 41/48 (85%) | 40/48 (83%) | 181s | 181s (n=43) | 202s (n=31) |
| qwen3 (Aug 11 MLflow) | 26/48 (54%) | 16/48 (33%) | 7/48 (15%) | 140s | — | — |

## Top performing combinations

### This run — best cells (detect ∧ RCA ∧ exec)

**Full wins:** **105/336** (maas_qwen36-27b 40, maas_glm-53-flash 27, maas_qwen3 19, maas_qwen38-27b 16, maas_granite 3)

| Context | Agent | Fault | MTTD | MTTR | MTTR verified |
|---------|-------|-------|------|------|---------------|
| C4 | `maas_granite` | config_corruption | 34s | 29s | — |
| C2 | `maas_qwen36-27b` | scale_zero | 43s | 43s | 88s |
| C3-4 | `maas_glm-53-flash` | scale_zero | 46s | 31s | 46s |
| C2 | `maas_glm-53-flash` | scale_zero | 47s | 30s | 47s |
| C3 | `maas_glm-53-flash` | scale_zero | 47s | 27s | 47s |
| C4 | `maas_glm-53-flash` | scale_zero | 49s | 30s | 49s |
| C4 | `maas_qwen36-27b` | scale_zero | 50s | 50s | 80s |
| C2-3 | `maas_qwen36-27b` | scale_zero | 50s | 50s | 74s |
| C2-3-4 | `maas_qwen36-27b` | scale_zero | 52s | 52s | 96s |
| C1-3 | `maas_qwen36-27b` | scale_zero | 52s | 52s | 84s |
| C3 | `maas_qwen38-27b` | scale_zero | 53s | 53s | 91s |
| C1-3-4 | `maas_qwen36-27b` | scale_zero | 54s | 54s | 73s |
| C1-3 | `maas_glm-53-flash` | scale_zero | 56s | 36s | 56s |
| C3 | `maas_qwen36-27b` | scale_zero | 57s | 57s | 87s |
| C2-3 | `maas_glm-53-flash` | scale_zero | 57s | 34s | 57s |
| C1-2-3 | `maas_glm-53-flash` | scale_zero | 61s | 41s | 61s |
| C1-2 | `maas_glm-53-flash` | scale_zero | 63s | 45s | 63s |
| C1 | `maas_qwen36-27b` | scale_zero | 64s | 64s | 118s |
| C1-4 | `maas_glm-53-flash` | scale_zero | 67s | 44s | 67s |
| C1-2 | `maas_qwen36-27b` | scale_zero | 68s | 68s | 116s |

### Top CONTEXT_C × agent (RCA rate, 3 faults per cell)

| Rank | Context | Agent | Detect | RCA | Exec |
|------|---------|-------|--------|-----|------|
| 1 | C4 | `maas_qwen36-27b` | 3/3 | 3/3 | 3/3 |
| 2 | C2-4 | `maas_qwen36-27b` | 3/3 | 3/3 | 3/3 |
| 3 | C2-4 | `maas_glm-53-flash` | 3/3 | 3/3 | 3/3 |
| 4 | C2-3-4 | `maas_qwen36-27b` | 3/3 | 3/3 | 3/3 |
| 5 | C2-3 | `maas_qwen36-27b` | 3/3 | 3/3 | 3/3 |
| 6 | C2-3 | `maas_qwen3` | 3/3 | 3/3 | 3/3 |
| 7 | C1-4 | `maas_qwen36-27b` | 3/3 | 3/3 | 3/3 |
| 8 | C1-3-4 | `maas_qwen36-27b` | 3/3 | 3/3 | 3/3 |
| 9 | C1-2-3-4 | `maas_qwen36-27b` | 3/3 | 3/3 | 3/3 |
| 10 | C1-2-3 | `maas_qwen36-27b` | 3/3 | 3/3 | 3/3 |
| 11 | C0 | `maas_qwen36-27b` | 3/3 | 3/3 | 3/3 |
| 12 | C3-4 | `maas_glm-53-flash` | 3/3 | 3/3 | 2/3 |
| 13 | C3 | `maas_glm-53-flash` | 3/3 | 3/3 | 2/3 |
| 14 | C2-4 | `maas_granite` | 3/3 | 3/3 | 2/3 |
| 15 | C2-3 | `maas_glm-53-flash` | 3/3 | 3/3 | 2/3 |

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
| C0 | deepseek | 1/3 RCA | 0/3 exec | 2/3 RCA, 0/3 exec |
| C0 | qwen3 | 1/3 RCA | 0/3 exec | 0/3 RCA, 0/3 exec |
| C0 | granite | 3/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C1 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1 | qwen3 | 0/3 RCA | 0/3 exec | 1/3 RCA, 1/3 exec |
| C1 | granite | 1/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C2 | deepseek | 3/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C2 | qwen3 | 0/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C2 | granite | 2/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C3 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C3 | qwen3 | 1/3 RCA | 1/3 exec | 3/3 RCA, 1/3 exec |
| C3 | granite | 1/3 RCA | 0/3 exec | 2/3 RCA, 0/3 exec |
| C4 | deepseek | 2/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C4 | qwen3 | 2/3 RCA | 1/3 exec | 1/3 RCA, 0/3 exec |
| C4 | granite | 3/3 RCA | 1/3 exec | 3/3 RCA, 0/3 exec |
| C1-4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1-4 | qwen3 | 1/3 RCA | 1/3 exec | 2/3 RCA, 0/3 exec |
| C1-4 | granite | 2/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-4 | deepseek | 3/3 RCA | 0/3 exec | 2/3 RCA, 0/3 exec |
| C2-4 | qwen3 | 0/3 RCA | 0/3 exec | 2/3 RCA, 1/3 exec |
| C2-4 | granite | 3/3 RCA | 2/3 exec | 3/3 RCA, 0/3 exec |
| C1-2-3-4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1-2-3-4 | qwen3 | 2/3 RCA | 2/3 exec | 1/3 RCA, 1/3 exec |
| C1-2-3-4 | granite | 1/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-3 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-3 | qwen3 | 3/3 RCA | 3/3 exec | 1/3 RCA, 0/3 exec |
| C2-3 | granite | 2/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-3-4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-3-4 | qwen3 | 1/3 RCA | 1/3 exec | 0/3 RCA, 0/3 exec |
| C2-3-4 | granite | 2/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |

## MLflow validation

- Harness JSON runs (this batch): **336**
- MLflow runs in window: **336**
- Matched by `run_id`: **336**
- Metric mismatches: **0**

## By context (21 runs each = 7 agents × 3 faults)

| Context | Detect | RCA | Exec | MTTD med | MTTR med | MTTR verified med |
|---|---|---|---|---|---|---|
| C0 | 16/21 (76%) | 10/21 (48%) | 5/21 (24%) | 127s | 118s (n=12) | 193s (n=4) |
| C1 | 15/21 (71%) | 10/21 (48%) | 5/21 (24%) | 125s | 106s (n=9) | 169s (n=5) |
| C2 | 15/21 (71%) | 11/21 (52%) | 5/21 (24%) | 120s | 87s (n=10) | 117s (n=5) |
| C3 | 14/21 (67%) | 10/21 (48%) | 5/21 (24%) | 103s | 93s (n=11) | 91s (n=5) |
| C4 | 15/21 (71%) | 13/21 (62%) | 8/21 (38%) | 144s | 106s (n=10) | 146s (n=6) |
| C1-2 | 14/21 (67%) | 12/21 (57%) | 7/21 (33%) | 94s | 96s (n=9) | 113s (n=6) |
| C1-3 | 14/21 (67%) | 12/21 (57%) | 5/21 (24%) | 99s | 79s (n=9) | 97s (n=5) |
| C1-4 | 15/21 (71%) | 13/21 (62%) | 7/21 (33%) | 136s | 159s (n=10) | 171s (n=6) |
| C2-3 | 15/21 (71%) | 15/21 (71%) | 9/21 (43%) | 111s | 111s (n=11) | 137s (n=7) |
| C2-4 | 15/21 (71%) | 13/21 (62%) | 9/21 (43%) | 142s | 124s (n=11) | 167s (n=6) |
| C3-4 | 13/21 (62%) | 11/21 (52%) | 7/21 (33%) | 141s | 142s (n=9) | 141s (n=7) |
| C1-2-3 | 15/21 (71%) | 12/21 (57%) | 7/21 (33%) | 152s | 127s (n=10) | 128s (n=5) |
| C1-2-4 | 16/21 (76%) | 12/21 (57%) | 6/21 (29%) | 129s | 105s (n=10) | 133s (n=5) |
| C1-3-4 | 14/21 (67%) | 13/21 (62%) | 7/21 (33%) | 130s | 141s (n=9) | 146s (n=5) |
| C2-3-4 | 14/21 (67%) | 13/21 (62%) | 6/21 (29%) | 141s | 157s (n=9) | 103s (n=4) |
| C1-2-3-4 | 14/21 (67%) | 11/21 (52%) | 7/21 (33%) | 134s | 133s (n=10) | 168s (n=6) |

## By agent × context (3 faults each)

One row per (agent, CONTEXT_C). Rates are over the 3 faults; MTTR / MTTR verified medians use cells where the field is present.

| Agent | Context | Detect | RCA | Exec | MTTD med | MTTR med | MTTR verified med |
|---|---|---|---|---|---|---|---|
| `maas_deepseek` | C0 | 3/3 | 1/3 | 0/3 | 112s | 269s (n=2) | — |
| `maas_deepseek` | C1 | 3/3 | 3/3 | 0/3 | 106s | — | — |
| `maas_deepseek` | C2 | 3/3 | 3/3 | 0/3 | 172s | — | — |
| `maas_deepseek` | C3 | 3/3 | 3/3 | 0/3 | 80s | — | — |
| `maas_deepseek` | C4 | 3/3 | 2/3 | 0/3 | 83s | — | — |
| `maas_deepseek` | C1-2 | 3/3 | 3/3 | 0/3 | 86s | — | — |
| `maas_deepseek` | C1-3 | 3/3 | 3/3 | 0/3 | 88s | — | — |
| `maas_deepseek` | C1-4 | 3/3 | 3/3 | 0/3 | 101s | — | — |
| `maas_deepseek` | C2-3 | 3/3 | 3/3 | 0/3 | 98s | — | — |
| `maas_deepseek` | C2-4 | 3/3 | 3/3 | 0/3 | 82s | — | — |
| `maas_deepseek` | C3-4 | 3/3 | 3/3 | 0/3 | 86s | — | — |
| `maas_deepseek` | C1-2-3 | 3/3 | 3/3 | 0/3 | 97s | — | — |
| `maas_deepseek` | C1-2-4 | 3/3 | 3/3 | 0/3 | 136s | — | — |
| `maas_deepseek` | C1-3-4 | 3/3 | 3/3 | 0/3 | 105s | — | — |
| `maas_deepseek` | C2-3-4 | 3/3 | 3/3 | 0/3 | 103s | — | — |
| `maas_deepseek` | C1-2-3-4 | 3/3 | 3/3 | 0/3 | 117s | — | — |
| `maas_glm-53-flash` | C0 | 3/3 | 1/3 | 1/3 | 169s | 116s (n=3) | 78s (n=1) |
| `maas_glm-53-flash` | C1 | 3/3 | 2/3 | 2/3 | 169s | 147s (n=3) | 128s (n=2) |
| `maas_glm-53-flash` | C2 | 3/3 | 3/3 | 2/3 | 117s | 100s (n=3) | 82s (n=2) |
| `maas_glm-53-flash` | C3 | 3/3 | 3/3 | 2/3 | 141s | 98s (n=3) | 94s (n=2) |
| `maas_glm-53-flash` | C4 | 3/3 | 2/3 | 2/3 | 146s | 88s (n=3) | 172s (n=2) |
| `maas_glm-53-flash` | C1-2 | 3/3 | 2/3 | 2/3 | 105s | 86s (n=3) | 84s (n=2) |
| `maas_glm-53-flash` | C1-3 | 3/3 | 3/3 | 2/3 | 96s | 79s (n=3) | 76s (n=2) |
| `maas_glm-53-flash` | C1-4 | 3/3 | 3/3 | 2/3 | 136s | 79s (n=3) | 120s (n=2) |
| `maas_glm-53-flash` | C2-3 | 3/3 | 3/3 | 2/3 | 124s | 104s (n=3) | 91s (n=2) |
| `maas_glm-53-flash` | C2-4 | 3/3 | 3/3 | 3/3 | 119s | 92s (n=3) | 96s (n=2) |
| `maas_glm-53-flash` | C3-4 | 3/3 | 3/3 | 2/3 | 157s | 142s (n=3) | 102s (n=2) |
| `maas_glm-53-flash` | C1-2-3 | 3/3 | 1/3 | 1/3 | 152s | 88s (n=3) | 61s (n=1) |
| `maas_glm-53-flash` | C1-2-4 | 3/3 | 1/3 | 1/3 | 109s | 77s (n=3) | 103s (n=1) |
| `maas_glm-53-flash` | C1-3-4 | 3/3 | 2/3 | 1/3 | 226s | 188s (n=3) | 74s (n=1) |
| `maas_glm-53-flash` | C2-3-4 | 3/3 | 2/3 | 1/3 | 250s | 53s (n=3) | 78s (n=1) |
| `maas_glm-53-flash` | C1-2-3-4 | 3/3 | 1/3 | 1/3 | 131s | 78s (n=3) | 80s (n=1) |
| `maas_granite` | C0 | 3/3 | 3/3 | 0/3 | 50s | — | — |
| `maas_granite` | C1 | 3/3 | 1/3 | 0/3 | 125s | — | — |
| `maas_granite` | C2 | 3/3 | 2/3 | 0/3 | 350s | 52s (n=1) | — |
| `maas_granite` | C3 | 1/3 | 1/3 | 0/3 | 632s | 569s (n=1) | — |
| `maas_granite` | C4 | 3/3 | 3/3 | 1/3 | 129s | 29s (n=1) | — |
| `maas_granite` | C1-2 | 2/3 | 2/3 | 0/3 | 402s | — | — |
| `maas_granite` | C1-3 | 3/3 | 3/3 | 0/3 | 568s | 49s (n=1) | — |
| `maas_granite` | C1-4 | 3/3 | 2/3 | 0/3 | 118s | 666s (n=1) | — |
| `maas_granite` | C2-3 | 2/3 | 2/3 | 0/3 | 92s | 65s (n=1) | — |
| `maas_granite` | C2-4 | 3/3 | 3/3 | 2/3 | 199s | 91s (n=2) | 142s (n=1) |
| `maas_granite` | C3-4 | 1/3 | 0/3 | 0/3 | 456s | — | — |
| `maas_granite` | C1-2-3 | 2/3 | 2/3 | 0/3 | 509s | — | — |
| `maas_granite` | C1-2-4 | 3/3 | 3/3 | 0/3 | 174s | — | — |
| `maas_granite` | C1-3-4 | 2/3 | 2/3 | 0/3 | 243s | — | — |
| `maas_granite` | C2-3-4 | 2/3 | 2/3 | 0/3 | 224s | — | — |
| `maas_granite` | C1-2-3-4 | 1/3 | 1/3 | 0/3 | 224s | — | — |
| `maas_llama31-70b` | C0 | 0/3 | 0/3 | 0/3 | — | — | — |
| `maas_llama31-70b` | C1 | 0/3 | 0/3 | 0/3 | — | — | — |
| `maas_llama31-70b` | C2 | 0/3 | 0/3 | 0/3 | — | — | — |
| `maas_llama31-70b` | C3 | 0/3 | 0/3 | 0/3 | — | — | — |
| `maas_llama31-70b` | C4 | 0/3 | 0/3 | 0/3 | — | — | — |
| `maas_llama31-70b` | C1-2 | 0/3 | 0/3 | 0/3 | — | — | — |
| `maas_llama31-70b` | C1-3 | 0/3 | 0/3 | 0/3 | — | — | — |
| `maas_llama31-70b` | C1-4 | 0/3 | 0/3 | 0/3 | — | — | — |
| `maas_llama31-70b` | C2-3 | 0/3 | 0/3 | 0/3 | — | — | — |
| `maas_llama31-70b` | C2-4 | 0/3 | 0/3 | 0/3 | — | — | — |
| `maas_llama31-70b` | C3-4 | 0/3 | 0/3 | 0/3 | — | — | — |
| `maas_llama31-70b` | C1-2-3 | 0/3 | 0/3 | 0/3 | — | — | — |
| `maas_llama31-70b` | C1-2-4 | 0/3 | 0/3 | 0/3 | — | — | — |
| `maas_llama31-70b` | C1-3-4 | 0/3 | 0/3 | 0/3 | — | — | — |
| `maas_llama31-70b` | C2-3-4 | 0/3 | 0/3 | 0/3 | — | — | — |
| `maas_llama31-70b` | C1-2-3-4 | 0/3 | 0/3 | 0/3 | — | — | — |
| `maas_qwen3` | C0 | 3/3 | 1/3 | 0/3 | 96s | 85s (n=3) | — |
| `maas_qwen3` | C1 | 3/3 | 0/3 | 0/3 | 139s | 106s (n=3) | — |
| `maas_qwen3` | C2 | 3/3 | 0/3 | 0/3 | 94s | 85s (n=3) | — |
| `maas_qwen3` | C3 | 3/3 | 1/3 | 1/3 | 96s | 92s (n=3) | 110s (n=1) |
| `maas_qwen3` | C4 | 2/3 | 2/3 | 1/3 | 146s | 120s (n=2) | 149s (n=1) |
| `maas_qwen3` | C1-2 | 3/3 | 2/3 | 2/3 | 131s | 107s (n=3) | 113s (n=1) |
| `maas_qwen3` | C1-3 | 2/3 | 0/3 | 0/3 | 184s | 172s (n=2) | — |
| `maas_qwen3` | C1-4 | 2/3 | 1/3 | 1/3 | 135s | 126s (n=2) | 169s (n=1) |
| `maas_qwen3` | C2-3 | 3/3 | 3/3 | 3/3 | 175s | 158s (n=3) | 201s (n=2) |
| `maas_qwen3` | C2-4 | 2/3 | 0/3 | 0/3 | 160s | 131s (n=2) | — |
| `maas_qwen3` | C3-4 | 3/3 | 2/3 | 2/3 | 141s | 114s (n=3) | 131s (n=2) |
| `maas_qwen3` | C1-2-3 | 3/3 | 2/3 | 2/3 | 180s | 149s (n=3) | 111s (n=1) |
| `maas_qwen3` | C1-2-4 | 3/3 | 2/3 | 2/3 | 132s | 110s (n=3) | 125s (n=1) |
| `maas_qwen3` | C1-3-4 | 2/3 | 2/3 | 2/3 | 176s | 164s (n=2) | 164s (n=1) |
| `maas_qwen3` | C2-3-4 | 2/3 | 1/3 | 1/3 | 197s | 180s (n=2) | — |
| `maas_qwen3` | C1-2-3-4 | 3/3 | 2/3 | 2/3 | 142s | 130s (n=3) | 156s (n=2) |
| `maas_qwen36-27b` | C0 | 3/3 | 3/3 | 3/3 | 254s | 254s (n=3) | 193s (n=2) |
| `maas_qwen36-27b` | C1 | 2/3 | 3/3 | 2/3 | 116s | 116s (n=2) | 160s (n=2) |
| `maas_qwen36-27b` | C2 | 2/3 | 2/3 | 2/3 | 122s | 122s (n=2) | 170s (n=2) |
| `maas_qwen36-27b` | C3 | 3/3 | 1/3 | 1/3 | 417s | 417s (n=3) | 87s (n=1) |
| `maas_qwen36-27b` | C4 | 3/3 | 3/3 | 3/3 | 187s | 187s (n=3) | 198s (n=2) |
| `maas_qwen36-27b` | C1-2 | 2/3 | 2/3 | 2/3 | 124s | 124s (n=2) | 170s (n=2) |
| `maas_qwen36-27b` | C1-3 | 2/3 | 2/3 | 2/3 | 255s | 255s (n=2) | 288s (n=2) |
| `maas_qwen36-27b` | C1-4 | 3/3 | 3/3 | 3/3 | 350s | 350s (n=3) | 321s (n=2) |
| `maas_qwen36-27b` | C2-3 | 3/3 | 3/3 | 3/3 | 232s | 232s (n=3) | 177s (n=2) |
| `maas_qwen36-27b` | C2-4 | 3/3 | 3/3 | 3/3 | 449s | 449s (n=3) | 346s (n=2) |
| `maas_qwen36-27b` | C3-4 | 2/3 | 2/3 | 2/3 | 590s | 590s (n=2) | 642s (n=2) |
| `maas_qwen36-27b` | C1-2-3 | 3/3 | 3/3 | 3/3 | 214s | 214s (n=3) | 186s (n=2) |
| `maas_qwen36-27b` | C1-2-4 | 3/3 | 2/3 | 2/3 | 126s | 126s (n=3) | 154s (n=2) |
| `maas_qwen36-27b` | C1-3-4 | 3/3 | 3/3 | 3/3 | 118s | 118s (n=3) | 114s (n=2) |
| `maas_qwen36-27b` | C2-3-4 | 3/3 | 3/3 | 3/3 | 165s | 165s (n=3) | 149s (n=2) |
| `maas_qwen36-27b` | C1-2-3-4 | 3/3 | 3/3 | 3/3 | 171s | 171s (n=3) | 449s (n=2) |
| `maas_qwen38-27b` | C0 | 1/3 | 1/3 | 1/3 | 913s | 913s (n=1) | 956s (n=1) |
| `maas_qwen38-27b` | C1 | 1/3 | 1/3 | 1/3 | 102s | 102s (n=1) | 199s (n=1) |
| `maas_qwen38-27b` | C2 | 1/3 | 1/3 | 1/3 | 90s | 90s (n=1) | 141s (n=1) |
| `maas_qwen38-27b` | C3 | 1/3 | 1/3 | 1/3 | 53s | 53s (n=1) | 91s (n=1) |
| `maas_qwen38-27b` | C4 | 1/3 | 1/3 | 1/3 | 92s | 92s (n=1) | 144s (n=1) |
| `maas_qwen38-27b` | C1-2 | 1/3 | 1/3 | 1/3 | 68s | 68s (n=1) | 113s (n=1) |
| `maas_qwen38-27b` | C1-3 | 1/3 | 1/3 | 1/3 | 101s | 101s (n=1) | 167s (n=1) |
| `maas_qwen38-27b` | C1-4 | 1/3 | 1/3 | 1/3 | 97s | 97s (n=1) | 142s (n=1) |
| `maas_qwen38-27b` | C2-3 | 1/3 | 1/3 | 1/3 | 111s | 111s (n=1) | 165s (n=1) |
| `maas_qwen38-27b` | C2-4 | 1/3 | 1/3 | 1/3 | 115s | 115s (n=1) | 191s (n=1) |
| `maas_qwen38-27b` | C3-4 | 1/3 | 1/3 | 1/3 | 74s | 74s (n=1) | 128s (n=1) |
| `maas_qwen38-27b` | C1-2-3 | 1/3 | 1/3 | 1/3 | 97s | 97s (n=1) | 132s (n=1) |
| `maas_qwen38-27b` | C1-2-4 | 1/3 | 1/3 | 1/3 | 101s | 101s (n=1) | 133s (n=1) |
| `maas_qwen38-27b` | C1-3-4 | 1/3 | 1/3 | 1/3 | 104s | 104s (n=1) | 146s (n=1) |
| `maas_qwen38-27b` | C2-3-4 | 1/3 | 2/3 | 1/3 | 73s | 73s (n=1) | 111s (n=1) |
| `maas_qwen38-27b` | C1-2-3-4 | 1/3 | 1/3 | 1/3 | 112s | 112s (n=1) | 164s (n=1) |

## Agent errors

- `maas_qwen3`: 6/48
- `maas_llama31-70b`: 5/48
- `maas_qwen36-27b`: 5/48
- `maas_qwen38-27b`: 3/48
- `maas_granite`: 1/48

---

## Recommendations

1. **Default RCA pair:** **maas_deepseek** + **maas_qwen36-27b** for diagnosis; pair with **maas_qwen36-27b** / **maas_glm-53-flash** when execution is required.
2. **Executors this run:** **maas_qwen36-27b**, **maas_glm-53-flash**.
3. **Llama:** Dropped from default lineup after 0% detect (optional agent code remains).
4. **Context:** Prefer high-RCA contexts from the by-context / agent×context tables.
5. **Re-add omitted models** (Nemotron/Kimi) only after MaaS preflight is green.
