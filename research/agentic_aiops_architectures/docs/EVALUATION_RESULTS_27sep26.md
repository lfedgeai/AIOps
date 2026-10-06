# Evaluation Results — 25–27 September 2026 (Option 2, skills + no-CHANGELOG)

**Generated:** 2026-09-27 00:05 AEST · **Audited:** 2026-09-28 (MLflow ↔ harness)  
**Design:** Option 2 — scenario **A** × all CONTEXT_C subsets (16) × **7 agents** × 3 faults = **336** cells  
**Runs:** **335/336** scored harness JSON (`failures=1`)  
**Variant:** `cart` · **Forced RAG:** on for C≠0  
**Tracking:** local MLflow (`http://127.0.0.1:5050`), experiment `agentic_aiops_mttd_mttr`  
**Kickoff:** 2026-09-25 14:06:25 AEST · **Complete:** 2026-09-27 00:05:07 AEST  
**Primary baseline:** [EVALUATION_RESULTS_23sep26.md](EVALUATION_RESULTS_23sep26.md) (336 runs, same lineup)  
**Compare:** [COMPARE_27sep26_vs_23sep26.md](COMPARE_27sep26_vs_23sep26.md) · archive `out/archive_sep22_option2/`  
**Run notes:** [RUN_NOTES_25Sep26.md](RUN_NOTES_25Sep26.md)  
**Learnings:** [Learnings_across_runs_report.md](Learnings_across_runs_report.md)

> **Lineup:** `maas_deepseek`, `maas_qwen3`, `maas_qwen35-9b`, `maas_qwen36-27b`, `maas_granite`, `maas_llama31-70b`, `maas_glm-53-flash`.  
> **Experiment deltas vs Sep 22–23:** shared AIOps skills in system prompt (`AGENT_SKILLS=1`); CHANGELOG removed from C2 docs corpus (`INDEX_VERSION=no-changelog-v1`); `AGENT_TIMEOUT=900` / detection window 1200s.  
> **Missing cell:** `C2-3` × `kill_pod` × `maas_qwen35-9b` — harness crashed in `score_remediation` (`TypeError`: nested `suggested_remediations` list-of-lists). An incomplete MLflow run exists for that cell (no outcome metrics).

---

## What this report measures

Each scored cell is one harness trial: inject a Kubernetes fault on the OpenTelemetry Demo **cart** service, give the agent a time window with telemetry/K8s tools (and optional RAG corpora), then score the outcome. Design size is **336**; rates below use **n=335** scored JSON (one crash).

| Axis | Values | Count |
|------|--------|------:|
| Scenario | A (single-service cart fault) | 1 |
| CONTEXT_C | `0` + all non-empty subsets of {1,2,3,4} | 16 |
| Agent | 7 MaaS models | 7 |
| Fault | `scale_zero`, `config_corruption`, `kill_pod` | 3 |
| **Design total** | 1 × 16 × 7 × 3 | **336** |
| **Scored** | harness JSON written | **335** |

| Metric | Meaning |
|--------|---------|
| **Detect** | Agent recognized a fault within the detection window |
| **RCA correct** | Agent identified the right failing component / root cause |
| **Remediation correct (OR)** | Appropriate fix proposed (judged), whether or not executed |
| **Remediation executed** | Agent performed a remediation action |
| **Recovery verified** | Cart baseline healthy after the agent’s actions |
| **MTTD** | Seconds from fault to detection (median over *detected* runs) |
| **`agent_error`** | Harness recorded an agent/API failure for that cell |

**How to read rates:** `successes/trials`. Per-agent row counts vary. Per-context rows are **21** trials (7 agents × 3 faults) when the matrix is complete. “Full win” = detect ∧ RCA ∧ exec on the same cell.

---

## Executive summary

Rates verified against harness JSON **and** MLflow (335 matched cells, **0** metric mismatches). Primary Δ vs Sep 22–23 same-lineup baseline:

| Metric | This run (n=335) | Sep 22–23 (n=336) | Δ |
|--------|------------------|-------------------|---|
| Detection | **266/335 (79%)** | 246/336 (73%) | +6 pp |
| RCA correct | **196/335 (59%)** | 115/336 (34%) | **+25 pp** |
| Remediation correct (OR) | **207/335 (62%)** | — | — |
| Remediation executed | **83/335 (25%)** | 43/336 (13%) | **+12 pp** |
| Recovery verified | **73/335 (22%)** | 42/336 (12%) | +10 pp |
| MTTD median (detected) | **100s** | — | — |
| `agent_error` | **21/335** | 41/336 | **−20** |

### Headline takeaways

1. **Skills + corpus hygiene correlated with large RCA/exec lifts** vs Sep 22–23 (same agents/faults/contexts). Not a pure model bake-off — prompts and C2 index changed.
2. **Detection vs execution:** detect **79%**; exec **25%**. DeepSeek/Granite still **0% exec** given detect∧RCA (0/45 and 0/46).
3. **When executors get RCA right, they usually act:** Qwen36 **37/38 (97%)**, GLM **30/33 (91%)**, Qwen35 **15/18 (83%)** conditional exec.
4. **Primary executors:** **qwen36-27b** 37/48, **glm-53-flash** 30/48, **qwen35-9b** 15/47; qwen3 still near-zero exec (1/48).
5. **qwen36 reliability:** `agent_error` **32→8**/48 after timeout raise; detect/RCA/exec all **~79%** when the cell completes.
6. **Llama** still dead for practical use (detect **1/48**).
7. **One harness crash** (nested remediations) — fix `score_remediation` to flatten/coerce before `" ".join`.

## By agent

| Agent | Detect | RCA | Exec | MTTD med |
|---|---|---|---|---|
| `maas_deepseek` | 48/48 (100%) | 45/48 (94%) | 0/48 (0%) | 72s |
| `maas_glm-53-flash` | 47/48 (98%) | 34/48 (71%) | 30/48 (62%) | 167s |
| `maas_granite` | 48/48 (100%) | 46/48 (96%) | 0/48 (0%) | 40s |
| `maas_llama31-70b` | 1/48 (2%) | 0/48 (0%) | 0/48 (0%) | 597s |
| `maas_qwen3` | 47/48 (98%) | 15/48 (31%) | 1/48 (2%) | 97s |
| `maas_qwen35-9b` | 37/47 (79%) | 18/47 (38%) | 15/47 (32%) | 140s |
| `maas_qwen36-27b` | 38/48 (79%) | 38/48 (79%) | 37/48 (77%) | 192s |

## Model family comparison

### Nemotron

| Lineup | Detect | RCA | Exec | Notes |
|---|---|---|---|---|
| `nemotron-nano-3` (Aug) | 38/96 (40%) | 37/96 (39%) | 34/96 (35%) | prior batch |
| `nemotron-cascade-2` (this run) | n/a | n/a | n/a | not in this batch |

### Qwen family (`qwen3-14b` vs `qwen35-9b` vs `qwen36-27b`)

| Model | Detect | RCA | Exec | MTTD med |
|---|---|---|---|---|
| qwen3 (RHDP) | 47/48 (98%) | 15/48 (31%) | 1/48 (2%) | 97s |
| qwen35-9b (MaaS) | 37/47 (79%) | 18/47 (38%) | 15/47 (32%) | 140s |
| qwen36-27b (MaaS) | 38/48 (79%) | 38/48 (79%) | 37/48 (77%) | 192s |
| qwen3 (Aug 11 MLflow) | 26/48 (54%) | 16/48 (33%) | 7/48 (15%) | 140s |

## Top performing combinations

### This run — best cells (detect ∧ RCA ∧ exec)

**Full wins:** **83/335** (maas_qwen36-27b 37, maas_glm-53-flash 30, maas_qwen35-9b 15, maas_qwen3 1)

| Context | Agent | Fault | MTTD |
|---------|-------|-------|------|
| C1-2 | `maas_qwen36-27b` | scale_zero | 46.398438s |
| C3 | `maas_qwen36-27b` | scale_zero | 47.645657s |
| C2-3 | `maas_qwen36-27b` | scale_zero | 52.53313s |
| C1 | `maas_qwen36-27b` | scale_zero | 53.506704s |
| C4 | `maas_glm-53-flash` | scale_zero | 55.777086s |
| C1-2-3 | `maas_qwen36-27b` | scale_zero | 60.560754s |
| C3-4 | `maas_glm-53-flash` | scale_zero | 60.704856s |
| C1-2 | `maas_glm-53-flash` | scale_zero | 61.943384s |
| C1 | `maas_glm-53-flash` | scale_zero | 62.55594s |
| C1-3-4 | `maas_glm-53-flash` | scale_zero | 64.450989s |
| C0 | `maas_glm-53-flash` | scale_zero | 64.66229s |
| C2-4 | `maas_glm-53-flash` | scale_zero | 66.395513s |
| C1-4 | `maas_qwen36-27b` | scale_zero | 66.800522s |
| C2 | `maas_glm-53-flash` | scale_zero | 67.859621s |
| C2-3-4 | `maas_glm-53-flash` | scale_zero | 68.593033s |
| C1-2-4 | `maas_glm-53-flash` | scale_zero | 70.473727s |
| C2 | `maas_qwen36-27b` | scale_zero | 70.722304s |
| C3 | `maas_glm-53-flash` | scale_zero | 72.132162s |
| C1-3 | `maas_glm-53-flash` | scale_zero | 73.434655s |
| C4 | `maas_qwen3` | scale_zero | 75.584883s |

### Top CONTEXT_C × agent (RCA rate, 3 faults per cell)

| Rank | Context | Agent | Detect | RCA | Exec |
|------|---------|-------|--------|-----|------|
| 1 | C4 | `maas_qwen36-27b` | 3/3 | 3/3 | 3/3 |
| 2 | C3-4 | `maas_glm-53-flash` | 3/3 | 3/3 | 3/3 |
| 3 | C2-4 | `maas_qwen36-27b` | 3/3 | 3/3 | 3/3 |
| 4 | C2-4 | `maas_glm-53-flash` | 3/3 | 3/3 | 3/3 |
| 5 | C1-2-3 | `maas_qwen36-27b` | 3/3 | 3/3 | 3/3 |
| 6 | C1-2 | `maas_qwen36-27b` | 3/3 | 3/3 | 3/3 |
| 7 | C1 | `maas_qwen36-27b` | 3/3 | 3/3 | 3/3 |
| 8 | C0 | `maas_glm-53-flash` | 3/3 | 3/3 | 3/3 |
| 9 | C2-3 | `maas_qwen36-27b` | 3/3 | 3/3 | 2/3 |
| 10 | C1-3-4 | `maas_glm-53-flash` | 3/3 | 3/3 | 2/3 |
| 11 | C1-2 | `maas_glm-53-flash` | 2/3 | 3/3 | 2/3 |
| 12 | C4 | `maas_granite` | 3/3 | 3/3 | 0/3 |
| 13 | C4 | `maas_deepseek` | 3/3 | 3/3 | 0/3 |
| 14 | C3-4 | `maas_granite` | 3/3 | 3/3 | 0/3 |
| 15 | C3 | `maas_granite` | 3/3 | 3/3 | 0/3 |

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
| C0 | qwen3 | 0/3 RCA | 0/3 exec | 0/3 RCA, 0/3 exec |
| C0 | granite | 3/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C1 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1 | qwen3 | 0/3 RCA | 0/3 exec | 1/3 RCA, 1/3 exec |
| C1 | granite | 3/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C2 | deepseek | 3/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C2 | qwen3 | 1/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C2 | granite | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C3 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C3 | qwen3 | 1/3 RCA | 0/3 exec | 3/3 RCA, 1/3 exec |
| C3 | granite | 3/3 RCA | 0/3 exec | 2/3 RCA, 0/3 exec |
| C4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C4 | qwen3 | 2/3 RCA | 1/3 exec | 1/3 RCA, 0/3 exec |
| C4 | granite | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1-4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1-4 | qwen3 | 3/3 RCA | 0/3 exec | 2/3 RCA, 0/3 exec |
| C1-4 | granite | 2/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-4 | deepseek | 3/3 RCA | 0/3 exec | 2/3 RCA, 0/3 exec |
| C2-4 | qwen3 | 1/3 RCA | 0/3 exec | 2/3 RCA, 1/3 exec |
| C2-4 | granite | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1-2-3-4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1-2-3-4 | qwen3 | 0/3 RCA | 0/3 exec | 1/3 RCA, 1/3 exec |
| C1-2-3-4 | granite | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-3 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-3 | qwen3 | 1/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C2-3 | granite | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-3-4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2-3-4 | qwen3 | 1/3 RCA | 0/3 exec | 0/3 RCA, 0/3 exec |
| C2-3-4 | granite | 3/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |

## MLflow validation (audited 2026-09-28)

| Check | Result |
|-------|--------|
| Harness JSON (mtime ≥ kickoff, 7-agent lineup) | **335** |
| MLflow runs in kickoff→complete window | **336** |
| Matched by `run_id` with outcome metrics | **335** |
| Mismatches on `detected` / `rca_correct` / `remediation_correct` / `remediation_executed` / `recovery_verified` | **0** |
| Orphan MLflow run | `20260925104620_maas_qwen35-9b_scenarioa_kill_pod_cart` — **no** outcome metrics; params show nested `remediation_steps` `[[...]]` (same TypeError). This is the missing C2-3 `kill_pod` qwen35 cell. |

**Verdict:** Report headline rates are **aligned with MLflow** for all scored cells. The +1 MLflow count is the crashed cell, not a silent metric drift.

## By context (21 runs each = 7 agents × 3 faults)

| Context | Detect | RCA | Exec | MTTD med |
|---|---|---|---|---|
| C0 | 18/21 (86%) | 9/21 (43%) | 5/21 (24%) | 154s |
| C1 | 17/21 (81%) | 12/21 (57%) | 5/21 (24%) | 72s |
| C2 | 16/21 (76%) | 13/21 (62%) | 4/21 (19%) | 86s |
| C3 | 17/21 (81%) | 12/21 (57%) | 5/21 (24%) | 68s |
| C4 | 17/21 (81%) | 14/21 (67%) | 7/21 (33%) | 76s |
| C1-2 | 16/21 (76%) | 15/21 (71%) | 6/21 (29%) | 62s |
| C1-3 | 16/21 (76%) | 12/21 (57%) | 5/21 (24%) | 91s |
| C1-4 | 17/21 (81%) | 13/21 (62%) | 5/21 (24%) | 104s |
| C2-3 | 16/20 (80%) | 13/20 (65%) | 4/20 (20%) | 92s |
| C2-4 | 17/21 (81%) | 14/21 (67%) | 7/21 (33%) | 104s |
| C3-4 | 16/21 (76%) | 12/21 (57%) | 6/21 (29%) | 107s |
| C1-2-3 | 16/21 (76%) | 12/21 (57%) | 5/21 (24%) | 98s |
| C1-2-4 | 17/21 (81%) | 11/21 (52%) | 5/21 (24%) | 96s |
| C1-3-4 | 16/21 (76%) | 11/21 (52%) | 5/21 (24%) | 119s |
| C2-3-4 | 17/21 (81%) | 11/21 (52%) | 4/21 (19%) | 119s |
| C1-2-3-4 | 17/21 (81%) | 12/21 (57%) | 5/21 (24%) | 116s |

## Agent errors

- `maas_qwen35-9b`: 10/47
- `maas_qwen36-27b`: 8/48
- `maas_llama31-70b`: 1/48
- `maas_glm-53-flash`: 1/48
- `maas_qwen3`: 1/48

---

## Recommendations

1. **Default RCA pair:** **granite** + **deepseek** for diagnosis; pair with **qwen36-27b** / **glm-53-flash** when execution is required (skills did **not** make DeepSeek/Granite act).
2. **Executors this run:** **qwen36-27b**, **glm-53-flash** (best conditional exec after RCA).
3. **Llama:** Drop **maas_llama31-70b** from the default lineup (1/48 detect).
4. **Harness hardening:** Flatten/coerce `suggested_remediations` in `score_remediation` so nested lists cannot abort a cell.
5. **Context:** Highest RCA density on **C1-2** (71%), **C4** / **C2-4** (67%); C0 RCA rose to 43% (was 19% in Sep22) with skills even without RAG.
6. **Next experiment:** isolate skills vs no-CHANGELOG vs timeout (one axis at a time) — this batch confounds all three.
7. See [Learnings_across_runs_report.md](Learnings_across_runs_report.md) for cross-run synthesis.
