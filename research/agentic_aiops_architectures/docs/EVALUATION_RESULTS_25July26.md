# Evaluation Results — 24–25 July 2026 (Option 1 re-run, fixed corpora)

**Date:** 2026-07-24 22:58 → 2026-07-25 03:04 AEST  
**Design:** Option 1 — scenario **A** × contexts **C0–C4** × 5 agents × 3 faults  
**Runs:** **75/75** completed (`failures=0`)  
**Variant:** `cart` (OpenTelemetry Demo)  
**Forced RAG:** on for C1–C4 (`FORCE_RAG=1`); C0 has no corpora  
**Corpus gate:** empty C1–C4 indexes abort before fault injection (`corpus_empty`) — **0 aborts** this run  
**Tracking:** local MLflow only (`http://127.0.0.1:5050`)  
**Outputs:** `out/context_C{0-4}_sa_{fault}_{agent}.json` (mtime ≥ 2026-07-24 22:50)  
**Log:** `out/c_matrix_scenario_a_c1-c4_run.log`

This re-run replaces the 22 July matrix, which was invalid for C3/C4 (empty indexes + fatal RAG abort → 0/30 detection). Fixes landed in `code/tools/rag_context.py` (C3/C4 globs + architecture filter) and early corpus validation in `run_harness.py` / `context_engineering.py`. Verified chunk counts before start: **C1=1784, C2=130, C3=129, C4=166**.

---

## Executive summary

With non-empty C3/C4 corpora, **RAG no longer looks harmful by construction**. Overall **52/75 (69%)** detection and **41/75 (55%)** RCA. Context ranking flips vs the broken 22 Jul run:

| Context | Detect | RCA | Remediation | Forced RAG used |
|---------|--------|-----|-------------|-----------------|
| **C0** (none) | 11/15 (73%) | 6/15 (40%) | 6/15 (40%) | 0/15 |
| **C1** (source) | **12/15 (80%)** | 9/15 (60%) | **10/15 (67%)** | 15/15 |
| **C2** (docs) | 10/15 (67%) | 9/15 (60%) | 8/15 (53%) | 15/15 |
| **C3** (arch) | 11/15 (73%) | **10/15 (67%)** | 8/15 (53%) | 15/15 |
| **C4** (deps) | 8/15 (53%) | 7/15 (47%) | 6/15 (40%) | 15/15 |

- **Best detection + remediation:** C1 (source RAG).  
- **Best RCA:** C3 (architecture RAG) — first fair evidence that architecture context helps.  
- **C0** remains competitive on detection but **lags on RCA** (40% vs 60–67% for C1–C3).  
- **C4** is weakest on detection; dependency graphs still add less signal for these cart faults than source/arch.  
- **Agent skill still dominates** context choice (Nemotron 93% detect / DeepSeek 87% / Scout & GPT-OSS 40%).

---

## Setup

| Item | Value |
|------|--------|
| Script | `INCLUDE_C0=1 bash scripts/run_c_matrix_scenario_a_c1-c4.sh` via `aiops-c-matrix-sa-c0-c4.service` |
| Scenario | `a` (Prometheus metrics + logs; no traces) |
| Contexts | C0, C1, C2, C3, C4 |
| Agents | `nemotron-nano-3`, `maas_deepseek`, `maas_qwen3`, `maas_llama-scout`, `maas_gpt-oss-120b` |
| Faults | `scale_zero`, `config_corruption`, `kill_pod` |
| Timeout | 900s / run |
| Corpus abort | empty C1–C4 → `agent_error=corpus_empty:…` (not observed) |

---

## Overall results (n=75)

| Metric | Rate |
|--------|------|
| Detection | **52/75 (69%)** |
| RCA correct | **41/75 (55%)** |
| Remediation correct | **38/75 (51%)** |
| Remediation executed | 19/75 (25%) |
| Recovery verified | 9/75 (12%) |
| RCA judge correct | 22/75 (29%) |
| Remediation judge correct | 30/75 (40%) |
| MTTD (detected only) | median **73.0s**, mean 96.3s (n=52) |
| Forced RAG tool used (C1–C4) | **60/60** (100%) |
| `compare_telemetry` used | 33/75 |
| `corpus_empty` aborts | **0/75** |

---

## By agent (15 runs each: 5 contexts × 3 faults)

| Agent | Detect | RCA | Rem | Exec | MTTD med | RAG (C1–C4) |
|-------|--------|-----|-----|------|----------|-------------|
| **nemotron-nano-3** | **14/15 (93%)** | **13/15 (87%)** | **13/15 (87%)** | **11/15 (73%)** | 68s | 12/12 |
| **maas_deepseek** | 13/15 (87%) | 12/15 (80%) | 9/15 (60%) | 0/15 | **37s** | 12/12 |
| **maas_qwen3** | 13/15 (87%) | 9/15 (60%) | 9/15 (60%) | 4/15 | 105s | 12/12 |
| maas_llama-scout | 6/15 (40%) | 2/15 (13%) | 2/15 (13%) | 0/15 | 25s | 12/12 |
| maas_gpt-oss-120b | 6/15 (40%) | 5/15 (33%) | 5/15 (33%) | 4/15 | 185s | 12/12 |

Nemotron remains the clear leader on accuracy and remediation execution. DeepSeek is fast and strong on detect/RCA but still never executes remediations in-harness. Llama Scout and GPT-OSS remain weak; Scout detects only on C0/C1 and **0/9** on C2–C4.

---

## Detection × agent × context

| Agent | C0 | C1 | C2 | C3 | C4 |
|-------|----|----|----|----|-----|
| nemotron-nano-3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 |
| maas_deepseek | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| maas_qwen3 | 3/3 | 1/3 | 3/3 | 3/3 | 3/3 |
| maas_llama-scout | 3/3 | 3/3 | **0/3** | **0/3** | **0/3** |
| maas_gpt-oss-120b | 1/3 | 2/3 | 1/3 | 2/3 | **0/3** |

## RCA × agent × context

| Agent | C0 | C1 | C2 | C3 | C4 |
|-------|----|----|----|----|-----|
| nemotron-nano-3 | 3/3 | 3/3 | 2/3 | 3/3 | 2/3 |
| maas_deepseek | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| maas_qwen3 | 1/3 | 0/3 | 3/3 | 3/3 | 2/3 |
| maas_llama-scout | 1/3 | 1/3 | 0/3 | 0/3 | 0/3 |
| maas_gpt-oss-120b | 1/3 | 2/3 | 1/3 | 1/3 | 0/3 |

Notable: DeepSeek’s C0 RCA is **0/3**, but **12/12** with any RAG corpus (C1–C4) — strongest single-agent evidence that retrieval helps diagnosis when telemetry alone is insufficient for that model.

---

## By fault (25 runs each)

| Fault | Detect | RCA | Remediation |
|-------|--------|-----|-------------|
| scale_zero | 18/25 (72%) | 14/25 (56%) | 12/25 (48%) |
| config_corruption | 18/25 (72%) | 14/25 (56%) | 13/25 (52%) |
| kill_pod | 16/25 (64%) | 13/25 (52%) | 13/25 (52%) |

Fault difficulty is relatively balanced under scenario A.

---

## Comparison vs 22 July (invalid C3/C4)

| Slice | 22 Jul (broken C3/C4) | 24–25 Jul (fixed) |
|-------|----------------------|-------------------|
| Overall detect | 32/75 (43%) | **52/75 (69%)** |
| C3 detect | 0/15 | **11/15 (73%)** |
| C4 detect | 0/15 | **8/15 (53%)** |
| C0 RCA | 6/15 (40%) | 6/15 (40%) |
| C1 RCA | 9/15 (60%) | 9/15 (60%) |
| C3 RCA | 0/15 | **10/15 (67%)** |
| `corpus_empty` | 30/75 (C3+C4) | **0/75** |

The earlier “C0 wins” story was an artifact of empty architecture/dependency indexes. With fixed corpora, **C1 leads detection; C3 leads RCA**.

---

## Interpretation

1. **Empty-corpus abort worked as designed** — no silent soft-fail; this run never hit the gate because indexes were non-empty.  
2. **Source RAG (C1)** is the best default context for scenario-A cart faults (detect + rem).  
3. **Architecture RAG (C3)** is the best RCA booster once the corpus actually contains README topology + compose wiring.  
4. **Dependency RAG (C4)** still underperforms; keep for Option 2 / fuller matrices, not as the primary context.  
5. **Model choice still outweighs context** — Scout/GPT-OSS drag the matrix; Nemotron/DeepSeek define the ceiling.  
6. **Remediation execution** remains the bottleneck (25% execute, 12% recovery verified), especially DeepSeek/Scout at 0% execute.

---

## Recommendations

1. Prefer **C1** (or C1+C3) for production-oriented scenario-A runs; do not treat C0 as “best” after this re-run.  
2. Keep **abort-on-empty-corpus** in CI so C3/C4 regressions cannot silently zero a matrix again.  
3. Prioritize agent quality (Nemotron / DeepSeek) over expanding context before Option 2.  
4. Investigate Llama Scout failure mode on C2–C4 (detect 0/9) — possible tool-loop / timeout interaction with forced RAG.  
5. Next study: Option 2 full matrix, or scenario B/C with C1+C3 only to reduce cost.

---

## Artifacts

- Result JSONs: `out/context_C{0-4}_sa_*.json` (filter mtime ≥ 2026-07-24 22:50)  
- Run log: `out/c_matrix_scenario_a_c1-c4_run.log`  
- Matrix wrapper: `scripts/run_option1_matrix_systemd.sh`  
- Corpus fix: `code/tools/rag_context.py` (`EmptyCorpusError`, C3/C4 globs)  
- Gate: `code/harness/run_harness.py` + `code/tools/context_engineering.py`
