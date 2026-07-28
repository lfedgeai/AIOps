# Evaluation Results — 25–26 July 2026 (Option 2: all CONTEXT_C permutations)

**Date:** 2026-07-25 10:01 → 2026-07-26 00:26 AEST (~**14.4 h**)  
**Design:** Option 2 — scenario **A** × **all CONTEXT_C subsets** × 5 agents × 3 faults  
**Runs:** **240/240** completed (`failures=0`)  
**Variant:** `cart` (OpenTelemetry Demo)  
**Forced RAG:** on whenever any of C1–C4 is present (`FORCE_RAG=1`)  
**Corpus gate:** empty indexes abort before fault injection — **0** `corpus_empty`  
**Tracking:** local MLflow only (`http://127.0.0.1:5050`, experiment `agentic_aiops_mttd_mttr`)  
**Outputs:** `out/context_C{label}_sa_{fault}_{agent}.json` (commas → hyphens in filenames; mtime ≥ 2026-07-25 10:00)  
**Log:** `out/c_matrix_scenario_a_c_permutations_run.log`  
**Script:** `scripts/run_c_matrix_scenario_a_c_permutations.sh` via `aiops-c-matrix-sa-c-permutations.service`

Follows the fixed-corpus Option 1 re-run ([EVALUATION_RESULTS_25July26.md](EVALUATION_RESULTS_25July26.md)). This study asks whether **combining** corpora helps or hurts once singles are non-empty.

---

## Executive summary

| Metric | Rate |
|--------|------|
| Detection | **126/240 (52%)** |
| RCA correct | **81/240 (34%)** |
| Remediation correct | **83/240 (35%)** |
| Remediation executed | 25/240 (**10%**) |
| Recovery verified | 23/240 (10%) |
| MTTD (detected) | median **70s**, mean 106s (n=126) |
| Forced RAG observed (C≠0) | 217/225 (96%) |
| `corpus_empty` | **0** |

**Main findings**

1. **More corpora ≠ better.** Detection falls with corpus count: C0 **87%** → singles **65%** → pairs **56%** → triples **33%** → all-four **27%**.
2. **Best multi-select for RCA:** **C{1,4}** and **C{2,4}** (both **8/15 = 53% RCA**); **C{1,4}** also strong on detect (**11/15 = 73%**).
3. **C0 detects but does not diagnose:** **13/15 (87%)** detect vs **1/15 (7%)** RCA — telemetry finds the symptom; without RAG, agents rarely name the right cause under forced-eval scoring.
4. **Agent skill dominates context:** **DeepSeek 45/48 (94%)** detect **and** RCA; Scout/GPT-OSS ~31–33% detect; Nemotron drops to **50%** detect / **25%** RCA across the full permutation space (vs 93%/87% on Option 1 singles).
5. **Forced multi-RAG tax:** requiring 2–4 RAG tools before other tools unlock burns window time; weak models and Nemotron degrade sharply on size≥3 sets.
6. **Remediation execution remains the bottleneck** (10% execute); DeepSeek is **0/48** execute despite excellent RCA.

---

## Setup

| Item | Value |
|------|--------|
| Scenario | `a` (`telemetry_k8s`: metrics + logs; no traces) |
| Contexts (16) | `0` + all non-empty subsets of `{1,2,3,4}` |
| Agents | `nemotron-nano-3`, `maas_deepseek`, `maas_qwen3`, `maas_llama-scout`, `maas_gpt-oss-120b` |
| Faults | `scale_zero`, `config_corruption`, `kill_pod` |
| Per cell | 5 × 3 = 15 runs per context label |
| Timeout | detection window ~360s; agent subprocess capped to remaining window |
| Corpora (preflight) | C1=1784, C2=130, C3=129, C4=166 chunks |

### Context vocabulary

| Code | Corpus / tool |
|------|----------------|
| C0 | No RAG |
| C1 | Source → `rag_search_source` |
| C2 | Docs → `rag_search_docs` |
| C3 | Architecture → `rag_search_architecture` |
| C4 | Dependencies → `rag_search_dependencies` |

Multi-select (e.g. `1,4`) forces **each** matching RAG tool before other tools unlock.

---

## Overall results (n=240)

| Metric | Rate |
|--------|------|
| Detection | 126/240 (52%) |
| RCA correct | 81/240 (34%) |
| Remediation correct | 83/240 (35%) |
| Remediation executed | 25/240 (10%) |
| Recovery verified | 23/240 (10%) |
| RCA judge correct | 57/240 (24%) |
| Remediation judge correct | 57/240 (24%) |
| Harness timeouts (`agent_error`) | 4/240 |
| `corpus_empty` | 0/240 |

Compared with Option 1 (75 singles-only, fixed corpora): Option 1 had **69%** detect / **55%** RCA. Diluting the matrix with large multi-corpus sets pulls the aggregate down — expected if size≥3 is harmful.

---

## By agent (48 runs each = 16 contexts × 3 faults)

| Agent | Detect | RCA | Rem | Exec | MTTD med |
|-------|--------|-----|-----|------|----------|
| **maas_deepseek** | **45/48 (94%)** | **45/48 (94%)** | **35/48 (73%)** | **0/48 (0%)** | **50s** |
| maas_qwen3 | 26/48 (54%) | 15/48 (31%) | 27/48 (56%) | 8/48 (17%) | 146s |
| nemotron-nano-3 | 24/48 (50%) | 12/48 (25%) | 11/48 (23%) | 11/48 (23%) | 134s |
| maas_llama-scout | 16/48 (33%) | 3/48 (6%) | 4/48 (8%) | 2/48 (4%) | 28s |
| maas_gpt-oss-120b | 15/48 (31%) | 6/48 (12%) | 6/48 (12%) | 4/48 (8%) | 100s |

**DeepSeek** is the clear winner on detect/RCA across nearly every context, including triples and `1,2,3,4` (**3/3** detect+RCA on full set). It still **never executes** remediations in-harness (suggests only).

**Nemotron** led Option 1 but here loses half of detections — concentrated on multi-corpus cells and non-`scale_zero` faults (see below). Many Nemotron miss harness JSONs show empty `ai_metrics_rounds` / tool lists; MLflow audit (later section) checks whether the agent loop still ran.

**Scout / GPT-OSS** remain weak; Scout again fails most C2+ singles and large sets.

---

## By fault (80 runs each)

| Fault | Detect | RCA | Remediation |
|-------|--------|-----|-------------|
| **scale_zero** | **53/80 (66%)** | **40/80 (50%)** | **38/80 (48%)** |
| kill_pod | 38/80 (48%) | 21/80 (26%) | 24/80 (30%) |
| config_corruption | 35/80 (44%) | 20/80 (25%) | 21/80 (26%) |

`scale_zero` remains easiest. Detect collapses further on large corpus sets for all faults (e.g. `scale_zero` size4: **1/5**).

---

## By context label (15 runs each)

| Context | Detect | RCA | Rem | Exec | MTTD med |
|---------|--------|-----|-----|------|----------|
| **C0** | **13/15 (87%)** | 1/15 (**7%**) | 2/15 (13%) | 0/15 | 61s |
| C1 | 12/15 (80%) | 5/15 (33%) | 5/15 (33%) | 2/15 | 56s |
| C2 | 8/15 (53%) | 5/15 (33%) | 8/15 (53%) | 3/15 | 120s |
| C3 | 11/15 (73%) | 6/15 (40%) | 7/15 (47%) | 3/15 | 67s |
| C4 | 8/15 (53%) | 7/15 (47%) | 6/15 (40%) | 3/15 | 91s |
| C{1,2} | 8/15 (53%) | 4/15 (27%) | 3/15 (20%) | 1/15 | 81s |
| C{1,3} | 6/15 (40%) | 5/15 (33%) | 7/15 (47%) | 2/15 | 56s |
| **C{1,4}** | **11/15 (73%)** | **8/15 (53%)** | **8/15 (53%)** | 2/15 | 75s |
| C{2,3} | 7/15 (47%) | 6/15 (40%) | 7/15 (47%) | 2/15 | 87s |
| **C{2,4}** | **10/15 (67%)** | **8/15 (53%)** | 7/15 (47%) | 3/15 | 200s |
| C{3,4} | 8/15 (53%) | 5/15 (33%) | 4/15 (27%) | 1/15 | 121s |
| C{1,2,3} | 6/15 (40%) | 6/15 (40%) | 6/15 (40%) | 1/15 | 65s |
| C{1,2,4} | 5/15 (33%) | 4/15 (27%) | 3/15 (20%) | 0/15 | 74s |
| C{1,3,4} | 5/15 (33%) | 4/15 (27%) | 3/15 (20%) | 1/15 | 63s |
| C{2,3,4} | 4/15 (27%) | 4/15 (27%) | 3/15 (20%) | 1/15 | 61s |
| C{1,2,3,4} | 4/15 (27%) | 3/15 (20%) | 4/15 (27%) | 0/15 | 111s |

### Rankings

| Goal | Best labels | Avoid |
|------|-------------|--------|
| Detection | C0, C1, C{1,4}, C3 | size≥3, especially `2,3,4` / `1,2,3,4` |
| RCA | **C{1,4}**, **C{2,4}**, C4 | **C0** (7%), full set |
| Balanced (det+RCA) | **C{1,4}** | C0 (det-only), triples+ |

---

## Effect of corpus cardinality

| # corpora | n | Detect | RCA | Rem | MTTD med |
|-----------|---|--------|-----|-----|----------|
| 0 (C0) | 15 | **87%** | **7%** | 13% | 61s |
| 1 | 60 | 65% | 38% | 43% | 68s |
| 2 | 90 | 56% | **40%** | 40% | 92s |
| 3 | 60 | 33% | 30% | 25% | 65s |
| 4 | 15 | **27%** | 20% | 27% | 111s |

**Interpretation:** One corpus helps RCA vs C0; two can still help RCA (peak at pairs that include C4); three+ mostly burns the forced-RAG budget and hurts detection. “Dump all context in” is actively harmful under this policy.

### Marginal presence (among C≠0 only)

| Bit | When present | When absent |
|-----|--------------|-------------|
| C1 | det 48% / rca 32% | det 53% / rca 39% |
| C2 | det 43% / rca 33% | det 58% / rca 38% |
| C3 | det 42% / rca 32% | det 59% / rca 39% |
| C4 | det 46% / rca **36%** | det 55% / rca 35% |

No single bit is a free lunch; **C2/C3 presence correlates with lower detection** (confounded with larger sets). C4 is the only bit that does not worsen RCA when present.

---

## Top-3 agents only (Nemotron + DeepSeek + Qwen)

Removes Scout/GPT-OSS noise (9 runs per context):

| Context | Detect | RCA | Rem |
|---------|--------|-----|-----|
| C0 | 7/9 | 1/9 | 2/9 |
| C1 | 7/9 | 4/9 | 4/9 |
| C2 | 6/9 | 5/9 | **7/9** |
| C3 | **8/9** | 5/9 | 6/9 |
| C4 | 6/9 | **6/9** | 5/9 |
| **C{1,4}** | **8/9** | **6/9** | **6/9** |
| **C{2,4}** | **8/9** | **6/9** | 4/9 |
| C{1,3} | 5/9 | 5/9 | **7/9** |
| C{1,2,3,4} | 4/9 | 3/9 | 4/9 |

Among capable models, **C{1,4}** and **C{2,4}** are the preferred multi-selects; singles C3/C4 remain competitive for RCA; full set still lags.

---

## Detection heatmap (agent × context)

Cells = detected / 3 faults.

| Agent | 0 | 1 | 2 | 3 | 4 | 1,2 | 1,3 | 1,4 | 2,3 | 2,4 | 3,4 | 1,2,3 | 1,2,4 | 1,3,4 | 2,3,4 | 1,2,3,4 |
|-------|---|---|---|---|---|-----|-----|-----|-----|-----|-----|-------|-------|-------|-------|---------|
| nemotron | 3 | 1 | 3 | 3 | 1 | 2 | 1 | 2 | 1 | 2 | 2 | 2 | **0** | 1 | **0** | **0** |
| deepseek | 1 | 3 | 2 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | **3** |
| qwen3 | 3 | 3 | 1 | 2 | 2 | **0** | 1 | 3 | 2 | 3 | 1 | 1 | 2 | **0** | 1 | 1 |
| scout | 3 | 3 | **0** | **0** | **0** | 3 | 1 | 2 | 1 | 1 | 2 | **0** | **0** | **0** | **0** | **0** |
| gpt-oss | 3 | 2 | 2 | 3 | 2 | **0** | **0** | 1 | **0** | 1 | **0** | **0** | **0** | 1 | **0** | **0** |

## RCA heatmap (agent × context)

| Agent | 0 | 1 | 2 | 3 | 4 | 1,2 | 1,3 | 1,4 | 2,3 | 2,4 | 3,4 | 1,2,3 | 1,2,4 | 1,3,4 | 2,3,4 | 1,2,3,4 |
|-------|---|---|---|---|---|-----|-----|-----|-----|-----|-----|-------|-------|-------|-------|---------|
| nemotron | 0 | 0 | 2 | 0 | 1 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 0 | 1 | 0 | 0 |
| deepseek | 1 | 3 | 2 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | **3** |
| qwen3 | 0 | 1 | 1 | 2 | 2 | 0 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 0 |
| scout | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| gpt-oss | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |

DeepSeek’s C0 RCA is only **1/3**, but **44/45** with any RAG — strongest evidence that retrieval unlocks diagnosis for that model.

---

## Comparison to Option 1 (singles slice)

Same labels C0–C4 inside this matrix vs Option 1 report (24–25 Jul):

| Context | Opt1 det / RCA | Opt2 singles det / RCA |
|---------|----------------|-------------------------|
| C0 | 73% / 40% | **87% / 7%** |
| C1 | 80% / 60% | 80% / 33% |
| C2 | 67% / 60% | 53% / 33% |
| C3 | 73% / 67% | 73% / 40% |
| C4 | 53% / 47% | 53% / 47% |

Detection on singles is broadly stable; **RCA on C0–C3 is lower in this campaign** (especially C0: 40%→7%). Same harness scoring, different calendar day / model-provider variance / judge noise. Treat Option 2 singles as a **noisy replicate**, not a contradiction of “RAG helps RCA,” which still holds: every non-C0 single beats C0 on RCA here.

Option 1’s Nemotron dominance does **not** generalize to the permutation space.

---

## Interpretation

1. **Policy cost of forced multi-RAG:** Each added corpus adds a mandatory tool round. Under a ~6 min window, size≥3 leaves little time for `compare_telemetry` / K8s actions — visible as detect collapse for everyone except DeepSeek.
2. **C0 = fast symptom catch, weak causal language:** High detect, near-zero RCA under our grader.
3. **Best practical defaults:** Prefer **C{1,4}** (source + deps) or single **C1/C3**; avoid “enable all corpora.”
4. **Model choice first:** DeepSeek (diagnose) + a model that actually executes remediations (Nemotron/Qwen when they work).
5. **C2 alone** still helps rem for strong models (top-3 rem 7/9) but hurts Scout; use selectively.

---

## Recommendations

1. Ship **C{1,4}** or **C1** as default context for scenario-A cart faults; do not enable C1–C4 together under forced RAG.
2. Cap forced RAG to **≤2** corpora per run in future matrices, or make multi-RAG optional after first hit.
3. Keep empty-corpus abort in CI.
4. Investigate DeepSeek **execute=0** (tool allowlist / prompt / dry-run?) separately from RCA quality.
5. Investigate Nemotron multi-corpus failures (window exhaustion vs empty agent_output) via MLflow (see audit section below).
6. Next study: scenario B/C with **C{1,4}** only × top-3 agents (cheaper than another 240-run sweep).

---

## Artifacts

- Result JSONs: `out/context_C*_sa_*.json` (filter mtime ≥ 2026-07-25 10:00)
- Run log: `out/c_matrix_scenario_a_c_permutations_run.log`
- Matrix: `scripts/run_c_matrix_scenario_a_c_permutations.sh`
- Prior singles study: [EVALUATION_RESULTS_25July26.md](EVALUATION_RESULTS_25July26.md)

---

## MLflow verification (prompts, flows, tool calls)

Audited all **240** MLflow runs in experiment `agentic_aiops_mttd_mttr` with `start_time` in the Option 2 window (2026-07-25 00:00–15:00 UTC). Artifacts present on **240/240** runs: `agent_llm_prompts.json`, `agent_llm_tool_calls.json`, `agent_llm_rounds.json`, `agent_llm_thinking.json`, `harness_run.json`.

### Confirmed (matches the report)

| Claim from harness JSON | MLflow result |
|-------------------------|---------------|
| 126/240 detect, 81/240 RCA, 83/240 rem, 25/240 exec | **Exact match** on logged metrics (0 mismatches vs `out/*.json`) |
| 16 contexts × 5 agents × 3 faults | Tag/param counts exact (`FINISHED` × 240) |
| Detect falls as corpus count rises | Same rates from MLflow-only aggregation |
| Best RCA: C{1,4} / C{2,4} (8/15) | Confirmed from MLflow metrics alone |
| C0: high detect, near-zero RCA | Confirmed (13/15 detect, 1/15 RCA) |
| DeepSeek execute = 0 | Confirmed; only **1** write-like tool across 48 runs (`restart_deployment` once) |
| Forced RAG fires first on C≠0 | First tool is required `rag_search_*` on **223/225** C≠0 runs |
| Required RAG tools completed | **214/225** called every required corpus tool; **9** partial; **2** none |

Forced-RAG ordering example (C≠0 first-tool histogram): `rag_search_source` 118, `rag_search_docs` 60, `rag_search_architecture` 30, `rag_search_dependencies` 15, `(none)` 2. C0 typically starts with `compare_telemetry` (9) or `search_logs` (6).

Corpus-size tool tax (artifact RAG count scales with label size):

| Size | mean RAG calls in artifact | Detect |
|------|----------------------------|--------|
| 0 | 0.0 | 13/15 |
| 1 | 1.0 | 39/60 |
| 2 | 2.0 | 50/90 |
| 3 | 2.8 | 20/60 |
| 4 | 3.6 | 4/15 |

On size≥3, only DeepSeek stays perfect (**15/15** detect); Scout **0/15**, GPT-OSS **1/15**, Nemotron **3/15**.

### Discrepancies / nuances (MLflow corrects the story)

**1. Harness `agent_output` under-reports Nemotron tool use**  
Many Option 2 harness JSONs for Nemotron misses show empty `ai_metrics_rounds` / no tools. MLflow shows **all 48 Nemotron runs have 8 tool calls**. Misses are not “no agent loop” — they **do** run forced RAG + telemetry/K8s tools, then fail to finish with a successful detection declaration.

Typical Nemotron miss on `C{1,2,3,4}`: tools = four `rag_search_*` → `compare_telemetry` → `get_pod_status` → `get_events` → `search_logs`, with the **last assistant turn still requesting another tool** (no final JSON). Logged `agent_output.json` often starts with `[rag_force] harness executing…` (not pure JSON), so harness parsing of declared RCA/rem is unreliable even when MLflow tool traces are complete. Also **45/48** Nemotron runs have `ai_rounds=None` in MLflow metrics (logging gap), while C0 successes log `ai_rounds=8`.

**2. DeepSeek “remediation_correct” is mostly declarative, not operational**  
**45/48** DeepSeek runs use **only** `rag_search_*` tools (no live telemetry/K8s after RAG). Example `C{1,4}` × `scale_zero`: tools = `rag_search_source`, `rag_search_dependencies` only → final assistant JSON declares `detected/rca/remediations` with **no** `scale_deployment`. Thinking text *plans* `search_logs` / `get_pod_status` but those calls never appear in `agent_llm_tool_calls.json`. So MLflow **confirms** high RCA scores and **explains** exec=0: the model answers from RAG + prior knowledge, then stops. Judge often still marks remediation text as correct.

**3. Harness string “RAG used” was slightly low**  
Harness-side string scan: **217/225**. Artifact tool lists: **223/225** with ≥1 RAG call. Two true no-RAG failures (Qwen `C{1,3,4}` scale_zero with empty tools; GPT-OSS `C{1,2,3}` config with rounds=1 and empty tool artifact).

**4. Llama Scout “tool call in prose”**  
Example C2 × `scale_zero`: after `rag_search_docs`, the assistant **writes** `[search_logs(...)]` in natural language instead of emitting a structured tool call, then thinking records `detected: false`. Explains Scout’s collapse when forced into RAG-first flows.

**5. No contradiction on headline rankings**  
MLflow-only re-ranking of contexts by RCA is identical to the harness table (C{1,4}/C{2,4} top; C0 bottom; full set near bottom). The cardinality / DeepSeek / weak-model stories hold; the audit mainly **adds mechanism** (RAG tax + no final answer; DeepSeek suggest-only; Nemotron metrics/artifact parsing gaps).

### MLflow takeaways for next runs

1. Trust **MLflow tool artifacts** over harness `agent_output` for Nemotron when diagnosing misses.  
2. Treat DeepSeek remediation_correct as **suggestion quality**, not cluster recovery.  
3. Cap forced corpora at ≤2 to leave rounds for a final structured answer.  
4. Fix Nemotron `ai_rounds` metric logging and strip `[rag_force]` prefixes from persisted `agent_output`.
