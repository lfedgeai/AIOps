# Learnings Across Runs — Execution Gap, Context, Skills, Next Tests

**Generated:** 2026-10-09 (post-matrix refresh)
**Primary evidence:**
- Option 2 baseline 2026-09-22 → 23 (336/336) — [EVALUATION_RESULTS_23sep26.md](EVALUATION_RESULTS_23sep26.md)
- Option 2 re-run 2026-09-25 → 27 (335/336 scored) — [EVALUATION_RESULTS_27sep26.md](EVALUATION_RESULTS_27sep26.md) · [COMPARE_27sep26_vs_23sep26.md](COMPARE_27sep26_vs_23sep26.md)
**Supporting:** July context studies — [EVALUATION_RESULTS_25July26.md](EVALUATION_RESULTS_25July26.md), [CONTEXT_ENGINEERING_EVAL_REPORT.md](CONTEXT_ENGINEERING_EVAL_REPORT.md)  
**Tracking:** local MLflow `agentic_aiops_mttd_mttr` (`http://127.0.0.1:5050`)  
**Shared prompts / skills:** [`code/agents/prompts.py`](../code/agents/prompts.py) · [`code/agents/skills/`](../code/agents/skills/) (playbooks removed post–Oct 2026)

---

## 0. MLflow ↔ report alignment (audited 2026-09-28)

| Check | Sep 25–27 re-run |
|-------|------------------|
| Harness JSON (7-agent, mtime ≥ kickoff) | **335** |
| MLflow runs in window | **336** |
| Matched `run_id` with outcome metrics | **335** |
| Mismatches on detect / RCA / rem_correct / exec / recovery | **0** |
| Headline rates in [EVALUATION_RESULTS_27sep26.md](EVALUATION_RESULTS_27sep26.md) | **Match** harness + MLflow |

**Orphan MLflow run:** `20260925104620_maas_qwen38-27b_scenarioa_kill_pod_cart` — incomplete (no outcome metrics). Same cell as the harness crash: nested `suggested_remediations` / `remediation_steps` shaped as `[[...]]`, which broke `" ".join(...)` in `score_remediation`. Missing scored cell: **C2&C3 × kill_pod × qwen38-27b**.

**Report fixes applied:** design total **336** (not “335/335”); primary baseline = Sep 22–23 (not Aug 11); experiment deltas documented; MLflow orphan explained.

---

## 1. Where the prompt says “FIX it”

Shared by every matrix agent:

| Source | Text |
|--------|------|
| `prompts.py` core | DETECT → DIAGNOSE → **FIX** → REPORT; “call them — do not only suggest” |
| Skills (since Sep 25) | detect / diagnose / **remediate** / verify appended to `BASE_SYSTEM_PROMPT` (playbook reference later removed) |
| `telemetry_k8s` profile | “**FIX** it using restart/scale/delete… **MUST** call… do not just suggest” |
| Verify rules | Re-check after write tools before claiming success |

**Scoring:** `remediation_executed` requires a successful write tool — JSON suggestions never count.

---

## 2. Why models don’t execute (still true after skills)

### Sep 22–23 (no shared skills)

| Pattern | Count |
|--------|------:|
| Detected but never executed | **203** |
| Rem tool called but exec=false | **117** |
| Detect∧RCA, no rem tool | **66** |
| Successful exec | **43** |

Conditional exec given detect∧RCA: DeepSeek/Granite **0%**; GLM ~89%; Qwen38/36 ~100% when they reached RCA.

### Sep 25–27 (skills + no-CHANGELOG + longer timeout)

| Agent | Detect∧RCA | Then executed |
|-------|------------:|--------------:|
| DeepSeek | 45 | **0%** |
| Granite | 46 | **0%** |
| GLM-53-flash | 33 | **91%** |
| Qwen36 | 38 | **97%** |
| Qwen38 | 18 | **83%** |
| Qwen3 | 15 | **7%** |

**What skills changed:** large lifts in RCA (+25 pp) and exec (+12 pp) **overall**, driven by executors (GLM/Qwen36) and Granite/DeepSeek **diagnosis** — not by turning diagnosers into actors.

**What skills did not change:** DeepSeek/Granite remain **diagnose→JSON** with 0% exec. Soft instructions + skills text ≠ hard actuation policy.

**Mechanisms:**

1. Instruction ≠ policy (recommend vs call write tools).
2. Wrong fix class (`scale_zero` → restart).
3. Forced RAG can burn early rounds.
4. Reliability tax (timeouts) — mitigated for Qwen36 (32→8 errors) by longer timeout.
5. No harness gate forbidding final JSON without a rem tool after `detected=true`.

---

## 3. How CONTEXT_C is delivered

### Corpus legend

| Label | Meaning | Tool(s) |
|-------|---------|---------|
| **C0** | No RAG (telemetry / K8s tools only) | — |
| **C1** | Source code (`src/**`) | `rag_search_source` |
| **C2** | Docs / README / changelog (CHANGELOG excluded in Sep 25–27) | `rag_search_docs` |
| **C3** | Architecture / topology overview | `rag_search_architecture` |
| **C4** | Dependencies (compose, Dockerfiles, collector configs) | `rag_search_dependencies` |

**Multi-corpus notation:** use **`&`** for “and” (explicit multi-select). Do **not** use a hyphen — `C2-4` looks like a range (2 through 4). Examples: **C1&C2** = source + docs (`CONTEXT_C=1,2`); **C2&C4** = docs + dependencies; **C1&C2&C3&C4** = all corpora. Filenames may still use hyphens (`context_C1-2_…json`); that is an artifact of replacing commas, not a range.

| Step | Mechanism |
|------|-----------|
| Config | `CONTEXT_C` → corpora C1–C4 → `rag_search_*` tools |
| Index | TF-IDF; chunks ~900 / overlap 120; **`INDEX_VERSION=no-changelog-v1`** |
| C2 docs | README, `docs/**`, CONTRIBUTING, `.github/**` — **CHANGELOG.md removed** (38 chunks vs ~130) |
| Force | Round 0 must call enabled RAG tools before other tools unlock |

Corpora are wired and used; live cluster state still dominates detection.

---

## 4. Does context matter?

**Sep 22–23:** C2/C3 roughly doubled RCA vs C0; exec flat ~10–14%.

**Sep 25–27 (with skills):** even **C0** RCA rose to **43%** (was 19%) — skills help without RAG. Best RCA contexts: **C1&C2 (71%)**, **C4** and **C2&C4 (67%)**. Exec still not driven by context alone (~19–33% by context).

Static docs help **topology/RCA**; they are a weak ops brain for **actuation**. CHANGELOG was noise for harness-injected K8s faults (not “what changed in prod”).

---

## 5. Chunking / corpus hygiene

CHANGELOG exclusion is **done** for this campaign. Next, still useful:

| Change | Priority |
|--------|----------|
| Ops-focused default RAG queries | High |
| Structure-aware MD/YAML chunks | Medium |
| Path boost for cart/deploy manifests | Medium |
| Size/overlap ablation on C2/C3 only | Low (after isolation runs) |

Do **not** expect chunking alone to fix DeepSeek/Granite 0% exec.

---

## 6. What matters most for self-adapting AIOps

1. **Actuation discipline** (hard gate or executor sidecar) — skills alone insufficient for diagnoser models.
2. **Observed-state remediation** (no hard-coded app playbooks — agents choose write tools from cluster signals).
3. **Live state first** for K8s faults; RAG for topology.
4. **Ops-grade corpora** + hygiene (CHANGELOG demotion done).
5. **Verify loop** — recovery 12%→22%; still far from closed-loop.
6. **Endpoint reliability** — drop Llama; Qwen36 now usable after timeout fix.
7. **Isolate experiment axes** — skills vs corpus vs timeout were confounded in Sep 25–27.

---

## 7. Recommended next tests

| # | Experiment | Status / note |
|---|------------|----------------|
| 1 | Shared diagnose→execute harness gate | **Still top priority** for 0% exec models |
| 2 | Two-agent cascade (DeepSeek/Granite → GLM/Qwen36) | Matches strengths |
| 3 | Flatten `suggested_remediations` in `score_remediation` | **Harness bug** — caused 1/336 crash |
| 4 | Ablate **skills only** vs **no-CHANGELOG only** vs **timeout only** | Untangle Sep25 confound |
| 5 | Interleave: 1 RAG → mandatory `get_pod_status` | Reduce RAG-only finish |
| 6 | Verify-or-retry scoring | Raise recovery |
| 7 | Drop Llama from default lineup | Confirmed 1/48 detect |
| 8 | Chunking ablation (after #4) | Secondary |

---

## 8. Links

- [EVALUATION_RESULTS_27sep26.md](EVALUATION_RESULTS_27sep26.md) — skills + no-CHANGELOG re-run (audited)
- [EVALUATION_RESULTS_23sep26.md](EVALUATION_RESULTS_23sep26.md) — Sep 22–23 baseline
- [COMPARE_27sep26_vs_23sep26.md](COMPARE_27sep26_vs_23sep26.md)
- [RUN_NOTES_25Sep26.md](RUN_NOTES_25Sep26.md) · [RUN_NOTES_14Sep26.md](RUN_NOTES_14Sep26.md)
- [CONTEXT_ENGINEERING_EVAL_REPORT.md](CONTEXT_ENGINEERING_EVAL_REPORT.md) · [ARCHITECTURE.md](ARCHITECTURE.md)
- Skills: [`code/agents/skills/`](../code/agents/skills/)

---

<!-- AUTO:POST_MATRIX_LEARNINGS -->
## Post-matrix update — 07 Oct 2026

**Kickoff:** `2026-10-07 20:58:06` · **New runs:** 336 · **Baseline archive:** `out/archive_sep22_option2` (n=288)
**Eval report:** [EVALUATION_RESULTS_09oct26.md](EVALUATION_RESULTS_09oct26.md)

Experiment vs Sep 22–23 baseline: shared AIOps skills in system prompt (AGENT_SKILLS=1), CHANGELOG removed from C2 docs corpus (INDEX_VERSION=no-changelog-v1), AGENT_TIMEOUT=900 / detection=1200. See docs/COMPARE_09oct26_vs_23sep26.md and docs/EVALUATION_RESULTS_09oct26.md.

| Scope | Detect | RCA | Exec | Recovery | agent_error |
|-------|-------:|----:|-----:|---------:|------------:|
| ALL | 69% → **70%** (+1) | 38% → **57%** (+19) | 12% → **31%** (+19) | 12% → **26%** (+14) | 40/288 → **20/336** |
| maas_deepseek | 94% → **100%** (+6) | 85% → **94%** (+8) | 0% → **0%** (+0) | 0% → **0%** (+0) | 0/48 → **0/48** |
| maas_glm-53-flash | 100% → **100%** (+0) | 38% → **73%** (+35) | 33% → **56%** (+23) | 33% → **54%** (+21) | 0/48 → **0/48** |
| maas_granite | 88% → **77%** (-10) | 44% → **67%** (+23) | 0% → **6%** (+6) | 0% → **2%** (+2) | 1/48 → **1/48** |
| maas_llama31-70b | 0% → **0%** (+0) | 0% → **0%** (+0) | 0% → **0%** (+0) | 0% → **0%** (+0) | 7/48 → **5/48** |
| maas_qwen3 | 100% → **88%** (-12) | 25% → **44%** (+19) | 8% → **40%** (+31) | 6% → **27%** (+21) | 0/48 → **6/48** |
| maas_qwen36-27b | 33% → **90%** (+56) | 33% → **85%** (+52) | 33% → **83%** (+50) | 33% → **65%** (+31) | 32/48 → **5/48** |
| maas_qwen38-27b | 0% → **33%** (+33) | 0% → **35%** (+35) | 0% → **33%** (+33) | 0% → **33%** (+33) | 0/0 → **3/48** |

_Auto-appended by `scripts/update_learnings_across_runs_report.py` after Option 2 COMPLETE._
