# Evaluation Results — Option 2 matrix (6 agent slots, 5 distinct models)

**Generated:** 2026-10-22 17:23 AEST · **revised 2026-10-23** (slot-identity & serving-infrastructure findings, corrected model labels, agent-failure verification, environment-vs-Oct 10–12 analysis)  
**Design:** Option 2 — scenario **A** × all CONTEXT_C subsets (16) × **6 agent slots** × 3 faults (the two Qwen3.8-27B slots differ only by serving gateway — see [slot-identity note](#qwen-family-qwen3-14b-vs-qwen38-27b--two-gateways))  
**Runs:** **288/288** completed  
**Variant:** `cart` · **Forced RAG:** on for C≠0  
**Tracking:** local MLflow (`http://127.0.0.1:5050`), experiment `agentic_aiops_mttd_mttr`  
**Primary baseline:** [EVALUATION_RESULTS_27sep26.md](EVALUATION_RESULTS_27sep26.md) (Option 2, skills + no-CHANGELOG)  
**Run notes:** [RUN_NOTES_22Oct26.md](RUN_NOTES_22Oct26.md)  

> **Lineup (this run):** `maas_deepseek`, `maas_glm-53-flash`, `maas_granite`, `maas_qwen3`, `qwen3.8-27b_maas-litemaas`, `qwen3.8-27b_maas-rhdp`.  
> **Comparability:** Shared prompts (`code/agents/prompts.py`) and API preflight. Primary Δ table uses Sep 25–27; older Aug-11 (different design size) appears only in legacy combo sections.  
> **Environment caveat (read Δ columns with this):** the cluster environment changed materially between batches — kafka broker was crash-looping (decoy kafka/checkout errors) and load-generator traffic was starved (~2–25 checkout spans/hour) during the Oct 10–12 batch; this run had a healthy kafka and ~1,300–3,200 checkout spans/hour. See the deliberate-changes table in [RUN_NOTES_22Oct26.md](RUN_NOTES_22Oct26.md). Deltas vs Sep 25–27 therefore mix agent capability with task difficulty (louder fault signal, no decoys); they are **not** a pure model bake-off. Sep MTTD was also produced by the pre-fix metric pipeline (exit-clock fallback), so the MTTD Δ row is not latency-comparable until Sep is re-derived with the fixed logic.

---

## What this report measures

Each of the **288** cells is one harness trial: inject a Kubernetes fault on a target workload (scenario A / cart variant), give the agent a time window with telemetry/K8s tools (and optional RAG corpora), then score the outcome.

| Axis | Values |
|------|--------|
| Scenario | A (single-service fault) |
| CONTEXT_C | 16 labels (`0` + subsets of {1,2,3,4}) — see corpus definitions below |
| Agent | 6 slots (5 distinct models — two slots run Qwen3.8-27B on different gateways) |
| Fault | `scale_zero`, `config_corruption`, `kill_pod` |

**Context corpora (C1–C4).** A CONTEXT_C label is the *set* of RAG corpora enabled for that
cell (e.g. `C1,4` = corpora 1 and 4; `C0` = none). Each corpus is a slice of the git-SHA-pinned
OpenTelemetry demo repo, chunked at 900 chars / 120 overlap and exposed to the agent as its own
`rag_search_*` function tool (queried in a forced round-0 phase per enabled corpus; retrieval
mechanics in the [retriever note](#by-context-18-runs-each--6-agents--3-faults) under By context):

| Context | Corpus | Content (file globs in the pinned repo) |
|---------|--------|------------------------------------------|
| C0 | — none | No RAG tools offered; the agent relies on telemetry + K8s tools only |
| C1 | `source` | Application source code: `src/**/*.{py,go,ts,js,rs,java,cs}` |
| C2 | `docs` | Product documentation: `README.md`, `docs/**/*.md`, `CONTRIBUTING.md`, `.github/**/*.md` (CHANGELOG.md deliberately excluded — noise for harness-injected K8s faults) |
| C3 | `architecture` | Service-topology material: README/compose plus files matching `*architecture*`/`*diagram*`/service-map, mermaid/flowchart blocks, and chunks mentioning service names — a chunk filter keeps only topology-relevant pieces |
| C4 | `dependencies` | Deployment/dependency configuration: `compose*.yaml`, `kubernetes/**`, `**/Dockerfile*`, otel-collector / prometheus / jaeger configs |

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
| Detection | **231/288 (80%)** | 266/335 (79%) |
| RCA correct | **209/288 (73%)** | 196/335 (59%) |
| Remediation correct (OR) | **217/288 (75%)** | 207/335 (62%) |
| Remediation executed | **87/288 (30%)** | 83/335 (25%) |
| Recovery verified | **70/288 (24%)** | 73/335 (22%) |
| MTTD median (detected) | **87s** | 100s |
| MTTR median | **95s** | — |
| MTTR verified median | **142s** | — |
| `agent_error` | **39/288** | 21/335 |

_Primary baseline = [EVALUATION_RESULTS_27sep26.md](EVALUATION_RESULTS_27sep26.md) (same Option 2 design). Older Aug-11 (5-agent / 240-cell) figures remain in family/combo sections where noted._

### Headline takeaways

1. **Detection vs execution:** detect **80%** (231/288); remediation executed **30%** (87/288) — most models still stop at diagnosis, but the gap decomposes into confabulation (deepseek/granite describe fixes without calling tools) and a **namespace interface trap** (all 73 write attempts by the diagnosers passed `namespace: "default"` and 404'd — see [exec-gap decomposition](#why-agents-stop-before-remediation-execution-exec-gap-decomposition)).
2. **RCA vs execute split:** **DeepSeek** RCA **43/48** / exec **0/48**; **Granite** RCA **41/48** / exec **0/48**.
3. **Primary executors this run:** **maas_glm-53-flash** 44/48, **qwen3.8-27b_maas-litemaas** 27/48, **qwen3.8-27b_maas-rhdp** 16/48.
4. **qwen3.8-27b (maas-litemaas):** detect/RCA/exec **28/48** with **14/48** `agent_error` (when healthy, detect usually co-occurs with RCA+exec).
5. **glm-53-flash:** detect **47/48**, RCA **47/48**, exec **44/48** (MTTD med 118s; by fault: scale_zero 33s · kill_pod 128s · config_corruption 154s — see [By fault](#by-fault-96-runs-each)).
6. **Exec spread beyond `scale_zero`, but `config_corruption` cannot recover:** exec is now scale_zero **48/96**, kill_pod **22/96**, config_corruption **17/96** (vs 48/1/1 in Oct 10–12) — yet config_corruption recovery is **0/96** because the injected fault (bad `DATABASE_HOST` env var on the deployment) cannot be undone by the available write tools (`patch_deployment` is still unimplemented). See [By fault](#by-fault-96-runs-each).
7. **Nemotron / Kimi:** not in this batch (omit from headlines until preflight-green).

### Why the numbers moved vs the Oct 10–12 batch (environment, not just capability)

This batch (Oct 20–22) is the first on the fixed MTTD/MTTR pipeline, but two **environmental**
differences vs the Oct 10–12 batch dominate the deltas — especially for act-first agents:

| | Oct 10–12 batch | Oct 20–22 batch (this run) |
|---|---|---|
| kafka broker | **crash-looping** (~122 lifetime pod restarts; checkout errored 42% — 684/1628 spans) | **healthy** (KRaft quorum env fix; 0 restarts) |
| checkout telemetry volume | **~2–25 spans/hour** (starved load-generator — fault signal nearly invisible) | **~1,300–3,200/hour** (fresh load-generator) |

**glm-53-flash write-targeting evidence** (the agent that jumped 48%→98% RCA, 35%→92% exec):
its Oct 10–12 writes went **kafka 59 / checkout 10 / cart 17** (122 writes — it was remediating
the sick-kafka decoy, not the injected cart fault); this run they went **cart 34 /
valkey-cart 13 / kafka 0** (48 writes). With the decoy gone and the fault signal 100× louder,
47/48 runs landed a targeted successful write. Diagnosers were far less affected in Oct 10–12
(deepseek 85%→90% RCA), which is why the jump is agent-specific: **the Oct 10–12 environment
specifically punished act-first agents**; glm's underlying capability likely did not change.

Scoring nuance: `valkey-cart` writes count as cart-targeted (substring match in
`write_targeted_expected`). Iteration budget was **not** a factor — both batches ran
`max_tool_iterations=20` (median rounds used: 18 in both). Full details:
[RUN_NOTES_22Oct26.md](RUN_NOTES_22Oct26.md).

## By agent (48 runs each)

| Agent | Detect | RCA | Exec | MTTD med | MTTR med | MTTR verified med |
|---|---|---|---|---|---|---|
| `maas_deepseek` | 48/48 (100%) | 43/48 (90%) | 0/48 (0%) | 64s | — | — |
| `maas_glm-53-flash` | 47/48 (98%) | 47/48 (98%) | 44/48 (92%) | 118s | 112s (n=44) | 91s (n=31) |
| `maas_granite` | 46/48 (96%) | 41/48 (85%) | 0/48 (0%) | 36s | — | — |
| `maas_qwen3` | 45/48 (94%) | 34/48 (71%) | 0/48 (0%) | 92s | — | — |
| `qwen3.8-27b_maas-litemaas` | 28/48 (58%) | 27/48 (56%) | 27/48 (56%) | 81s | 75s (n=27) | 106s (n=23) |
| `qwen3.8-27b_maas-rhdp` | 17/48 (35%) | 17/48 (35%) | 16/48 (33%) | 93s | 92s (n=16) | 156s (n=16) |

## By fault (96 runs each)

> Recomputed from the 288 batch JSONs (`out/context_*.json`, mtime ≥ 2026-10-20 12:00 UTC). This cut was present in the Oct 12 corrected report and is the most informative view of the exec/recovery gains — read the by-agent MTTD/MTTR medians through it (e.g. glm's 118s MTTD is a mix of 33s scale_zero, 128s kill_pod, 154s config_corruption).

| Fault | Detect | RCA | Exec | Recovery | `agent_error` | MTTD med | MTTR med | MTTR verified med |
|---|---|---|---|---|---|---|---|---|
| `scale_zero` | 96/96 (100%) | 90/96 (94%) | 48/96 (50%) | 48/96 (50%) | **0** | 64s | 60s (n=48) | 95s (n=48) |
| `kill_pod` | 68/96 (71%) | 61/96 (64%) | 22/96 (23%) | 22/96 (23%) | 20 | 112s | 234s (n=22) | 248s (n=22) |
| `config_corruption` | 67/96 (70%) | 58/96 (60%) | 17/96 (18%) | **0/96 (0%)** | 19 | 95s | 169s (n=17) | — (n=0) |

### Exec by fault × agent (16 cells each)

| Agent | config_corruption | kill_pod | scale_zero |
|---|---|---|---|
| `maas_deepseek` | 0/16 | 0/16 | 0/16 |
| `maas_glm-53-flash` | 13/16 | 15/16 | 16/16 |
| `maas_granite` | 0/16 | 0/16 | 0/16 |
| `maas_qwen3` | 0/16 | 0/16 | 0/16 |
| `qwen3.8-27b_maas-litemaas` | 4/16 | 7/16 | 16/16 |
| `qwen3.8-27b_maas-rhdp` | 0/16 | 0/16 | 16/16 |

### Reading this table

- **Exec gains are no longer a one-fault story** — config_corruption 1/96 → 17/96 and kill_pod 1/96 → 22/96 vs the Oct 10–12 batch (scale_zero was 48/96 then and now). The healthy-kafka / loud-signal environment is the main driver (see [Why the numbers moved](#why-the-numbers-moved-vs-the-oct-1012-batch-environment-not-just-capability)).
- **`config_corruption` recovery is 0/96 by construction.** The fault is a corrupted `DATABASE_HOST` env var on the cart deployment (`run_harness.py`), and the only implemented write tools are `restart_deployment` / `scale_deployment` / `delete_pod` (`WRITE_TOOLS`, `code/tools/agent_tools.py`); `patch_deployment` is listed in the execute-gate tool set and counted as action-appropriate in scoring (`code/harness/evaluation.py`) but has **no `execute_tool` implementation**. A restart recreates pods with the same corrupt env, so the 17 "executed" remediations (judged action-appropriate) can never verify recovery. The design ceiling for **recovery verified** is therefore **192/288 (67%)**, and `remediation correct (OR) = 75%` credits config_corruption fixes that cannot succeed with the current tool set.
- **`kill_pod` recovery is partially the platform, not the agent.** Exec = recovery exactly (22/22), with a median write→recovery lag of **18.5s** — the deployment controller recreates the deleted pod on its own, so a well-timed agent write co-occurs with self-healing (echoes Oct 12 rec: treat as self-healing or use a different success criterion).
- **All 39 `agent_error`s sit on the two slow faults** (config_corruption 19, kill_pod 20, scale_zero 0) — the faults that need more rounds are also where the qwen3.8-27b gateway timeouts land (see [Agent errors](#agent-errors)).

## Model family comparison

### Nemotron

| Lineup | Detect | RCA | Exec | Notes |
|---|---|---|---|---|
| `nemotron-nano-3` (Aug) | 38/96 (40%) | 37/96 (39%) | 34/96 (35%) | prior batch |
| `nemotron-cascade-2` (this run) | n/a | n/a | n/a | not in this batch |

### Qwen family (`qwen3-14b` vs Qwen3.8-27B × two gateways)

> **⚠️ Slot-identity finding (verified 23 Oct 26):** `qwen3.8-27b_maas-litemaas` and `qwen3.8-27b_maas-rhdp`
> are **the same underlying model — Qwen 3.8 27B** — served through two different gateways.
> Evidence: MLflow round records show `Qwen3.8-27B` for the qwen36 slot and `qwen38-27b` for
> the qwen38 slot (both naming conventions for Qwen 3.8 27B); the qwen36 agent's env, code
> default, and gateway (`litemaas`, which serves exactly one model, `Qwen3.8-27B`) have all
> pointed at Qwen 3.8 since the slot was created — no Qwen 3.6 exists anywhere in this setup.
> Read the two rows below as a **same-model, different-gateway A/B** (litemaas vs maas-rhdp),
> not as two models: the deltas measure serving-stack reliability/latency, not model capability.
> **Label mapping used throughout this report:** `qwen3.8-27b_maas-litemaas` is the raw slot
> `maas_qwen36-27b`; `qwen3.8-27b_maas-rhdp` is the raw slot `maas_qwen38-27b` (raw JSONs and
> MLflow still carry the original approach ids). The "6-agent lineup" is therefore effectively
> **5 distinct models**. Head-to-head detail: [Serving infrastructure experiment](#serving-infrastructure-experiment-qwen38-27b--maas-litemaas-vs-maas-rhdp).

| Model | Detect | RCA | Exec | MTTD med | MTTR med | MTTR verified med |
|---|---|---|---|---|---|---|
| qwen3 (RHDP) | 45/48 (94%) | 34/48 (71%) | 0/48 (0%) | 92s | — | — |
| qwen3.8-27b_maas-rhdp | 17/48 (35%) | 17/48 (35%) | 16/48 (33%) | 93s | 92s (n=16) | 156s (n=16) |
| qwen3.8-27b_maas-litemaas | 28/48 (58%) | 27/48 (56%) | 27/48 (56%) | 81s | 75s (n=27) | 106s (n=23) |
| qwen3 (Aug 11 MLflow) | 26/48 (54%) | 16/48 (33%) | 7/48 (15%) | 140s | — | — |

## Serving infrastructure experiment: Qwen3.8-27B — maas-litemaas vs maas-rhdp

Unintended but clean A/B: the same model weights (Qwen 3.8 27B) ran as two separate matrix
slots through two different serving stacks — 48 cells each, identical prompts, faults,
contexts, judge, iteration budget, and execute gate. **Every difference below is attributable
to the serving infrastructure, not the model.**

| | `qwen3.8-27b_maas-litemaas` | `qwen3.8-27b_maas-rhdp` | Δ |
|---|---|---|---|
| Detect | 28/48 (58%) | 17/48 (35%) | **+23 pts** |
| RCA correct | 27/48 (56%) | 17/48 (35%) | +21 pts |
| Remediation executed | 27/48 (56%) | 16/48 (33%) | +23 pts |
| Detect (clean runs only, no `agent_error`) | 28/34 (**82%**) | 17/27 (**63%**) | +19 pts |
| `agent_error` (all subprocess timeouts) | 14/48 | 21/48 | −7 runs |
| MTTD median | 81s | 93s | −12s |
| MTTR median (n = exec) | 75s (n=27) | 92s (n=16) | −17s |
| MTTR verified median | 106s (n=23) | 156s (n=16) | −50s |

### Exec by fault — where the difference lives

| Gateway | config_corruption | kill_pod | scale_zero |
|---|---|---|---|
| maas-litemaas | 4/16 | 7/16 | 16/16 |
| maas-rhdp | 0/16 | 0/16 | 16/16 |

**Reading:** the two gateways are **identical on the easy fault** (scale_zero: 16/16 exec
both — few rounds needed, any serving stack suffices) and differ **only on the round-hungry
faults** (config_corruption, kill_pod), where the agent must sustain a long tool loop within
the detection window. maas-rhdp's slower responses (MTTD +12s, MTTR +17s per successful
action, 21 vs 14 window-exhaustion timeouts) cause the loop to run out of budget before
diagnosis-and-write completes — 0/32 exec on the hard faults vs 11/32 for litemaas.

**Conclusions:**
1. **Serving infrastructure moves outcomes as much as model choice does** — a 21–23 point
   swing in detect/RCA/exec from the gateway alone, larger than most model-to-model gaps in
   this matrix.
2. Model evaluations that don't record/pin the serving stack are confounded — the "qwen38 is
   worse than qwen36" reading from earlier batches was actually "maas-rhdp is slower than
   litemaas for this model."
3. Deployment guidance: route Qwen3.8-27B via **maas-litemaas**; if maas-rhdp must be used,
   budget more detection-window time (or fewer forced-RAG rounds) to compensate for its
   latency.
4. For future matrices: treat gateway as a first-class experimental variable (the
   `_maas-*` suffix labeling used here makes that explicit), and verify slot → model identity
   at preflight (this went unnoticed for three batches).

## Top performing combinations

### This run — best cells (detect ∧ RCA ∧ exec)

**Full wins:** **87/288** (maas_glm-53-flash 44, qwen3.8-27b_maas-litemaas 27, qwen3.8-27b_maas-rhdp 16)

| Context | Agent | Fault | MTTD | MTTR | MTTR verified |
|---------|-------|-------|------|------|---------------|
| C2 | `maas_glm-53-flash` | scale_zero | 29s | 29s | 53s |
| C2,4 | `maas_glm-53-flash` | scale_zero | 30s | 30s | 46s |
| C1 | `maas_glm-53-flash` | scale_zero | 31s | 31s | 65s |
| C4 | `maas_glm-53-flash` | scale_zero | 32s | 32s | 46s |
| C1,2 | `maas_glm-53-flash` | scale_zero | 32s | 32s | 51s |
| C3,4 | `maas_glm-53-flash` | scale_zero | 32s | 32s | 60s |
| C3 | `maas_glm-53-flash` | scale_zero | 33s | 33s | 90s |
| C0 | `maas_glm-53-flash` | scale_zero | 33s | 33s | 69s |
| C1,2,4 | `maas_glm-53-flash` | scale_zero | 33s | 33s | 56s |
| C1,3 | `maas_glm-53-flash` | scale_zero | 33s | 33s | 60s |
| C2,3 | `maas_glm-53-flash` | scale_zero | 36s | 36s | 97s |
| C1,2,3,4 | `maas_glm-53-flash` | scale_zero | 37s | 37s | 87s |
| C1,4 | `maas_glm-53-flash` | scale_zero | 37s | 37s | 58s |
| C1,3,4 | `maas_glm-53-flash` | scale_zero | 38s | 38s | 91s |
| C1,2 | `qwen3.8-27b_maas-litemaas` | scale_zero | 42s | 42s | 73s |
| C2,3,4 | `maas_glm-53-flash` | scale_zero | 42s | 42s | 90s |
| C0 | `qwen3.8-27b_maas-litemaas` | scale_zero | 44s | 44s | 78s |
| C1,2,3 | `maas_glm-53-flash` | scale_zero | 48s | 48s | 67s |
| C2 | `qwen3.8-27b_maas-litemaas` | scale_zero | 49s | 49s | 76s |
| C3 | `qwen3.8-27b_maas-litemaas` | scale_zero | 50s | 50s | 105s |

### Top CONTEXT_C × agent (RCA rate, 3 faults per cell)

| Rank | Context | Agent | Detect | RCA | Exec |
|------|---------|-------|--------|-----|------|
| 1 | C4 | `maas_glm-53-flash` | 3/3 | 3/3 | 3/3 |
| 2 | C3 | `maas_glm-53-flash` | 3/3 | 3/3 | 3/3 |
| 3 | C2,4 | `maas_glm-53-flash` | 3/3 | 3/3 | 3/3 |
| 4 | C2,3,4 | `qwen3.8-27b_maas-litemaas` | 3/3 | 3/3 | 3/3 |
| 5 | C2,3,4 | `maas_glm-53-flash` | 3/3 | 3/3 | 3/3 |
| 6 | C2,3 | `qwen3.8-27b_maas-litemaas` | 3/3 | 3/3 | 3/3 |
| 7 | C2 | `maas_glm-53-flash` | 3/3 | 3/3 | 3/3 |
| 8 | C1,4 | `maas_glm-53-flash` | 3/3 | 3/3 | 3/3 |
| 9 | C1,3,4 | `maas_glm-53-flash` | 3/3 | 3/3 | 3/3 |
| 10 | C1,3 | `maas_glm-53-flash` | 3/3 | 3/3 | 3/3 |
| 11 | C1,2,3,4 | `maas_glm-53-flash` | 3/3 | 3/3 | 3/3 |
| 12 | C1,2,3 | `maas_glm-53-flash` | 3/3 | 3/3 | 3/3 |
| 13 | C1,2 | `maas_glm-53-flash` | 3/3 | 3/3 | 3/3 |
| 14 | C1 | `maas_glm-53-flash` | 3/3 | 3/3 | 3/3 |
| 15 | C3,4 | `maas_glm-53-flash` | 3/3 | 3/3 | 2/3 |

### Aug 11 top combos (from prior report — working agents only)

| Context | Best RCA agents (Aug) | RCA | Exec highlight |
|---------|----------------------|-----|----------------|
| C3 | DeepSeek, Granite | 8/15 each | Qwen only executor |
| C{1,4} | DeepSeek | 8/15 | — |
| C0 | Granite, DeepSeek | weak RCA (3/15 C0) | — |
| Full set C{1,2,3,4} | DeepSeek | 3/3 detect+RCA (Jul Option2) | 0 exec |

### Context × agent: this run vs Aug 11 legacy batch (comparable agents: deepseek, qwen3, granite)

> Every (context × agent) cell for the three agents present in both batches — **not a "top" ranking**. Aug 11 was a different design size (see comparability note at the top of this report); it is shown for legacy continuity only. Rows are ordered by best current RCA, then current exec.

| Context | Agent | Now RCA | Now exec | Aug 11 (MLflow) |
|---|---|---|---|---|
| C1 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1 | granite | 3/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C1 | qwen3 | 3/3 RCA | 0/3 exec | 1/3 RCA, 1/3 exec |
| C1,4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1,4 | granite | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1,4 | qwen3 | 3/3 RCA | 0/3 exec | 2/3 RCA, 0/3 exec |
| C2 | deepseek | 3/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C2 | granite | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2,3 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2,3 | granite | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2,3,4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2,4 | deepseek | 3/3 RCA | 0/3 exec | 2/3 RCA, 0/3 exec |
| C2,4 | granite | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C2,4 | qwen3 | 3/3 RCA | 0/3 exec | 2/3 RCA, 1/3 exec |
| C3 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C4 | deepseek | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C4 | granite | 3/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1,2,3,4 | deepseek | 2/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1,2,3,4 | granite | 2/3 RCA | 0/3 exec | 3/3 RCA, 0/3 exec |
| C1,2,3,4 | qwen3 | 2/3 RCA | 0/3 exec | 1/3 RCA, 1/3 exec |
| C2,3 | qwen3 | 2/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C2,3,4 | granite | 2/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C2,3,4 | qwen3 | 2/3 RCA | 0/3 exec | 0/3 RCA, 0/3 exec |
| C3 | granite | 2/3 RCA | 0/3 exec | 2/3 RCA, 0/3 exec |
| C4 | qwen3 | 2/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C3 | qwen3 | 1/3 RCA | 0/3 exec | 3/3 RCA, 1/3 exec |
| C0 | deepseek | 0/3 RCA | 0/3 exec | 2/3 RCA, 0/3 exec |
| C0 | granite | 0/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |
| C0 | qwen3 | 0/3 RCA | 0/3 exec | 0/3 RCA, 0/3 exec |
| C2 | qwen3 | 0/3 RCA | 0/3 exec | 1/3 RCA, 0/3 exec |

## MLflow validation

- Harness JSON runs (this batch): **288**
- MLflow runs in window: **287**
- Matched by `run_id`: **287**
- Metric mismatches: **0**
- Window gap explained: the one unmatched harness run (`20261020124111`, deepseek × C0 × scale_zero) has its MLflow record started at the **first aborted kickoff** (2026-10-20 12:41 UTC — 3 min before the resumed kickoff at 12:44 UTC), so it falls outside the audit window. All **288/288** harness runs match an MLflow record when the window is widened to include the aborted kickoff (verified 22 Oct); it is one of the 4 resumed cells noted in [RUN_NOTES_22Oct26.md](RUN_NOTES_22Oct26.md).

## By context (18 runs each = 6 agents × 3 faults)

> **Retriever note:** CONTEXT_C corpora (C1–C4) are served by a **lexical TF-IDF RAG** (sklearn `TfidfVectorizer`, unigrams+bigrams, 50k features, cosine top-k=5) over the git-SHA-pinned OpenTelemetry demo repo, chunked at 900 chars / 120 overlap (`code/tools/rag_context.py`, `INDEX_VERSION=no-changelog-v1`). No embeddings or vector store are involved: retrieval matches keyword overlap only, and the agent formulates its own queries (RAG-as-tool with a forced round-0 phase per enabled corpus). Interpret by-context deltas with this in mind.

| Context | Detect | RCA | Exec | MTTD med | MTTR med | MTTR verified med |
|---|---|---|---|---|---|---|
| C0 | 15/18 (83%) | 6/18 (33%) | 5/18 (28%) | 77s | 65s (n=5) | 146s (n=5) |
| C1 | 15/18 (83%) | 14/18 (78%) | 5/18 (28%) | 75s | 75s (n=5) | 112s (n=4) |
| C2 | 13/18 (72%) | 11/18 (61%) | 5/18 (28%) | 58s | 75s (n=5) | 114s (n=4) |
| C3 | 13/18 (72%) | 11/18 (61%) | 5/18 (28%) | 63s | 72s (n=5) | 117s (n=4) |
| C4 | 15/18 (83%) | 15/18 (83%) | 6/18 (33%) | 64s | 99s (n=6) | 125s (n=4) |
| C1,2 | 15/18 (83%) | 15/18 (83%) | 6/18 (33%) | 98s | 118s (n=6) | 153s (n=5) |
| C1,3 | 14/18 (78%) | 14/18 (78%) | 5/18 (28%) | 78s | 91s (n=5) | 113s (n=4) |
| C1,4 | 14/18 (78%) | 14/18 (78%) | 5/18 (28%) | 96s | 87s (n=5) | 131s (n=4) |
| C2,3 | 14/18 (78%) | 14/18 (78%) | 6/18 (33%) | 87s | 96s (n=6) | 119s (n=4) |
| C2,4 | 14/18 (78%) | 14/18 (78%) | 5/18 (28%) | 69s | 70s (n=5) | 122s (n=4) |
| C3,4 | 15/18 (83%) | 14/18 (78%) | 5/18 (28%) | 89s | 81s (n=5) | 146s (n=5) |
| C1,2,3 | 14/18 (78%) | 13/18 (72%) | 5/18 (28%) | 96s | 125s (n=5) | 153s (n=4) |
| C1,2,4 | 16/18 (89%) | 14/18 (78%) | 5/18 (28%) | 99s | 119s (n=5) | 173s (n=5) |
| C1,3,4 | 14/18 (78%) | 14/18 (78%) | 6/18 (33%) | 87s | 84s (n=6) | 91s (n=5) |
| C2,3,4 | 15/18 (83%) | 14/18 (78%) | 7/18 (39%) | 111s | 152s (n=7) | 199s (n=5) |
| C1,2,3,4 | 15/18 (83%) | 12/18 (67%) | 6/18 (33%) | 101s | 144s (n=6) | 136s (n=4) |

## By agent × context (3 faults each)

One row per (agent, CONTEXT_C). Rates are over the 3 faults; MTTR / MTTR verified medians use cells where the field is present.

| Agent | Context | Detect | RCA | Exec | MTTD med | MTTR med | MTTR verified med |
|---|---|---|---|---|---|---|---|
| `maas_deepseek` | C0 | 3/3 | 0/3 | 0/3 | 82s | — | — |
| `maas_deepseek` | C1 | 3/3 | 3/3 | 0/3 | 44s | — | — |
| `maas_deepseek` | C2 | 3/3 | 3/3 | 0/3 | 48s | — | — |
| `maas_deepseek` | C3 | 3/3 | 3/3 | 0/3 | 45s | — | — |
| `maas_deepseek` | C4 | 3/3 | 3/3 | 0/3 | 46s | — | — |
| `maas_deepseek` | C1,2 | 3/3 | 3/3 | 0/3 | 74s | — | — |
| `maas_deepseek` | C1,3 | 3/3 | 3/3 | 0/3 | 59s | — | — |
| `maas_deepseek` | C1,4 | 3/3 | 3/3 | 0/3 | 83s | — | — |
| `maas_deepseek` | C2,3 | 3/3 | 3/3 | 0/3 | 50s | — | — |
| `maas_deepseek` | C2,4 | 3/3 | 3/3 | 0/3 | 56s | — | — |
| `maas_deepseek` | C3,4 | 3/3 | 3/3 | 0/3 | 57s | — | — |
| `maas_deepseek` | C1,2,3 | 3/3 | 2/3 | 0/3 | 87s | — | — |
| `maas_deepseek` | C1,2,4 | 3/3 | 3/3 | 0/3 | 59s | — | — |
| `maas_deepseek` | C1,3,4 | 3/3 | 3/3 | 0/3 | 94s | — | — |
| `maas_deepseek` | C2,3,4 | 3/3 | 3/3 | 0/3 | 245s | — | — |
| `maas_deepseek` | C1,2,3,4 | 3/3 | 2/3 | 0/3 | 99s | — | — |
| `maas_glm-53-flash` | C0 | 3/3 | 3/3 | 2/3 | 330s | 249s (n=2) | 274s (n=2) |
| `maas_glm-53-flash` | C1 | 3/3 | 3/3 | 3/3 | 121s | 121s (n=3) | 103s (n=2) |
| `maas_glm-53-flash` | C2 | 3/3 | 3/3 | 3/3 | 75s | 75s (n=3) | 102s (n=2) |
| `maas_glm-53-flash` | C3 | 3/3 | 3/3 | 3/3 | 128s | 128s (n=3) | 119s (n=2) |
| `maas_glm-53-flash` | C4 | 3/3 | 3/3 | 3/3 | 105s | 105s (n=3) | 129s (n=2) |
| `maas_glm-53-flash` | C1,2 | 3/3 | 3/3 | 3/3 | 128s | 128s (n=3) | 102s (n=2) |
| `maas_glm-53-flash` | C1,3 | 3/3 | 3/3 | 3/3 | 107s | 107s (n=3) | 93s (n=2) |
| `maas_glm-53-flash` | C1,4 | 3/3 | 3/3 | 3/3 | 77s | 77s (n=3) | 77s (n=2) |
| `maas_glm-53-flash` | C2,3 | 2/3 | 2/3 | 2/3 | 69s | 69s (n=2) | 97s (n=1) |
| `maas_glm-53-flash` | C2,4 | 3/3 | 3/3 | 3/3 | 118s | 118s (n=3) | 95s (n=2) |
| `maas_glm-53-flash` | C3,4 | 3/3 | 3/3 | 2/3 | 295s | 163s (n=2) | 181s (n=2) |
| `maas_glm-53-flash` | C1,2,3 | 3/3 | 3/3 | 3/3 | 125s | 125s (n=3) | 108s (n=2) |
| `maas_glm-53-flash` | C1,2,4 | 3/3 | 3/3 | 2/3 | 102s | 156s (n=2) | 170s (n=2) |
| `maas_glm-53-flash` | C1,3,4 | 3/3 | 3/3 | 3/3 | 73s | 73s (n=3) | 90s (n=2) |
| `maas_glm-53-flash` | C2,3,4 | 3/3 | 3/3 | 3/3 | 152s | 152s (n=3) | 198s (n=2) |
| `maas_glm-53-flash` | C1,2,3,4 | 3/3 | 3/3 | 3/3 | 169s | 169s (n=3) | 138s (n=2) |
| `maas_granite` | C0 | 3/3 | 0/3 | 0/3 | 35s | — | — |
| `maas_granite` | C1 | 3/3 | 3/3 | 0/3 | 584s | — | — |
| `maas_granite` | C2 | 2/3 | 3/3 | 0/3 | 227s | — | — |
| `maas_granite` | C3 | 2/3 | 2/3 | 0/3 | 25s | — | — |
| `maas_granite` | C4 | 3/3 | 3/3 | 0/3 | 27s | — | — |
| `maas_granite` | C1,2 | 3/3 | 3/3 | 0/3 | 33s | — | — |
| `maas_granite` | C1,3 | 3/3 | 3/3 | 0/3 | 37s | — | — |
| `maas_granite` | C1,4 | 3/3 | 3/3 | 0/3 | 227s | — | — |
| `maas_granite` | C2,3 | 3/3 | 3/3 | 0/3 | 25s | — | — |
| `maas_granite` | C2,4 | 3/3 | 3/3 | 0/3 | 28s | — | — |
| `maas_granite` | C3,4 | 3/3 | 3/3 | 0/3 | 227s | — | — |
| `maas_granite` | C1,2,3 | 3/3 | 3/3 | 0/3 | 29s | — | — |
| `maas_granite` | C1,2,4 | 3/3 | 2/3 | 0/3 | 37s | — | — |
| `maas_granite` | C1,3,4 | 3/3 | 3/3 | 0/3 | 73s | — | — |
| `maas_granite` | C2,3,4 | 3/3 | 2/3 | 0/3 | 29s | — | — |
| `maas_granite` | C1,2,3,4 | 3/3 | 2/3 | 0/3 | 99s | — | — |
| `maas_qwen3` | C0 | 3/3 | 0/3 | 0/3 | 55s | — | — |
| `maas_qwen3` | C1 | 3/3 | 3/3 | 0/3 | 65s | — | — |
| `maas_qwen3` | C2 | 3/3 | 0/3 | 0/3 | 100s | — | — |
| `maas_qwen3` | C3 | 3/3 | 1/3 | 0/3 | 69s | — | — |
| `maas_qwen3` | C4 | 3/3 | 2/3 | 0/3 | 64s | — | — |
| `maas_qwen3` | C1,2 | 3/3 | 3/3 | 0/3 | 104s | — | — |
| `maas_qwen3` | C1,3 | 3/3 | 3/3 | 0/3 | 87s | — | — |
| `maas_qwen3` | C1,4 | 3/3 | 3/3 | 0/3 | 95s | — | — |
| `maas_qwen3` | C2,3 | 2/3 | 2/3 | 0/3 | 138s | — | — |
| `maas_qwen3` | C2,4 | 3/3 | 3/3 | 0/3 | 71s | — | — |
| `maas_qwen3` | C3,4 | 3/3 | 2/3 | 0/3 | 89s | — | — |
| `maas_qwen3` | C1,2,3 | 3/3 | 3/3 | 0/3 | 105s | — | — |
| `maas_qwen3` | C1,2,4 | 3/3 | 3/3 | 0/3 | 100s | — | — |
| `maas_qwen3` | C1,3,4 | 2/3 | 2/3 | 0/3 | 98s | — | — |
| `maas_qwen3` | C2,3,4 | 2/3 | 2/3 | 0/3 | 103s | — | — |
| `maas_qwen3` | C1,2,3,4 | 3/3 | 2/3 | 0/3 | 101s | — | — |
| `qwen3.8-27b_maas-litemaas` | C0 | 2/3 | 2/3 | 2/3 | 412s | 412s (n=2) | 436s (n=2) |
| `qwen3.8-27b_maas-litemaas` | C1 | 2/3 | 1/3 | 1/3 | 291s | 50s (n=1) | 83s (n=1) |
| `qwen3.8-27b_maas-litemaas` | C2 | 1/3 | 1/3 | 1/3 | 49s | 49s (n=1) | 76s (n=1) |
| `qwen3.8-27b_maas-litemaas` | C3 | 1/3 | 1/3 | 1/3 | 50s | 50s (n=1) | 105s (n=1) |
| `qwen3.8-27b_maas-litemaas` | C4 | 2/3 | 2/3 | 2/3 | 358s | 358s (n=2) | 92s (n=1) |
| `qwen3.8-27b_maas-litemaas` | C1,2 | 2/3 | 2/3 | 2/3 | 282s | 282s (n=2) | 306s (n=2) |
| `qwen3.8-27b_maas-litemaas` | C1,3 | 1/3 | 1/3 | 1/3 | 70s | 70s (n=1) | 101s (n=1) |
| `qwen3.8-27b_maas-litemaas` | C1,4 | 1/3 | 1/3 | 1/3 | 87s | 87s (n=1) | 167s (n=1) |
| `qwen3.8-27b_maas-litemaas` | C2,3 | 3/3 | 3/3 | 3/3 | 531s | 531s (n=3) | 451s (n=2) |
| `qwen3.8-27b_maas-litemaas` | C2,4 | 1/3 | 1/3 | 1/3 | 70s | 70s (n=1) | 157s (n=1) |
| `qwen3.8-27b_maas-litemaas` | C3,4 | 2/3 | 2/3 | 2/3 | 233s | 233s (n=2) | 263s (n=2) |
| `qwen3.8-27b_maas-litemaas` | C1,2,3 | 1/3 | 1/3 | 1/3 | 179s | 179s (n=1) | 242s (n=1) |
| `qwen3.8-27b_maas-litemaas` | C1,2,4 | 2/3 | 2/3 | 2/3 | 174s | 174s (n=2) | 208s (n=2) |
| `qwen3.8-27b_maas-litemaas` | C1,3,4 | 2/3 | 2/3 | 2/3 | 308s | 308s (n=2) | 342s (n=2) |
| `qwen3.8-27b_maas-litemaas` | C2,3,4 | 3/3 | 3/3 | 3/3 | 350s | 350s (n=3) | 292s (n=2) |
| `qwen3.8-27b_maas-litemaas` | C1,2,3,4 | 2/3 | 2/3 | 2/3 | 223s | 223s (n=2) | 89s (n=1) |
| `qwen3.8-27b_maas-rhdp` | C0 | 1/3 | 1/3 | 1/3 | 65s | 65s (n=1) | 146s (n=1) |
| `qwen3.8-27b_maas-rhdp` | C1 | 1/3 | 1/3 | 1/3 | 75s | 75s (n=1) | 148s (n=1) |
| `qwen3.8-27b_maas-rhdp` | C2 | 1/3 | 1/3 | 1/3 | 87s | 87s (n=1) | 156s (n=1) |
| `qwen3.8-27b_maas-rhdp` | C3 | 1/3 | 1/3 | 1/3 | 72s | 72s (n=1) | 129s (n=1) |
| `qwen3.8-27b_maas-rhdp` | C4 | 1/3 | 2/3 | 1/3 | 93s | 93s (n=1) | 158s (n=1) |
| `qwen3.8-27b_maas-rhdp` | C1,2 | 1/3 | 1/3 | 1/3 | 107s | 107s (n=1) | 197s (n=1) |
| `qwen3.8-27b_maas-rhdp` | C1,3 | 1/3 | 1/3 | 1/3 | 91s | 91s (n=1) | 151s (n=1) |
| `qwen3.8-27b_maas-rhdp` | C1,4 | 1/3 | 1/3 | 1/3 | 292s | 292s (n=1) | 371s (n=1) |
| `qwen3.8-27b_maas-rhdp` | C2,3 | 1/3 | 1/3 | 1/3 | 90s | 90s (n=1) | 142s (n=1) |
| `qwen3.8-27b_maas-rhdp` | C2,4 | 1/3 | 1/3 | 1/3 | 62s | 62s (n=1) | 99s (n=1) |
| `qwen3.8-27b_maas-rhdp` | C3,4 | 1/3 | 1/3 | 1/3 | 81s | 81s (n=1) | 146s (n=1) |
| `qwen3.8-27b_maas-rhdp` | C1,2,3 | 1/3 | 1/3 | 1/3 | 103s | 103s (n=1) | 157s (n=1) |
| `qwen3.8-27b_maas-rhdp` | C1,2,4 | 2/3 | 1/3 | 1/3 | 436s | 119s (n=1) | 173s (n=1) |
| `qwen3.8-27b_maas-rhdp` | C1,3,4 | 1/3 | 1/3 | 1/3 | 95s | 95s (n=1) | 189s (n=1) |
| `qwen3.8-27b_maas-rhdp` | C2,3,4 | 1/3 | 1/3 | 1/3 | 111s | 111s (n=1) | 199s (n=1) |
| `qwen3.8-27b_maas-rhdp` | C1,2,3,4 | 1/3 | 1/3 | 1/3 | 118s | 118s (n=1) | 183s (n=1) |

## Why agents stop before remediation execution (exec-gap decomposition)

Detect is 80% but exec is 30%. The gap is concentrated in the three diagnosers (deepseek,
granite, qwen3: 0/48 exec each). Verifying against the MLflow round/tool-call artifacts,
the gap decomposes into **three distinct mechanisms** — only one of which is a capability gap:

### Mechanism 1 — Confabulated action (deepseek, granite)

Their final JSONs *describe* fixes in **past tense** — e.g. `"suggested_remediations":
["Restarted deployment 'cart-service-deployment' due to CrashLoopBackOff…"]` — narrating an
action as already performed **without calling any tool**. The execute gate blocks the final
JSON and nudges ("You MUST call one of: scale_deployment, restart_deployment, delete_pod");
the models respond by re-emitting essentially the same JSON. Deepseek: 46/48 runs
gate-blocked (34 exhausted all 3 nudges); granite: 48/48 (32 exhausted). After the nudge cap
the gate yields (by design) → `remediation_executed=false`. Past-tense remediation text
appears in 25/48 (deepseek) and 32/48 (granite) runs. Note the diagnosis itself is right
(RCA 90%/85%, proposed fix correct 44/48 and 45/48) — the models *know* the fix but emit it
as prose instead of a tool call.

### Mechanism 2 — Namespace guessing (all three diagnosers, when they DO act)

qwen3 complies with the gate — it called a write tool in **43/48 runs**, mostly correctly
targeted (`scale_deployment(cart)` 30×) — yet every call failed. The reason: **all 73
write-tool calls by the three diagnosers passed `namespace: "default"`**, and the demo runs
in `otel-demo` → K8s API 404 ("no such deployment in default") → write fails → no exec.
The executors never hit this: glm passed `namespace: "otel-demo"` in 48/48 calls (all OK);
qwen3.8-27b on both gateways *omitted* the parameter (24+17 calls, all OK — omission falls
back to the harness's target namespace).

| Agent | Write calls | Namespace used | Outcome |
|---|---|---|---|
| glm-53-flash | 48 | `otel-demo` (explicit, every call) | 48/48 OK |
| qwen3.8-27b_maas-litemaas | 28 | omitted (→ otel-demo) | 28/28 OK |
| qwen3.8-27b_maas-rhdp | 17 | omitted (→ otel-demo) | 17/17 OK |
| qwen3 | 45 | **`default`** | 45/45 **404** |
| deepseek | 14 | **`default`** (13), garbage (1) | 14/14 **404** |
| granite | 14 | **`default`** | 14/14 **404** |

**Interface root cause:** the tool schema documents the parameter as
`"namespace": "Kubernetes namespace (default from harness env)"` — it hints a default exists
but never names it, and the system prompt never states the workload namespace. Models that
don't know it fill in the strongest prior from training data: `"default"`. The executors
either knew or omitted; the diagnosers guessed. **qwen3's 0/48 exec is therefore not a
capability gap** — with the namespace resolved, its 30 correctly-targeted `scale_deployment`
calls would have landed (mostly on scale_zero cells).

### Mechanism 3 — Invented deployment names (minor)

A minority of the failing calls target names that don't exist in `otel-demo`
(`cart-service`, `web-svc`, `vault-backend`) — plausible-sounding names from the model's
prior rather than observed cluster state. Secondary to the namespace issue: most targeting
(`cart`) was correct.

### Fixes implied

1. **Tool schema**: state the namespace explicitly (`"omit to target the workload namespace
   (otel-demo)"`) or drop the parameter entirely so the harness default always applies.
2. **Self-healing error**: on 404, return `"no deployment 'cart' in namespace 'default' —
   target namespace is otel-demo"` so agents can retry correctly within the same run.
3. **Prompt**: state the workload namespace once in the system prompt.
4. For confabulators (deepseek/granite): nudge text could explicitly reject past-tense
   claims ("you have not called any remediation tool; describing a restart is not
   performing one").

## Agent errors

- `qwen3.8-27b_maas-rhdp`: 21/48
- `qwen3.8-27b_maas-litemaas`: 14/48
- `maas_qwen3`: 3/48
- `maas_glm-53-flash`: 1/48

`agent_error` **rose 21 → 39** vs Oct 10–12 and is concentrated on the two slow faults (config_corruption 19, kill_pod 20, scale_zero 0 — see [By fault](#by-fault-96-runs-each)). All observed errors are **timeouts** (316s–900s), i.e. the qwen3.8-27b gateway paths running out of budget on the faults that need more rounds — an infra/reliability signal, not a capability signal, but it depresses the headline by-agent rates for those two models.

Verified against the raw `agent_error` strings (39/39): every one is a `subprocess timeout` — **no API/connection failures, no empty stdout, no JSON parse failures**. 22 of 39 hit the full **900s** window cap (agent still working when the detection window expired); the other 17 hit the remaining-window boundary (316s–848s). These are slow-endpoint/long-loop budget exhaustions, not crashes. Per agent: qwen3.8-27b_maas-rhdp 21, qwen3.8-27b_maas-litemaas 14, qwen3 3, glm-53-flash 1 — all timeouts.

### Error-adjusted view (rates over runs with no `agent_error`)

Headline by-agent rates count error runs as non-detect/non-RCA/non-exec. On clean runs only:

| Agent | Clean runs | Detect | RCA | Exec |
|---|---|---|---|---|
| `maas_deepseek` | 48 | 48/48 (100%) | 43/48 (90%) | 0/48 |
| `maas_glm-53-flash` | 47 | 47/47 (100%) | 47/47 (100%) | 44/47 (94%) |
| `maas_granite` | 48 | 46/48 (96%) | 41/48 (85%) | 0/48 |
| `maas_qwen3` | 45 | 45/45 (100%) | 34/45 (76%) | 0/45 |
| `qwen3.8-27b_maas-litemaas` | 34 | 28/34 (**82%**) | 27/34 (79%) | 27/34 (79%) |
| `qwen3.8-27b_maas-rhdp` | 27 | 17/27 (**63%**) | 17/27 (63%) | 16/27 (59%) |

maas-litemaas's "58% detect" headline is **82%** on clean runs and maas-rhdp's "35%" is **63%** — the diagnosers' advantage narrows materially once infra reliability is separated from capability. Read both tables together: the clean-run view is the capability signal, the error count is the reliability/ops signal (a deployment pick needs both).

---

## Recommendations

1. **Default RCA pair:** **maas_glm-53-flash** + **maas_deepseek** for diagnosis; pair with **maas_glm-53-flash** / **qwen3.8-27b_maas-litemaas** when execution is required.
2. **Executors this run:** **maas_glm-53-flash**, **qwen3.8-27b_maas-litemaas**.
3. **Llama:** Dropped from default lineup after 0% detect (optional agent code remains).
4. **Context:** Prefer high-RCA contexts from the by-context / agent×context tables.
5. **Re-add omitted models** (Nemotron/Kimi) only after MaaS preflight is green.
6. **`config_corruption` needs a real write path (carried from Oct 12 rec, still open):** implement `patch_deployment` in `execute_tool` (it is already advertised in the execute-gate tool set and counted as action-appropriate in scoring but does not exist), or change the fault's success criterion. Until then, report **recovery ceiling 192/288 (67%)** for this design and treat the 17 config_corruption execs as attempted-but-unwinnable rather than capability signal.
   > **✅ RESOLVED 23 Oct 26:** `patch_deployment` + `get_deployment_config` implemented and verified — a follow-up 40-cell matrix (`config_corruption` × 5 agents × 8 contexts) produced **16/40 verified recoveries vs 0/96 here**, with guardrails against hallucinated-arg patches. See [EVALUATION_RESULTS_ccw_23oct26.md](EVALUATION_RESULTS_ccw_23oct26.md) and [HARNESS_DESIGN.md](HARNESS_DESIGN.md). The recovery ceiling no longer applies.
7. **`kill_pod` scoring (carried from Oct 12 rec, still open):** recovery is 22/22 = exec with an 18.5s median write→recovery lag while the deployment controller self-heals the pod — either score it against a criterion the agent controls (e.g. correct diagnosis + targeted action) or separate self-healing from agent-attributable recovery.
8. **Report error-adjusted rates alongside headline rates:** qwen36/38 detect is 82%/63% on clean runs vs 58%/35% headline — the current tables conflate MaaS endpoint reliability (39 timeouts, all on non-scale_zero faults) with model capability. Investigate the qwen36/38 timeouts before the next batch (sub-900s values suggest per-call/iteration limits, not the 900s agent budget).
9. **Latency trend hygiene:** Sep 25–27 MTTD/MTTR were produced by the pre-fix pipeline; re-derive them with the fixed logic (as done for Oct 10–12 in `scripts/recompute_corrected_metrics.py`) before drawing any MTTD/MTTR trend conclusions across batches.
