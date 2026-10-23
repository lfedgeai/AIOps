# Evaluation Results — October 2026 (consolidated)

**Generated:** 2026-10-23 · **Covers:** all evaluation runs from 22 Oct 26 forward  
**Runs combined:** Option-2 matrix (Oct 20–22, 288 cells) + `config_corruption` winnability matrix (Oct 22–23, 40 cells)  
**Tracking:** local MLflow `agentic_aiops_mttd_mttr` — both runs audited, **0 metric mismatches**  
**Detail reports:** [EVALUATION_RESULTS_22oct26.md](EVALUATION_RESULTS_22oct26.md) (Run A) · [EVALUATION_RESULTS_ccw_23oct26.md](EVALUATION_RESULTS_ccw_23oct26.md) (Run B) · [HARNESS_DESIGN.md](HARNESS_DESIGN.md) (harness design rationale + learning log)

> **How to read this report.** Run A is the regular Option-2 matrix on the *fixed* metric pipeline. Run B is a focused follow-up that added the `patch_deployment`/`get_deployment_config` tool path and re-tested one fault. They share the harness core and every environment knob; the only intended difference is the tool set (§3.2). October numbers before 22 Oct (09 Oct, 10–12 Oct batches) appear here only as baselines — their headline latency metrics were produced by the buggy pipeline (§3.1).

---

## 1. Executive summary — October in one table

| Metric | Run A: Option-2 (n=288) | Run B: ccw winnable (n=40) |
|---|---|---|
| Detection | 231/288 (80%) | 40/40 (100%) |
| RCA correct | 209/288 (73%) | 33/40 (83%) |
| Remediation executed | 87/288 (30%) | 16/40 (40%) |
| **Recovery verified** | 70/288 (24%) | **16/40 (40%)** |
| Full wins (det∧RCA∧exec∧recovery) | 87/288 | 16/40 |
| MTTD median (detected) | 87 s | 41–58 s (per agent) |
| `agent_error` | 39/288 | 0/40 |

**Headline findings:**

1. **glm-53-flash is the first near-complete agent in the series** — Run A: 47/48 detect, 47/48 RCA, 44/48 exec; Run B: **8/8 full wins** on the previously-unwinnable fault, config MTTD 154 s → 41 s.
2. **qwen36-27b joins it as the second executor** — Run B: 8/8 full wins with zero timeouts (vs 29% `agent_error` in Run A), because the config-fix path needs few rounds.
3. **The diagnosers still don't execute** — deepseek, qwen3, granite: 0 exec in both runs, even with the exact fix path available and the corrupted config visible. The diagnose→execute gap is behavioral, not tool-availability.
4. **`config_corruption` went from structurally unwinnable (0/96 recovery in Run A) to 16/40 verified recoveries in Run B** — via the new tool path (§3.2).
5. **The Oct 10–12 → Oct 20–22 jumps are environment-driven, not model-driven**: healthy kafka + ~100× telemetry volume removed decoy errors that specifically punished act-first agents (write-targeting evidence in §4.3).

## 2. Timeline of what happened, and why

| Date | Event |
|---|---|
| 10–12 Oct | Option-2 batch run on the **buggy metric pipeline** (MTTD/MTTR measurement bugs) |
| 19 Oct | 6 metric bugs diagnosed and fixed by replay over the recorded batch → [EVALUATION_RESULTS_12oct26_corrected.md](EVALUATION_RESULTS_12oct26_corrected.md) |
| 20–22 Oct | **Run A**: first full 288-cell matrix on the fixed pipeline; kafka healthy, load-generator restored |
| 22 Oct | Run A analyzed, verified against raw JSONs + MLflow, report patched (by-fault cut, error-adjusted rates, environment caveats) |
| 22 Oct | `patch_deployment` + `get_deployment_config` implemented and live-tested (guardrail bugs found and fixed, §3.2) |
| 22–23 Oct | **Run B**: 40-cell winnability matrix (killed once by session churn mid-run; resumed cleanly via the resume model) |
| 23 Oct | Run B results; October consolidated (this report) |

## 3. Bug fixes and the reasons behind them

### 3.1 Metric-pipeline fixes (why October numbers are trustworthy)

**Root cause:** the harness defined a JSON-over-stdout protocol between agent and harness, then later grew an interactive control layer around it — execute-gate nudges, forced-RAG notices, context-usage reporting. Those features printed to **stdout** by default, corrupting the very channel the metrics parsed. Independently, metric derivation had **two parallel code paths** (MTTR in `run_harness.py`, exec scoring in `evaluation.py`) that drifted apart. Six bugs, all in *derivation from correctly-recorded raw data* — fixed 19 Oct, first full run 20–22 Oct:

| # | Bug | Why it existed | What it distorted | Fix |
|---|---|---|---|---|
| 1–3 | Gate nudge, RAG notice, context-window report printed to **stdout** | control layer added after the protocol was defined; `print()` defaults to stdout | agent JSON stdout unparsable → bug 4 triggered on every run | all diagnostics → **stderr** |
| 4 | MTTD parsed stdout with strict `json.loads`; on any noise it **silently fell back to the subprocess-exit clock** | no failure signal on the fallback path | MTTD ≈ end-of-run + post-loop MLflow uploads — glm inflated ~60 s | one tolerant parser (`parse_agent_signals`) everywhere |
| 5 | `detection_timestamp` stamped only at the **first successful write** | detection and remediation were assumed to coincide | declare-then-act agents' detection claims never stamped (detection ≡ remediation) | stamp at first `detected=true` claim surviving gate-blocked rounds |
| 6 | `mttr_seconds` keyed on **any** successful write, while exec required *targeted + action-appropriate* | two selectors for the same concept, maintained separately | MTTR present with exec=false; wrong-target writes skewed medians late | one shared selector `first_effective_remediation_write` drives both → **MTTR ⟺ exec by construction** |

**Net effect on reported numbers:** Run A's latency columns are cross-agent comparable for the first time; detection/RCA/exec/recovery counts of the Oct 10–12 batch were unaffected (the bugs were in derivation, not recording). Prior trend lines (Sep) still need re-derivation before latency comparisons.

### 3.2 Tool-path fixes (making `config_corruption` winnable)

**The structural bug:** the fault injects a bad `DATABASE_HOST` env var into the `cart` deployment *spec*. The app never reads it (pod stays Running), so it was invisible in logs — **0/96 agent outputs ever referenced it** — and the only write tools (restart/scale/delete) recreate pods with the same corrupt spec. Recovery 0/96 was by construction, capping the whole matrix's recovery ceiling at 192/288 (67%). Reason: the tool set covered *runtime* remediations but had no *spec/config* remediation, and no read path exposed deployment config.

**The fix (Run B's experiment axis):**

| Tool | Purpose | Why needed |
|---|---|---|
| `get_deployment_config` (read) | exposes image, replicas, per-container env | makes the corrupted config **discoverable** |
| `patch_deployment` (write) | remove/override env vars via read-modify-write → rolling update | makes the fault **fixable**; restart/scale cannot |

Three guardrails were added during validation, each closing an observed failure mode:

| Guardrail | Bug observed | Reason behind the fix |
|---|---|---|
| **G1** `set_env` may only override **existing** vars | an agent hallucinated `set_env={"CORRECT_VAR": "value"}` — the junk var was *added to the live deployment*, survived harness recovery, and scored as a successful remediation | a write tool consuming arbitrary LLM strings must limit its effects to *observed* state; hallucinated names must not mutate the cluster or earn exec credit |
| **G2** no-op patches return `Error` (failed write) | an empty/mismatched patch would otherwise count as `remediation_executed` | scoring counts successful writes; a write that changes nothing is not a remediation |
| **G3** 409-Conflict retry (4×, backoff) | patching during a prior patch's rollout fails — the rollout controller bumps `resourceVersion` between read and write | legitimate patch → verify → re-patch sequences must not fail spuriously |

**Result:** 0/96 → **16/40 verified recoveries**, every one a genuine `patch_deployment` removing both injected vars, verified *before* harness auto-recovery (so only the agent's own fix counts). See §5.

### 3.3 Operational fix (supervision)

Run B was killed mid-cell on its first launch: a batch started as a session background shell is a child of the session's process runner and is reaped when the session refreshes. The resume model recovered it without duplicates or gaps, and the design requirement is now explicit: matrix runs are launched detached (`setsid`) or under a **systemd user unit** (the existing production pattern, which also orders MLflow/ClickHouse dependencies and isolates the kubeconfig). Details: [HARNESS_DESIGN.md](HARNESS_DESIGN.md) (P10, §6–7).

## 4. Run A — Option-2 matrix (288 cells, Oct 20–22)

First full matrix on the fixed metric pipeline. Full tables: [EVALUATION_RESULTS_22oct26.md](EVALUATION_RESULTS_22oct26.md).

### 4.1 By agent (48 cells each)

| Agent | Detect | RCA | Exec | Recovery | MTTD med | Clean-run detect* |
|---|---|---|---|---|---|---|
| `maas_deepseek` | 48/48 | 43/48 (90%) | 0 | 0 | 64 s | 100% |
| `maas_glm-53-flash` | 47/48 (98%) | 47/48 (98%) | 44/48 (92%) | 31/44 | 118 s | 100% |
| `maas_granite` | 46/48 (96%) | 41/48 (85%) | 0 | 0 | 36 s | 96% |
| `maas_qwen3` | 45/48 (94%) | 34/48 (71%) | 0 | 0 | 92 s | 100% |
| `maas_qwen36-27b` | 28/48 (58%) | 27/48 (56%) | 27/48 (56%) | 23/27 | 81 s | **82%** |
| `maas_qwen38-27b` | 17/48 (35%) | 17/48 (35%) | 16/48 (33%) | 16/16 | 93 s | **63%** |

\* rates over runs with no `agent_error`. qwen36/qwen38 carried 14/48 and 21/48 timeouts respectively (all on the two slow faults); their headline rates conflate endpoint reliability with capability — the clean-run column is the capability signal.

### 4.2 By fault (96 cells each)

| Fault | Detect | RCA | Exec | Recovery | Errors | MTTD med |
|---|---|---|---|---|---|---|
| `scale_zero` | 96/96 | 90/96 | 48/96 (50%) | 48/96 | **0** | 64 s |
| `kill_pod` | 68/96 (71%) | 61/96 (64%) | 22/96 (23%) | 22/96 | 20 | 112 s |
| `config_corruption` | 67/96 (70%) | 58/96 (60%) | 17/96 (18%) | **0/96** | 19 | 95 s |

- Exec spread beyond `scale_zero` (config 1/96 → 17/96, kill 1/96 → 22/96 vs Oct 10–12) — environment-driven.
- `config_corruption` recovery 0/96 was structural (§3.2); `kill_pod` recovery is partly platform self-healing (exec=recovery 22/22, median write→recovery lag 18.5 s).
- All 39 `agent_error`s sit on the two slow faults — the faults needing more rounds are where slow endpoints time out.

### 4.3 Why the numbers moved vs Oct 10–12 (environment, not capability)

The kafka broker was crash-looping during Oct 10–12 (decoy kafka/checkout errors) and the load-generator was starved (~2–25 checkout spans/hour); Run A had a healthy kafka and ~1,300–3,200 spans/hour. Write-targeting evidence for the agent that jumped most (glm: 48%→98% RCA, 35%→92% exec): its Oct 10–12 writes went **kafka 59 / checkout 10 / cart 17** (it was remediating the decoy); in Run A they went **cart 34 / valkey-cart 13 / kafka 0**. The Oct 10–12 environment specifically punished act-first agents; diagnosers were far less affected (deepseek 85%→90% RCA). Deltas vs Sep 25–27 similarly mix capability with task difficulty and are **not** a pure model bake-off.

## 5. Run B — `config_corruption` winnability matrix (40 cells, Oct 22–23)

Harness design and resume model: [HARNESS_DESIGN.md](HARNESS_DESIGN.md). Full results: [EVALUATION_RESULTS_ccw_23oct26.md](EVALUATION_RESULTS_ccw_23oct26.md).

| Agent | Detect | RCA | Exec | Recovery | MTTD med | MTTR med | Recovered med |
|---|---|---|---|---|---|---|---|
| `maas_deepseek` | 8/8 | 7/8 | 0/8 | 0/8 | 89 s | — | — |
| `maas_qwen3` | 8/8 | 6/8 | 0/8 | 0/8 | 89 s | — | — |
| `maas_qwen36-27b` | **8/8** | **8/8** | **8/8** | **8/8** | 58 s | 58 s | 84 s |
| `maas_granite` | 8/8 | 4/8 | 0/8 | 0/8 | 46 s | — | — |
| `maas_glm-53-flash` | **8/8** | **8/8** | **8/8** | **8/8** | 41 s | 41 s | 58 s |

- **16/40 full wins, all by the two executors, at every context including C0** — context neither helps nor hinders the winnable path.
- Diagnosis is faster via config inspection: glm's config-fault MTTD **154 s → 41 s** vs Run A.
- qwen36's redemption: 0 errors here vs 29% in Run A — the config path needs few rounds, so its slower endpoint never times out.
- deepseek *saw* the real env vars via `get_deployment_config` (Oct forensics) yet still fails to convert discovery into a targeted write — and its one Run-A "exec" was the hallucinated-args loophole closed by G1 (the re-run cell scored an honest 0).

**Caveat:** 14 of 40 cells were carried over from Run B's first (killed) launch, which ran pre-guardrail tooling and with an inert junk env var visible on cart; no carried-over cell contains a successful patch write, so outcomes are unaffected (detail in the Run B report).

## 6. Combined October picture

1. **Executor pair:** `maas_glm-53-flash` (fastest, most complete) + `maas_qwen36-27b`. **Diagnoser pair:** `maas_deepseek` + `maas_qwen3`/`maas_granite`. Pairing recommendation unchanged from the Sep/Oct reports, now with config-fix capability proven.
2. **"qwen36 vs qwen38" is a serving-infrastructure story, not a model story:** the `maas_qwen36-27b` agent actually serves **Qwen3.8-27B** on a single-model LiteMaaS endpoint (env var name is historical); `maas_qwen38-27b` serves `qwen38-27b` on the shared RHDP platform (also hosting the judge + two other matrix agents) at ~2–3× lower token throughput → 44% timeouts vs 29% → depressed headline rates. On clean runs the gap narrows (63% vs 82% detect).
3. **Environment sensitivity is the biggest confound in the series:** act-first agents swing tens of points with cluster health (§4.3). Cross-batch deltas must be read with the environment table, not as capability trends.
4. **The C0 context cliff persists for diagnosers** (RCA ~33% at C0 vs 61–83% with corpora in Run A; 2/5 vs 4–5/5 in Run B) — but the executors win at C0 regardless.
5. **Latency columns are trustworthy from 22 Oct onward only** (§3.1); Sep/Oct-early MTTD/MTTR need re-derivation before any trend claim.

## 7. Recommendations (consolidated)

1. **Pairing:** deepseek + glm-53-flash for diagnosis-first; glm-53-flash / qwen36-27b when execution is required.
2. **Adopt the config tool path in the main matrix** — the 192/288 recovery ceiling is lifted; re-include `config_corruption` recovery in headline metrics.
3. **Extend `patch_deployment` to the other spec-level faults** (`memory_limit`, `readiness_probe_fail`) — same problem class.
4. **Re-test qwen38-27b on the config fault** (cheap in rounds; its endpoint penalty may not matter) and investigate the qwen36/38 sub-900s timeouts.
5. **Report error-adjusted rates alongside headline rates** (Run A showed 35% vs 63% clean-run detect for qwen38).
6. **Latency trend hygiene:** re-derive Sep 25–27 MTTD/MTTR with the fixed pipeline before drawing cross-month trends.
7. **`kill_pod` scoring:** separate self-healing from agent-attributable recovery, or score against a criterion the agent controls.
8. **Supervision:** run all future matrices under systemd user units (dependency-ordered, kubeconfig-isolated), not session-attached shells.
9. **Relabel `maas_qwen36-27b`** (e.g. `maas_qwen38-27b-litemaas`) to stop the model-identity confusion in future reports.

## 8. Validation and audit trail

| Check | Run A (n=288) | Run B (n=40) |
|---|---|---|
| Harness JSONs ↔ MLflow matched by `run_id` | 288/288 | 40/40 |
| Metric mismatches | 0 | 0 |
| Independent recomputation from raw JSONs | all tables verified | all tables verified |
| Orphans / invalidations | 1 window-boundary orphan (explained in Run A report) | 1 killed-cell orphan marked FAILED; 1 loophole-scored cell invalidated and re-run |
