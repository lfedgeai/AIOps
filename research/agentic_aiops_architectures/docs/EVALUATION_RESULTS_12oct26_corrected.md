# Evaluation Results — Option 2 matrix (6 agents) — CORRECTED METRICS

**Generated:** 2026-10-19 (correction pass over the Oct 10–12 batch)
**Design:** Option 2 — scenario **A** × all CONTEXT_C subsets (16) × **6 agents** × 3 faults
**Runs:** **288/288** (kickoff `2026-10-10 22:44:19` local, same batch as [EVALUATION_RESULTS_12oct26.md](EVALUATION_RESULTS_12oct26.md))
**Variant:** `cart` · **Forced RAG:** on for C≠0 · **Execute gate:** on
**Lineup:** `maas_deepseek`, `maas_glm-53-flash`, `maas_granite`, `maas_qwen3`, `maas_qwen36-27b`, `maas_qwen38-27b`

> **What is this document?** The published Oct 12 report contained **measurement bugs in MTTD/MTTR derivation** (detection/remediation timestamps), not in the underlying trial data. This report recomputes MTTD/MTTR for the same 288 runs using the fixed metric logic (applied to the harness on 19 Oct 26). Detection, RCA, remediation-executed, and recovery results are **unchanged** — only the latency metrics and their sample sizes change. Verification checks are in §8.

---

## 1. How these numbers were produced

All four bugs were in **metric derivation from correctly-recorded raw data** (agent stdout with signal timestamps; per-write records with tool, arguments, success, timestamps). The raw data needed by the fixed logic is present in every run JSON, so the corrected metrics are obtained by **replaying the fixed derivation** over the recorded batch:

- **MTTD (fixed)** = robust-parsed `signals.detection_timestamp` − `fault_injection_time` — the same tolerant parser the harness result pipeline uses.
- **MTTR (fixed)** = completion time of the **first successful write that is targeted at the expected component AND action-appropriate for the fault type** — the identical selector now shared with `remediation_executed` scoring (`first_effective_remediation_write`).

A live re-run of the matrix was not possible at correction time: the `otel-demo` namespace is scaled to 0/0 and the `agentic-aiops` ClickHouse is crash-looping (startup probe refused; see §9). Re-running live requires restoring that stack first; the recomputation is exactly equivalent for these metrics because it consumes the same inputs the fixed harness consumes.

**Known conservative bias:** the agent's `detection_timestamp` was stamped at the first successful *write* (or end of run when the agent never wrote). The fixed code additionally stamps at the first mid-loop `detected=true` **claim** (surviving later gate-blocked rounds). For declare-then-act agents whose claim preceded their write, future live runs will report slightly **smaller** MTTD than the values here; these values are upper bounds. MTTR values are exact.

## 2. Corrected metric definitions

| Metric | Meaning |
| --- | --- |
| **Detect** | Agent recognized a fault within the detection window (unchanged) |
| **RCA correct** | Right failing component / root cause (unchanged) |
| **Remediation correct (OR)** | Appropriate fix proposed (judged) (unchanged) |
| **Remediation executed (Exec)** | Successful write, targeted at the expected component, action-appropriate for the fault (unchanged definition) |
| **Recovery verified** | Target workload healthy after the agent's actions (unchanged) |
| **MTTD** | fault → agent-recorded detection timestamp (first `detected=true` claim or first successful write, whichever the agent reached first), parsed with the tolerant parser |
| **MTTR** | fault → **first successful, targeted, action-appropriate write** — present exactly when `remediation_executed` is true |
| **MTTR verified** | fault → recovery verified (unchanged) |

Invariant now guaranteed by construction: **MTTR exists ⟺ Exec is true**, and **MTTD ≤ MTTR per run**.

## 3. Executive summary — published vs corrected

| Metric | Published (12 Oct) | **Corrected** |
| --- | --- | --- |
| Detection | 247/288 (86%) | 247/288 (86%) — unchanged |
| RCA correct | 175/288 (61%) | 175/288 (61%) — unchanged |
| Remediation correct (OR) | 203/288 (70%) | 203/288 (70%) — unchanged |
| Remediation executed | 50/288 (17%) | 50/288 (17%) — unchanged |
| Recovery verified | 49/288 (17%) | 49/288 (17%) — unchanged |
| MTTD median (detected) | 117s | **108s** (n=247) |
| MTTR median | 105s (n=105) | **77s (n=50)** |
| MTTR verified median | 128s | 128s (n=49) — unchanged |
| `agent_error` | 21/288 | 21/288 — unchanged |

The published MTTR median mixed 105 runs (any successful write, incl. wrong-target/wrong-action) with the 50 exec runs; the corrected median is over exactly the 50 fault-appropriate remediations.

### Headline takeaways (corrected)

1. **Detection vs execution:** detect 86%, exec 17% — the diagnosis/execution split stands and was not a measurement artifact.
2. **glm-53-flash was materially faster than reported:** MTTD **120s → 57s**, MTTR **57s (n=45) → 38s (n=17)**. Its published MTTD was the harness clock at subprocess exit (stdout parse failure), not the agent's detection moment.
3. **qwen36-27b MTTR was overstated:** **117s (n=35) → 67s (n=16)** — its wrong-target/wrong-action writes were included and skewed late.
4. **qwen3's phantom MTTR is gone:** 93s (n=1) with 0/48 exec → **—** (no fault-appropriate write).
5. **Execution is a one-fault story:** 48 of 50 execs are `scale_zero`; `config_corruption` 1/96, `kill_pod` 1/96.
6. **Diagnosers never write:** DeepSeek (RCA 41/48) and Granite (RCA 37/48) executed 0 appropriate writes; their MTTD is essentially end-of-run time (no detection-then-act split exists).

## 4. By agent (48 runs each)

| Agent | Detect | RCA | Exec | MTTD med (corr) | MTTD med (publ) | MTTR med (corr, n) | MTTR med (publ, n) | MTTR verified med |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `maas_deepseek` | 48/48 (100%) | 41/48 (85%) | 0/48 | **102s** | 102s | — | — | — |
| `maas_glm-53-flash` | 45/48 (94%) | 23/48 (48%) | 17/48 (35%) | **57s** | 120s | **38s (n=17)** | 57s (n=45) | 81s (n=17) |
| `maas_granite` | 47/48 (98%) | 37/48 (77%) | 0/48 | **71s** | 72s | — | — | — |
| `maas_qwen3` | 48/48 (100%) | 40/48 (83%) | 0/48 | **124s** | 125s | — | 93s (n=1) | — |
| `maas_qwen36-27b` | 35/48 (73%) | 17/48 (35%) | 16/48 (33%) | **117s** | 117s | **67s (n=16)** | 117s (n=35) | 115s (n=16) |
| `maas_qwen38-27b` | 24/48 (50%) | 17/48 (35%) | 17/48 (35%) | **158s** | 158s | **133s (n=17)** | 158s (n=24) | 249s (n=16) |

**MTTR n now equals the Exec count for every agent** (17/16/17 for the executors; 0 elsewhere).

### Why MTTD changed for some agents and not others

The old MTTD path re-parsed agent stdout with a strict `json.loads`; any diagnostic line on stdout (execute-gate nudge, forced-RAG notice, context-window report) made it fail silently and fall back to **the harness clock at subprocess exit**. Share of runs whose stdout strictly parsed (old path healthy):

| Agent | Strict-parse OK | Consequence in published report |
| --- | --- | --- |
| `maas_deepseek` | 0/48 | every MTTD = exit clock (≈ end-of-run; small error since agent exits ~1s after stamping) |
| `maas_glm-53-flash` | 0/48 | every MTTD = exit clock (**~60s inflation**: post-loop MLflow/artifact uploads) |
| `maas_granite` | 0/48 | every MTTD = exit clock (small error) |
| `maas_qwen3` | 17/48 | mixed |
| `maas_qwen36-27b` | 36/48 | detected runs mostly clean → MTTD unchanged |
| `maas_qwen38-27b` | 42/48 | detected runs mostly clean → MTTD unchanged |

## 5. By context (18 runs each = 6 agents × 3 faults)

| Context | Detect | RCA | Exec | MTTD med (corr) | MTTR med (corr, n) | MTTD med (publ) | MTTR med (publ) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | 15/18 (83%) | 3/18 (17%) | 3/18 | 98s | 83s (n=3) | 104s | 93s (n=7) |
| C1 | 16/18 (89%) | 11/18 (61%) | 3/18 | 99s | 55s (n=3) | 102s | 96s (n=7) |
| C2 | 15/18 (83%) | 11/18 (61%) | 3/18 | 137s | 53s (n=3) | 137s | 97s (n=6) |
| C3 | 16/18 (89%) | 12/18 (67%) | 3/18 | 92s | 55s (n=3) | 96s | 157s (n=7) |
| C4 | 15/18 (83%) | 11/18 (61%) | 3/18 | 87s | 57s (n=3) | 133s | 104s (n=6) |
| C1-2 | 16/18 (89%) | 13/18 (72%) | 3/18 | 105s | 108s (n=3) | 113s | 123s (n=7) |
| C1-3 | 15/18 (83%) | 12/18 (67%) | 3/18 | 108s | 65s (n=3) | 108s | 113s (n=6) |
| C1-4 | 14/18 (78%) | 11/18 (61%) | 3/18 | 127s | 62s (n=3) | 136s | 97s (n=6) |
| C2-3 | 14/18 (78%) | 11/18 (61%) | 3/18 | 99s | 56s (n=3) | 109s | 92s (n=5) |
| C2-4 | 15/18 (83%) | 12/18 (67%) | 3/18 | 103s | 62s (n=3) | 104s | 82s (n=6) |
| C3-4 | 16/18 (89%) | 10/18 (56%) | 3/18 | 130s | 101s (n=3) | 130s | 120s (n=7) |
| C1-2-3 | 16/18 (89%) | 12/18 (67%) | 3/18 | 108s | 99s (n=3) | 124s | 99s (n=7) |
| C1-2-4 | 16/18 (89%) | 11/18 (61%) | 3/18 | 97s | 69s (n=3) | 119s | 84s (n=7) |
| C1-3-4 | 17/18 (94%) | 11/18 (61%) | 3/18 | 118s | 77s (n=3) | 120s | 139s (n=8) |
| C2-3-4 | 15/18 (83%) | 12/18 (67%) | 4/18 | 123s | 109s (n=4) | 123s | 110s (n=6) |
| C1-2-3-4 | 16/18 (89%) | 12/18 (67%) | 4/18 | 134s | 106s (n=4) | 134s | 155s (n=7) |

The context cliff stands: **C0 RCA 3/18 (17%)** vs ~61–72% for any non-empty context; exec is ~1/6 per context regardless of which corpora are enabled.

## 6. Execution by fault type (96 runs each)

| Fault | Detect | Exec | MTTR med (corr) |
| --- | --- | --- | --- |
| `scale_zero` | 95/96 | **48/96 (50%)** | 73s |
| `config_corruption` | 76/96 | **1/96 (1%)** | 818s |
| `kill_pod` | 76/96 | **1/96 (1%)** | 398s |

The corrected data makes the fault skew explicit: essentially all successful remediation is `scale_zero` (trivially detectable, trivially reversible with `scale_deployment`).

## 7. Top full-win cells (detect ∧ RCA ∧ exec)

**Full wins: 50/288** (glm-53-flash 17, qwen38-27b 17, qwen36-27b 16) — unchanged. Top cells by corrected MTTD (all `scale_zero`):

| Context | Agent | Fault | MTTD (corr) | MTTR (corr) | MTTR verified |
| --- | --- | --- | --- | --- | --- |
| C1-4 | `maas_glm-53-flash` | scale_zero | 33s | 33s | 55s |
| C2-3 | `maas_glm-53-flash` | scale_zero | 33s | 33s | 63s |
| C2-4 | `maas_glm-53-flash` | scale_zero | 35s | 35s | 87s |
| C1-3 | `maas_glm-53-flash` | scale_zero | 35s | 35s | 70s |
| C1 | `maas_glm-53-flash` | scale_zero | 36s | 36s | 85s |
| C3-4 | `maas_glm-53-flash` | scale_zero | 37s | 37s | 71s |
| C4 | `maas_glm-53-flash` | scale_zero | 37s | 37s | 54s |
| C1-3-4 | `maas_glm-53-flash` | scale_zero | 37s | 37s | 81s |
| C3 | `maas_glm-53-flash` | scale_zero | 38s | 38s | 62s |
| C1-2 | `maas_glm-53-flash` | scale_zero | 39s | 39s | 87s |
| C1-2-4 | `maas_glm-53-flash` | scale_zero | 39s | 39s | 63s |
| C2 | `maas_glm-53-flash` | scale_zero | 40s | 40s | 55s |

## 8. Verification (run before writing this report)

| Check | Result |
| --- | --- |
| Batch completeness | 288 runs = 6 agents × 16 contexts × 3 faults ✓ |
| Exec recomputed from per-write records vs recorded | 50 vs 50, **0 flips** ✓ (single-write targeting+action requirement changed no outcomes) |
| Unchanged counts vs published report | detect 247 ✓, RCA 175 ✓, rem(OR) 203 ✓, exec 50 ✓, recovery 49 ✓, agent_error 21 ✓ |
| MTTR-present ⟺ exec-true | holds for all 288 runs ✓ |
| MTTD ≤ MTTR per run | holds for all 50 MTTR runs ✓ |
| Detection timestamp source | 247/247 detected runs recovered from agent signals (no fallbacks) ✓ |
| Fixed-code unit tests | gate nudge → stderr with clean stdout ✓; first-claim stamp survives later write ✓; act-first detection≡write semantics ✓; tolerant parser recovers timestamp from polluted stdout ✓ |

Corrected per-run rows: `out/analysis_12oct26_corrected.json` (produced by `scripts/recompute_corrected_metrics.py`).

## 9. Bugs fixed (applied to `code/` on 19 Oct 26)

| # | Bug | Location | Fix |
| --- | --- | --- | --- |
| 1 | Execute-gate nudge printed to **stdout**, corrupting the agent JSON protocol | `code/agents/execute_gate.py` | `file=sys.stderr` |
| 2 | Forced-RAG notice printed to **stdout** | `code/agents/rag_force.py` | `file=sys.stderr` |
| 3 | Context-window report printed to **stdout** on every run with usage stats (the every-run polluter behind glm's inflated MTTD) | `code/agents/model_context.py` | `file=sys.stderr` |
| 4 | MTTD used a strict `json.loads` on raw stdout; on any noise it silently fell back to the harness clock at subprocess exit | `code/harness/run_harness.py` (detection loop) | use `parse_agent_signals()` — same tolerant parser as the result pipeline |
| 5 | `detection_timestamp` only stamped at first successful write → detection ≡ remediation for act-first agents; declare-first agents' claims were never stamped | `code/agents/remediation_signals.py` + all 11 matrix agents | `stamp_detection_claim()` stamps at the first mid-loop `detected=true` claim (write stamp only if no claim came first) |
| 6 | `mttr_seconds` keyed on `remediation_executed_time` (any successful write) while Exec required targeted + action-appropriate → MTTR with 0 exec, n ≫ exec | `code/harness/evaluation.py`, `code/harness/run_harness.py` | shared `first_effective_remediation_write()` selector drives both MTTR and `remediation_executed` |

## 10. Live re-run status

A live re-run of the matrix was **blocked by infrastructure state** at correction time (not by the fixes):

- `otel-demo` namespace: all deployments scaled to **0/0** (cart, flagd, otel-collector, …) — no fault-injection target running.
- `agentic-aiops/clickhouse`: CrashLoopBackOff — startup probe `:8123/ping` refused; 122 restarts; PVC bound (50Gi). The `demo5-observability` ClickHouse is also restart-cycling.

Restore path (same as matrix preflight): scale the `otel-demo` stack back up, resolve the ClickHouse startup failure, re-establish the port-forward, then re-run `scripts/run_c_matrix_scenario_a_c_permutations.sh` (supports `MATRIX_RESUME=1`). The fixed harness will additionally benefit from claim-time detection stamping (§1), which can only reduce MTTD for declare-first agents relative to the values in this report.

## 11. Recommendations (updated)

1. **Default RCA pair** stands: `maas_deepseek` + `maas_qwen3` for diagnosis.
2. **Executor ranking by corrected MTTR:** `maas_glm-53-flash` **38s** (n=17) is the fastest appropriate-remediation agent — clearly ahead of qwen36-27b (67s, n=16) and qwen38-27b (133s, n=17). Pair glm-53-flash with the diagnosers when execution is required.
3. **Treat `kill_pod` as self-healing / not remediation-scored**, or require a different success criterion — 1/96 exec with agents mostly waiting for the deployment to recreate the pod is a design artifact, not a capability signal.
4. **`config_corruption` needs a write path agents can find** — 1/96 exec suggests the available write tools (restart/scale/delete) rarely constitute the judged-appropriate fix; consider `patch_deployment` (currently listed in the gate's tool set but not implemented in `execute_tool`).
5. **Re-add Nemotron/Kimi only after MaaS preflight is green** (unchanged).
6. **Future reports:** MTTD/MTTR are now comparable across agents (one parser, one clock, shared selector); the Sep/Oct MTTD trend lines in the longitudinal report should be re-derived from recomputed batches before drawing latency conclusions.
