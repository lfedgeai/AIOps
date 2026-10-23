# Evaluation Results — config_corruption "winnable" matrix (ccw)

**Generated:** 2026-10-23 14:20 AEST  
**Design:** `config_corruption` only × 8 CONTEXT_C subsets × **5 agents** = **40 cells** (harness design and resume model: [HARNESS_DESIGN.md](HARNESS_DESIGN.md))  
**Runs:** **40/40** completed, **failures=0**, `agent_error`=**0**  
**Kickoff:** 2026-10-22 22:30 AEDT (resumed 2026-10-23 11:42 after a session-churn kill; 14 cells carried over, 2 re-run)  
**Tracking:** local MLflow `agentic_aiops_mttd_mttr` — **40/40 matched by `run_id`, 0 metric mismatches**  
**Purpose:** verify that the newly implemented `patch_deployment` + `get_deployment_config` tools make `config_corruption` winnable.  
**Baseline:** [EVALUATION_RESULTS_22oct26.md](EVALUATION_RESULTS_22oct26.md) §By fault — `config_corruption` recovery **0/96** (structurally impossible: the fault is a bad `DATABASE_HOST` env var in the deployment spec; only restart/scale/delete existed as write tools).

> **What changed vs the Oct 20–22 batch (the experiment):** `get_deployment_config` (read) and `patch_deployment` (write) added to every agent's tool profile; execute-gate nudge and remediate skill updated. Everything else identical (prompts, skills, forced RAG, timeouts, judge, scoring). `config_corruption` exec/recovery numbers are therefore **not comparable** to Oct 22 by design — that is the point of this run. qwen38-27b excluded (44% `agent_error` in the Oct batch).

---

## Headline: the fault is now winnable

| Metric | This run (n=40) | Oct 22 config_corruption (n=96) |
|---|---|---|
| Detection | **40/40 (100%)** | 67/96 (70%) |
| RCA correct | **33/40 (83%)** | 58/96 (60%) |
| Remediation executed | **16/40 (40%)** | 17/96 (18%) |
| **Recovery verified** | **16/40 (40%)** | **0/96 (0%)** |
| Full wins (det∧RCA∧exec∧recovery) | **16/40** | 0 |
| `agent_error` | **0** | 19 |

Every one of the 16 recoveries was a genuine `patch_deployment` write removing both injected env vars (`DATABASE_HOST` + `HARNESS_INJECTED_FAULT`), verified by the harness **before** auto-recovery. The recovery ceiling documented in the Oct 22 report (192/288 = 67% for the full matrix) no longer applies.

## By agent (8 cells each)

| Agent | Detect | RCA | Exec | Recovery | MTTD med | MTTR med | MTTR verified med |
|---|---|---|---|---|---|---|---|
| `maas_deepseek` | 8/8 | 7/8 | 0/8 | 0/8 | 89s | — | — |
| `maas_qwen3` | 8/8 | 6/8 | 0/8 | 0/8 | 89s | — | — |
| `maas_qwen36-27b` | **8/8** | **8/8** | **8/8** | **8/8** | 58s | 58s | 84s |
| `maas_granite` | 8/8 | 4/8 | 0/8 | 0/8 | 46s | — | — |
| `maas_glm-53-flash` | **8/8** | **8/8** | **8/8** | **8/8** | 41s | 41s | 58s |

- **glm-53-flash: perfect sweep** — 8/8 full wins, MTTD 33–48s, recovered within 48–65s of injection, on every context including C0.
- **qwen36-27b: perfect sweep** — 8/8 full wins (MTTD 47–79s, recovered 75–109s). Notable redemption: 14/48 `agent_error` and only 4/16 config execs in the Oct batch → **0 errors, 8/8 wins** here. The config-inspection path needs few rounds, so its slow endpoint never hit the timeout.
- **The diagnosers are unchanged:** deepseek (RCA 7/8), qwen3 (6/8), granite (4/8) all detected but executed **0** remediations. deepseek mentioned `patch_deployment` in 4/8 outputs and once *saw* the real env vars via `get_deployment_config` (Oct 22 forensics) but still fails to convert discovery into a targeted write; its one Oct 22 "exec" was the `set_env` hallucination loophole, now closed and re-tested (redone C1 cell: honest 0 exec). qwen3 never touched the new tools.
- **Diagnosis is faster via the config path:** glm's config_corruption MTTD median dropped from **154s** (Oct 22, telemetry-trawling) to **41s** — checking `get_deployment_config` is a quicker route to root cause than correlating logs/traces.

## Full wins (16)

All 16 are glm-53-flash and qwen36-27b across all 8 contexts (see run JSONs for per-cell latencies; range: MTTD 33–79s, recovery 48–109s).

## By context (5 agents each)

| Context | RCA | Exec | Recovery |
|---|---|---|---|
| C0 | 2/5 | 2/5 | 2/5 |
| C1 | 5/5 | 2/5 | 2/5 |
| C2 | 3/5 | 2/5 | 2/5 |
| C3 | 5/5 | 2/5 | 2/5 |
| C4 | 4/5 | 2/5 | 2/5 |
| C1-2 | 4/5 | 2/5 | 2/5 |
| C2-3-4 | 5/5 | 2/5 | 2/5 |
| C1-2-3-4 | 5/5 | 2/5 | 2/5 |

Exec/recovery is exactly the two executors at every context — **context makes no difference to winning**: the winnable path works even at C0 (no corpora), and extra RAG context neither helps nor hinders it. The **C0 RCA cliff persists for diagnosers** (C0 RCA 2/5 vs 4–5/5 for non-empty contexts) — glm/qwen36 carry C0; deepseek/granite/qwen3 lose RCA there, consistent with the Oct 22 pattern.

## Tool guardrails (validated in-run)

- `set_env` add-new-var refusal — deepseek's hallucinated patch class can no longer mutate the deployment or score exec.
- No-op patch rejection — cannot fake `remediation_executed`.
- 409-conflict retry — patch→verify→re-patch sequences succeed during rollouts.

## Caveats

1. **14 carried-over cells** (1, 3–15) ran in run #1 with the pre-guardrail tool and with an inert junk `CORRECT_VAR` env var visible on cart (deepseek's loophole patch, since scrubbed). No carried-over cell contains a successful patch write (all three diagnosers finished 0/8 exec), so outcomes are unaffected; at most `get_deployment_config` output was noisier for those cells.
2. The invalidated deepseek × C1 cell (loophole exec) was re-run under fixed semantics (honest 0 exec); its original MLflow run remains in the window (documented, unmatched by design).
3. qwen38-27b not tested on this fault; small n (8 cells/agent, 5/context); single fault type.

## Recommendations

1. **Adopt the config tool path for the main matrix** — `patch_deployment`/`get_deployment_config` are proven; re-include `config_corruption` recovery in headline metrics for future Option-2 batches (ceiling no longer 192/288).
2. **Executor pairing stands:** glm-53-flash (fastest: 41s MTTD) and qwen36-27b as the executors; deepseek/qwen3/granite as diagnosers. The diagnose→execute gap for the diagnosers is *not* a tool-availability problem — they decline to write even with the exact fix path available and visible.
3. **Re-test qwen38-27b** on this fault (it was excluded for reliability, and this workload produced zero timeouts for qwen36 — the config path is cheap in rounds).
4. **Consider `patch_deployment` for the other spec-level faults** (`memory_limit`, `readiness_probe_fail`) — same structural problem class, same fix pattern.
