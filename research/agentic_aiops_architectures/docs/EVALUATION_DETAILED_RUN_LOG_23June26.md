# Full Matrix Evaluation — Detailed Run Log & Analysis

**Date**: 23 June 2026  
**Total runs**: 132 (128 planned + 4 duplicate scale_zero runs from scenario A)  
**Duration**: ~12 hours  
**MLflow**: 200 runs logged at `http://localhost:5050`

---

## Executive Summary

| Metric | Value |
|---|---|
| Overall detection rate | **108/132 (82%)** |
| Detection rate (excl. readiness_probe_fail) | **108/116 (93%)** |
| Undetectable fault | `readiness_probe_fail` — 0% across all agents/scenarios |
| Best detection scenario | **Scenario A (89%)** — single agent with all tools |
| Fastest agent | **Llama Scout: 22-36s** avg MTTD |
| Most autonomous (executes fixes) | **Nemotron: 8 fixes in scenario A** |
| Best inference speed | **Nemotron: 177 tok/s, 0.95s TTFT** |

---

## Detailed Run Log

### Scenario 0 (Baseline — telemetry only, no K8s tools)

| # | Fault | Agent | Det | MTTD | Rounds | Tools | Tokens | TTFT |
|---|---|---|---|---|---|---|---|---|
| 1 | config_corruption | deepseek | Y | 180s | 5 | 3 | 11170 | - |
| 2 | config_corruption | llama_scout | Y | 207s | 2 | 3 | 2713 | - |
| 3 | config_corruption | nemotron | Y | 49s | 3 | 2 | 9479 | 0.96s |
| 4 | config_corruption | qwen3 | Y | 60s | 2 | 3 | 0* | 0.79s |
| 5 | dependency_removal | deepseek | Y | 236s | 4 | 1 | 8788 | - |
| 6 | dependency_removal | llama_scout | Y | 135s | 2 | 4 | 2696 | - |
| 7 | dependency_removal | nemotron | Y | 190s | 6 | 5 | 19794 | 0.93s |
| 8 | dependency_removal | qwen3 | Y | 172s | 3 | 4 | 0* | 0.80s |
| 9 | kill_pod | deepseek | Y | 245s | 3 | 5 | 5761 | - |
| 10 | kill_pod | llama_scout | **N** | - | - | - | - | - |
| 11 | kill_pod | nemotron | Y | 81s | 6 | 5 | 20323 | 0.93s |
| 12 | kill_pod | qwen3 | Y | 77s | 3 | 4 | 0* | 0.76s |
| 13 | memory_limit | deepseek | Y | 55s | 2 | 1 | 3558 | - |
| 14 | memory_limit | llama_scout | Y | 95s | 2 | 3 | 2859 | - |
| 15 | memory_limit | nemotron | Y | 43s | 5 | 4 | 11816 | 0.93s |
| 16 | memory_limit | qwen3 | Y | 70s | 3 | 4 | 0* | 0.82s |
| 17 | network_partition | deepseek | **N** | - | 6 | 4 | 19001 | - |
| 18 | network_partition | llama_scout | Y | 57s | 2 | 3 | 2712 | - |
| 19 | network_partition | nemotron | Y | 54s | 5 | 4 | 16970 | 0.94s |
| 20 | network_partition | qwen3 | Y | 82s | 3 | 4 | 0* | 0.78s |
| 21-24 | readiness_probe_fail | ALL | **N** | - | - | - | - | - |
| 25 | replica_overload | deepseek | Y | 71s | 4 | 2 | 11697 | - |
| 26 | replica_overload | llama_scout | Y | 92s | 2 | 3 | 3918 | - |
| 27 | replica_overload | nemotron | Y | 57s | 5 | 4 | 17501 | 0.95s |
| 28 | replica_overload | qwen3 | Y | 66s | 3 | 4 | 0* | 0.80s |
| 29 | scale_zero | deepseek | Y | 261s | 4 | 1 | 9432 | - |
| 30 | scale_zero | llama_scout | Y | 57s | 2 | 3 | 2713 | - |
| 31 | scale_zero | nemotron | Y | 155s | 4 | 3 | 12060 | 0.95s |
| 32 | scale_zero | qwen3 | Y | 234s | 3 | 4 | 0* | 0.77s |

*Qwen3 tokens report 0 due to streaming token counting not available from LiteLLM.

**Scenario 0 summary**: 26/32 detected (81%). No MTTR possible (no K8s write tools).

---

### Scenario A (Full autonomous — telemetry + K8s tools)

| # | Fault | Agent | Det | MTTD | MTTR | Fix Tool | Rounds | Tools |
|---|---|---|---|---|---|---|---|---|
| 33 | config_corruption | deepseek | Y | 64s | - | - | 3 | 1 |
| 34 | config_corruption | llama_scout | Y | 22s | - | - | 2 | 5 |
| 35 | config_corruption | **nemotron** | Y | 64s | **64s** | restart_deployment | 6 | 5 |
| 36 | config_corruption | qwen3 | Y | 61s | - | - | 3 | 5 |
| 37 | dependency_removal | **deepseek** | Y | 58s | **58s** | restart_deployment | 3 | 6 |
| 38 | dependency_removal | llama_scout | Y | 23s | - | - | 2 | 5 |
| 39 | dependency_removal | **nemotron** | Y | 172s | **172s** | delete_pod | 7 | 6 |
| 40 | dependency_removal | qwen3 | Y | 193s | - | - | 4 | 6 |
| 41 | kill_pod | deepseek | Y | 176s | - | - | 2 | 5 |
| 42 | kill_pod | llama_scout | Y | 22s | - | - | 2 | 5 |
| 43 | kill_pod | **nemotron** | Y | 173s | **173s** | restart_deployment | 7 | 6 |
| 44 | kill_pod | **qwen3** | Y | 54s | **54s** | restart_deployment | 3 | 8 |
| 45 | memory_limit | **deepseek** | Y | 279s | **279s** | restart_deployment | 6 | 9 |
| 46 | memory_limit | llama_scout | Y | 22s | - | - | 2 | 5 |
| 47 | memory_limit | **nemotron** | Y | 65s | **65s** | restart_deployment | 8 | 7 |
| 48 | memory_limit | qwen3 | Y | 74s | - | - | 3 | 5 |
| 49 | network_partition | **deepseek** | Y | 87s | **87s** | restart_deployment | 7 | 8 |
| 50 | network_partition | llama_scout | Y | 22s | - | - | 2 | 5 |
| 51 | network_partition | **nemotron** | Y | 169s | **169s** | restart_deployment | 6 | 5 |
| 52 | network_partition | **qwen3** | Y | 51s | **51s** | restart_deployment | 3 | 6 |
| 53-56 | readiness_probe_fail | ALL | **N** | - | - | - | - | - |
| 57 | replica_overload | deepseek | Y | 119s | - | - | 3 | 1 |
| 58 | replica_overload | llama_scout | Y | 23s | - | - | 2 | 5 |
| 59 | replica_overload | **nemotron** | Y | 59s | **59s** | restart_deployment | 6 | 5 |
| 60 | replica_overload | **qwen3** | Y | 64s | **64s** | restart_deployment | 3 | 6 |
| 61-68 | scale_zero | ALL | Y | 22-152s | 40-153s | delete_pod/restart | 2-7 | 1-7 |

**Scenario A summary**: 32/36 detected (89%). **15 autonomous fixes executed.**

---

### Scenario B (Signal specialists: logs / traces / metrics)

| Detection rate | 23/32 (72%) — WORST scenario |
|---|---|

Multi-agent orchestrator runs internally. Per-agent tool/token metrics not propagated (shows 0 in output).

**Failures** (beyond readiness_probe_fail):
- Qwen3 timed out on 3 faults (kill_pod, memory_limit, scale_zero)
- Nemotron failed on config_corruption

---

### Scenario C (Domain specialists: hardware / platform / application)

| Detection rate | 27/32 (84%) |
|---|---|

Better than B but still behind A. Qwen3 timed out on dependency_removal.

---

## Agent Behavior Analysis

### Nemotron-Nano-3

| Metric | Value |
|---|---|
| Inference speed | 177 tok/s, 0.93-1.21s TTFT |
| Rounds per detection | 5-8 (thorough) |
| Token budget per run | 10K-48K |
| Fix rate (scenario A) | **100%** (8/8 detected → fixed) |
| Weakness | Slowest MTTD (avg 140s) |

### Llama Scout 17B

| Metric | Value |
|---|---|
| Rounds per detection | 2 (minimum possible) |
| MTTD | **22-36s** (fastest, consistently) |
| Fix rate | **0%** (never executes write tools) |
| Token budget per run | 2.7K-8.5K (cheapest) |
| Behavior | Perfect scout — detects immediately, reports, never acts |

### Qwen3-14B

| Metric | Value |
|---|---|
| Rounds per detection | 3 |
| MTTD | 51-234s (variable) |
| Fix rate (scenario A) | 50% (4/8) |
| Weakness | Times out in multi-agent scenarios B/C |

### DeepSeek R1 Distill 14B

| Metric | Value |
|---|---|
| Rounds per detection | 2-7 (unpredictable) |
| Tool calls | **1 in 40% of runs** (prompt-based limitation) |
| MTTD | 55-279s (slowest avg: 171s) |
| Fix rate (scenario A) | 38% (3/8) |
| Critical issue | Often stops after 1 `search_logs` call |

---

## Critical Findings

### 1. Multi-agent scenarios (B/C) lose tool-level observability

Orchestrator handles tool dispatch internally → 0 tokens/tools reported in run output.

### 2. Llama Scout hits a 22s detection floor

Limited by harness architecture: 15s telemetry propagation + ~7s inference = minimum possible MTTD.

### 3. Token cost varies 10x

Nemotron (~33K/run) vs Llama Scout (~7K/run). Thoroughness costs 5x more tokens.

### 4. DeepSeek's prompt-based tool calling is non-deterministic

1 tool in 40% of runs, 9 tools in others. Model decides when to stop exploring.

### 5. readiness_probe_fail needs a new approach

No telemetry signal, no crash, pod runs fine — just not marked ready. Requires an endpoint-checking tool.

---

## CRITICAL FINDING: Root Cause Accuracy Problem

### The Numbers

- **RCA accuracy**: 4/32 (12%) correctly identified `cart` as the fault target
- **Remediation accuracy**: 0/15 fixes applied to the correct service (all 15 MTTR entries fixed the WRONG service)

### What Agents Identified vs What Was Broken

All faults were injected on `cart`. Here's what agents concluded:

| Fault (on cart) | Agent | Identified root cause | Fix applied to |
|---|---|---|---|
| config_corruption | nemotron | load-generator | load-generator (WRONG) |
| dependency_removal | deepseek | load-generator | load-generator (WRONG) |
| dependency_removal | nemotron | email | email pod (WRONG) |
| kill_pod | nemotron | postgresql | postgresql (WRONG) |
| kill_pod | qwen3 | postgresql | postgresql (WRONG) |
| memory_limit | deepseek | load-generator | load-generator (WRONG) |
| memory_limit | nemotron | email service | email pod (WRONG) |
| network_partition | deepseek | resource exhaustion | email (WRONG) |
| network_partition | nemotron | load-generator | load-generator (WRONG) |
| network_partition | qwen3 | postgresql | postgresql (WRONG) |
| replica_overload | nemotron | load-generator | load-generator (WRONG) |
| replica_overload | qwen3 | load-generator | load-generator (WRONG) |
| scale_zero | nemotron | load-generator | load-generator (WRONG) |
| scale_zero | nemotron | load-generator | load-generator (WRONG) |
| scale_zero | qwen3 | postgresql | postgresql (WRONG) |

Only **Llama Scout** occasionally identified cart correctly (4 out of 8 detected runs) — but it never executes fixes, so even correct RCA didn't lead to correct remediation.

### Why This Happens

The OTEL demo cluster has persistent pre-existing issues that produce telemetry noise:

| Pre-existing issue | Signal strength | How it misleads |
|---|---|---|
| load-generator Chromium crashes | HIGH — constant BrowserType.launch errors in logs | Dominates `search_logs(query="error")` results |
| email service CrashLoopBackOff | MEDIUM — 6+ restarts visible in pod status | Appears as "unhealthy pod" in `get_pod_status` |
| postgresql 2013 restarts | MEDIUM — high restart count | Appears as chronically unstable |
| accounting/checkout/fraud Pending | LOW — pods stuck in Pending | Visible but no log output |

When the agent calls `search_logs(query="error")`, it gets 20 results — 19 of which are load-generator Chromium crashes (pre-existing), and maybe 1 is the actual cart error (if any). The agents have no concept of "this is old noise" vs "this is new."

### What This Means

1. **Detection rate metrics are inflated** — agents report `detected=true` because they DO see errors (cluster is always unhealthy). But they're detecting ambient noise, not the specific injected fault.

2. **MTTR is measuring the wrong thing** — the agent "fixes" load-generator (which wasn't the problem), and the actual cart fault persists.

3. **A clean cluster would give misleadingly good results** — if there were zero background errors, any fault signal would be unambiguous. But production clusters are never clean.

4. **The real problem is temporal correlation** — agents need to compare "before fault" vs "after fault" to identify what CHANGED, not what's LOUDEST.

### How to Fix This (Next Iteration)

1. **Baseline comparison in the prompt**: "First, check what errors existed 5 minutes BEFORE the since_ts. Then check what's NEW since since_ts. Only report NEW issues as the fault."

2. **Add a `compare_timewindows` tool**: Returns only errors/events that appeared AFTER the fault injection time and were NOT present before.

3. **Weight recency**: Errors that appeared within 30 seconds of since_ts are more likely the fault than errors that have been recurring for hours.

4. **Verify the fix worked**: After executing a remediation, re-check if the symptom that triggered detection is now gone. If not, the agent fixed the wrong thing.

5. **Add fault-specific correlation**: If the harness knows it scaled `cart` to 0, it can validate whether the agent's root cause mentions `cart`. This is the `rca_correct` metric which already exists in the harness but wasn't prominent in our analysis until now.

