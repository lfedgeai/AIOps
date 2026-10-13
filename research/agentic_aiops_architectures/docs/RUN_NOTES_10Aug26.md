# Run notes — 10 August 2026 (Option 2 re-run)

## Agent lineup change

**Dropped from the default matrix** (agent code kept under `code/agents/` for optional one-offs):

| Dropped | Why |
|---------|-----|
| `maas_llama-scout` (Llama Scout 17B) | Weak under forced RAG (Option 2 Jul: 33% detect / 6% RCA); often emits tools as prose instead of native `tool_calls` |
| `maas_gpt-oss-120b` (GPT-OSS 120B) | Same failure pattern, worse wall-clock (Option 2 Jul: 31% detect / 12% RCA); no unique signal vs Scout |

**Added:**

| Added | Model id | Notes |
|-------|----------|--------|
| `maas_granite` | `granite-3-2-8b-instruct` | Native tools (+ content fallback) |
| `maas_llama31-70b` | `llama-31-70b-cpu` | Native when forced; content JSON fallback; long read timeout (600s) |

**Default matrix agents (5):** `nemotron-nano-3`, `maas_deepseek`, `maas_qwen3`, `maas_granite`, `maas_llama31-70b`

## Suite

Option 2 — scenario **A** × all CONTEXT_C subsets (16) × 5 agents × 3 faults = **240 runs**  
Script: `scripts/run_c_matrix_scenario_a_c_permutations.sh`  
Log: `out/c_matrix_scenario_a_c_permutations_run.log`  
Service: `aiops-c-matrix-sa-c-permutations.service`  
Started: **Mon 10 Aug 2026 13:04 AEST**  
**Completed: Tue 11 Aug 2026 03:43:26 AEST (`failures=0/240`)**  
Full report: [EVALUATION_RESULTS_11Aug26.md](EVALUATION_RESULTS_11Aug26.md)

## Smoke (2026-08-10)

Both harness runs completed without `agent_error` (infra + agent process OK). Quality is a separate question for the full matrix.

| Agent | Context | Fault | Detected | RCA | Notes |
|-------|---------|-------|----------|-----|-------|
| `maas_granite` | C0 | `scale_zero` | Yes (~30s MTTD) | No | Declared detect; 0 tool calls in metrics — watch in matrix |
| `maas_llama31-70b` | C0 | `scale_zero` | No | No | ~4 min wall; pipeline OK, no detect this cell |

Artifacts: `out/smoke_10aug26/`

## Infra fixes before kickoff

- Scaled `deploy/clickhouse` in `agentic-aiops` back to 1 (was 0)
- `scripts/clickhouse_pf_with_login.sh` — PF unit now re-logins as kubeadmin (stale token was flapping)
- Cart/valkey restored to 1/1

## Follow-up (post-run)

- Nemotron/Llama 0% detect were **API failures** (HTML / connection), not fair model scores — see [EVALUATION_RESULTS_11Aug26.md](EVALUATION_RESULTS_11Aug26.md).
- **Going forward:** matrices call `scripts/preflight_agent_apis.py` and **abort** if any default-matrix LLM endpoint fails chat/completions (HTML error page, connection error, missing creds).
