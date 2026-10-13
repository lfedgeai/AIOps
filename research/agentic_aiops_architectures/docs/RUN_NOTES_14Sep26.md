# Run notes — September 2026 (next matrix, harness hardening)

## Changes since Aug 12 Option 2 run

| Setting | Aug 12 run | Next run |
|---------|------------|----------|
| `max_tool_iterations` | **8** | **20** (`config/harness.yaml`) |
| `AGENT_TIMEOUT` | 420s | **600s** (matrix scripts) |
| `detection_timeout` | 360s | **720s** |
| Pre-run cluster gate | none | **`scripts/pre_run_health_gate.py`** |
| Context window metrics | not logged | **`ai_peak_prompt_tokens`**, `ai_model_context_limit`, `ai_context_tokens_available`, `ai_context_utilization_pct` |

## Kickoff checklist

```bash
cd research/agentic_aiops_architectures
source config/.env
python3.13 scripts/preflight_agent_apis.py
python3.13 scripts/pre_run_health_gate.py
# optional smoke:
./scripts/run_harness.sh --agents maas_qwen36-27b --scenario a --context-c 1,2,3,4 --flag scale_zero --variant cart
```

Matrix: `scripts/run_option2_permutations_systemd.sh` or `scripts/run_c_matrix_scenario_a_c_permutations.sh`

**Kickoff:** Tue 15 Sep 2026 — restarted **without nemotron** (MaaS 503); **336 runs** (7 agents × 16 contexts × 3 faults).

Prior attempts: Mon 14 03:19 (stopped run 46 health-gate); Mon 15 01:45 (nemotron preflight fail).  
**Log:** `out/c_matrix_scenario_a_c_permutations_run.log`  
**Report (auto on completion):** `docs/EVALUATION_RESULTS_<date>.md` via `analyze_option2_run.py`

Monitor:
```bash
tail -f out/c_matrix_scenario_a_c_permutations_run.log
find out -name 'context_C*_sa_*_*.json' -newermt "$(cat out/.matrix_kickoff_aest)" | wc -l  # target 384
systemctl --user status aiops-c-matrix-sa-c-permutations.service
```

---

## MLflow logging (experiment `agentic_aiops_mttd_mttr`)

**Tracking URI:** `http://127.0.0.1:5050` (local SQLite + `mlartifacts/`)

### Per-run tags

| Tag | Example |
|-----|---------|
| `context_c` | `0`, `1`, `2,3,4`, … |
| `scenario` | `a` |
| `tool_profiles` | `telemetry_k8s` |
| `mlflow.note.content` | Markdown summary (approach, MTTD, RCA, **context window** on new runs) |

### Harness metrics (always)

- `detected`, `mttd_seconds`, `mttr_seconds`, `mttr_verified_seconds`
- `rca_correct`, `rca_declared`, `rca_inferred`, `rca_judge_correct`
- `remediation_correct`, `remediation_declared`, `remediation_executed`, `remediation_judge_correct`
- `recovery_verified`, `recovery_appropriate`

### Agent AI metrics (when LLM invoked)

| Metric | Meaning |
|--------|---------|
| `ai_rounds` | LLM rounds used (was capped at **8** in Aug run) |
| `ai_prompt_tokens` | Sum of per-round prompt tokens (overcounts history; do not use as context fill) |
| `ai_completion_tokens` | Sum of completion tokens |
| `ai_total_tokens` | Sum of round totals |
| `ai_peak_prompt_tokens` | **Peak context fill** — max prompt size in any single round |
| `ai_model_context_limit` | From `GET /v1/models` when exposed (262144 on Prelude vLLM) |
| `ai_context_tokens_available` | `limit − peak` |
| `ai_context_utilization_pct` | Peak / limit × 100 |
| `ai_avg_ttft_sec`, `ai_total_llm_latency_sec`, `ai_total_tool_exec_sec`, `ai_total_tool_calls`, `ai_avg_tokens_per_sec` | Timing / throughput |

### Artifacts

| Artifact | Contents |
|----------|----------|
| `agent_ai_metrics.txt` | Human-readable per-round TTFT, tokens, tools |
| `agent_llm_rounds.json` | Per-round request/response |
| `agent_prompts.txt` / `agent_tool_calls.txt` | Transcript |
| `harness_run.json` | Full harness result |
| `artifacts/context_c.json` | RAG corpus metadata (chunk counts) |
| `artifacts/dataset_metadata.json` | ClickHouse source, fault, since timestamp |

### Filtering in MLflow UI

- **By context:** tag `context_c = 1,2,3,4` or `0`
- **By agent:** param `approach = maas_deepseek` (or read `**Approach**` in run note)
- **Iteration exhaustion:** metric `ai_rounds >= 8` (Aug run) or `>= 20` (next run)
- **Context pressure:** metric `ai_peak_prompt_tokens` vs `ai_model_context_limit`

### Validation after run

```bash
python3.13 scripts/analyze_option2_run.py
# Compare harness JSON vs MLflow: mismatches should be 0
```

---

## Aug 12 investigation: “C0 beats context” — findings

**Verdict: C0 is not broadly better.** Pooled **C≠0** beats C0 on detection (52% vs 46%) and RCA (45% vs 29%). C0 only leads on **remediation executed** (29% vs 21%) — driven by executor agents spending less budget on forced RAG.

### What actually limited context runs (Aug 12, 8-iteration cap)

| Hypothesis | Evidence |
|------------|----------|
| Context window full? | **No.** Peak prompt ~3–15k tokens; limits 32k–262k. Not the bottleneck. |
| Iteration budget exhausted? | **Yes — primary limiter.** **50%** of all runs hit exactly **8 rounds**. Undetected@8r rises with corpus count: 21% (C0) → **38%** (C1,2,3,4). |
| RAG eating rounds? | **Yes.** 4 forced RAG corpora consume ~4 rounds before telemetry/K8s unlock. |
| More corpora → worse RCA? | **Yes in Aug data:** 1 corpus 49% RCA → 4 corpora **33%** RCA (n=24 each for 4-corpus cells). |

### Per-agent pattern (C0 vs C≠0)

| Agent type | C0 | With context |
|------------|-----|--------------|
| **RCA leaders** (DeepSeek, Granite) | C0 RCA **0%** (3/3 cells) | C3 / C1-4: **100%** RCA |
| **Executors** (qwen36, nemotron, kimi) | Higher exec (less RAG tax) | More rounds on RAG → more @8r exhaustion |

### Implication for next run

- **20 iterations** should reduce false “context hurts” signal from budget exhaustion.
- Re-compare C0 vs C2/C3 after next matrix; expect RCA leaders to gain more from context than executors.
- Watch `ai_rounds` distribution — if many runs still hit 20, bump further or reduce forced-RAG rounds.

Prior report: [EVALUATION_RESULTS_12Aug26.md](EVALUATION_RESULTS_12Aug26.md)
