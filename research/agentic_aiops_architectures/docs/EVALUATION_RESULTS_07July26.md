# Context Engineering Evaluation Results — 7 July 2026

## Run Summary

| Item | Value |
|------|-------|
| **Matrix size** | 48 runs (4 agents × 3 faults × 4 context corpora) |
| **Failures** | 0/48 harness failures |
| **Runtime** | ~3 hours (Mon 6 Jul 22:07 → Tue 7 Jul 01:23 AEST) |
| **Scenario** | **A only** (single agent, telemetry + Kubernetes) |
| **Target service** | `cart` deployment in `otel-demo` namespace |
| **MLflow** | All runs logged at `http://localhost:5050` — filter by tag `context_c` |
| **Artifacts** | `out/context_C{1-4}_sa_{fault}_{agent}.json` |

This run is the first evaluation of the **C-model context engineering** design (replacing the earlier L0–L2 levels). It isolates the effect of RAG corpus access while holding scenario topology constant.

---

## Experiment Design

Two orthogonal axes define each harness run: **scenario** (agent topology / tool profile) and **context corpora** (optional RAG tools).

### Scenarios (`SCENARIO` / `--scenario`)

Scenarios control *who* investigates and *which tools* they receive. They do **not** change the injected fault mechanics.

| ID | Name | Mode | Tool profile | Capabilities |
|----|------|------|--------------|--------------|
| **0** | `baseline_telemetry` | Single agent | `telemetry_only` | Logs, traces, metrics, ClickHouse. **No K8s API** — detect only, cannot inspect pods or remediate. |
| **A** | `multimodal_telemetry_k8s` | Single agent | `telemetry_k8s` | Full telemetry **plus** `get_pod_status`, `get_events`, `get_pod_logs`, and write tools (`restart_deployment`, `scale_deployment`, `delete_pod`). **Used in this matrix.** |
| **B** | `signal_specialists` | Multi-agent (3) | `logs_only` / `traces_only` / `metrics_only` | Each agent sees one signal type; orchestrator merges results. |
| **C** | `domain_specialists` | Multi-agent (3) | `hardware_metrics` / `platform_k8s` / `application_telemetry` | Split by domain: host metrics, K8s platform, app telemetry. |

> **Note:** The June 2026 full matrix exercised scenarios 0/A/B/C across 8 faults. **This July run fixed scenario A** to measure context-engineering impact without confounding multi-agent effects.

### Context engineering levels (`CONTEXT_C` / `--context-c`)

Context levels control *optional RAG tools* layered on top of the scenario tool profile. They are **explicit multi-select** (e.g. `1,4` enables C1 + C4 together) and **not cumulative** (C2 does not imply C1).

| ID | Name | Tool | Corpus (OpenTelemetry Demo repo) |
|----|------|------|----------------------------------|
| **C0** | Baseline | *(none)* | Live telemetry + scenario tools only. No RAG. |
| **C1** | Source code | `rag_search_source` | Application source under `src/**` |
| **C2** | Documentation | `rag_search_docs` | README, docs, changelog |
| **C3** | Architecture | `rag_search_architecture` | Mermaid diagrams, topology docs |
| **C4** | Dependencies | `rag_search_dependencies` | `docker-compose` and Kubernetes manifests |

**Rules in this evaluation:**

- Each run used **one corpus only** (C1, C2, C3, or C4 — not combined).
- **C0 was not included** in this matrix (available via `INCLUDE_C0=1` in `scripts/run_c_matrix.sh`).
- RAG is **tool-call only** — no auto-injection of chunks into the prompt.
- **`compare_telemetry`** is a standard telemetry tool (always available in scenario A); it is **not** a context layer.
- **`get_baseline_telemetry`** and generic **`rag_search`** were removed in the C-model refactor.

### Faults injected

All faults target the **cart** service (`--variant cart`):

| Fault | Mechanism | Expected remediation |
|-------|-----------|---------------------|
| `scale_zero` | Scale cart deployment to 0 replicas | `scale_deployment` cart → ≥1 |
| `config_corruption` | Inject invalid env var on cart | Remove corruption / restart |
| `kill_pod` | Delete cart pod(s) | Self-healing via deployment controller (verify recovery) |

### Agents

| Harness name | Model (MaaS / endpoint) |
|--------------|-------------------------|
| `nemotron-nano-3` | NVIDIA Nemotron 3 Nano (dedicated endpoint) |
| `maas_deepseek` | DeepSeek R1 distill Qwen 14B |
| `maas_qwen3` | Qwen3 14B |
| `maas_llama-scout` | Llama Scout 17B |

### Scoring

Results use the **hybrid oracle judge** introduced after the June matrix:

- **RCA correct** — declared, inferred from tool calls, or judge-confirmed identification of `cart` as root cause.
- **Remediation correct** — judge confirms the fix addressed the injected fault.
- **Recovery verified** — post-remediation telemetry/K8s checks (0% in this run).

---

## Top-Line Results

| Metric | Value | June 2026 baseline (scenario A, all faults) |
|--------|-------|---------------------------------------------|
| Detection rate | **31/48 (65%)** | ~89% (scenario A, 8 faults) |
| RCA accuracy | **13/48 (27%)** | **12%** (cart identified correctly) |
| Remediation correct | **9/48 (19%)** | ~12% autonomous fix rate |
| Remediation executed | **5/48 (10%)** | Low across all agents |
| Recovery verified | **0/48 (0%)** | Not measured in June |
| Judge RCA (strict) | **6/48 (12%)** | N/A |
| MTTD median | **63s** | Nemotron ~140s, Llama Scout ~46s |
| MTTD mean | **94s** | — |

**Headline:** RCA accuracy **more than doubled** vs the June deep-dive (12% → 27%), driven almost entirely by **Llama Scout (67% RCA)**. Context corpora had **no measurable effect** because agents never called RAG tools. Qwen3 **failed on every run** (empty LLM response).

---

## Key Insights

### 1. Llama Scout dominates RCA; Nemotron detects but misdiagnoses

| Agent | Detection | RCA | Remediation correct | MTTD median | RAG tools called |
|-------|-----------|-----|---------------------|-------------|------------------|
| **Llama Scout** | **12/12 (100%)** | **8/12 (67%)** | **6/12 (50%)** | **27s** | 0 |
| Nemotron | 12/12 (100%) | 3/12 (25%) | 2/12 (17%) | 64s | 0 |
| DeepSeek | 7/12 (58%) | 2/12 (17%) | 1/12 (8%) | 254s | 0 |
| Qwen3 | **0/12 (0%)** | 0/12 (0%) | 0/12 (0%) | — | 0 |

Llama Scout achieved **100% RCA on C1 (source code)** despite never calling `rag_search_source`. Nemotron and DeepSeek frequently acted on **wrong deployments** (email, load-generator, otel-collector) despite high detection rates.

### 2. Context corpora (C1–C4) did not affect outcomes — RAG was never used

| Context | Detection | RCA | `compare_telemetry` used | RAG tool used |
|---------|-----------|-----|--------------------------|---------------|
| C1 source | 8/12 (67%) | 3/12 (25%) | 3/12 | **0/12** |
| C2 docs | 7/12 (58%) | 3/12 (25%) | 3/12 | **0/12** |
| C3 architecture | 9/12 (75%) | 3/12 (25%) | 3/12 | **0/12** |
| C4 dependencies | 7/12 (58%) | 4/12 (33%) | 3/12 | **0/12** |

Across all 48 runs, **zero** calls to `rag_search_source`, `rag_search_docs`, `rag_search_architecture`, or `rag_search_dependencies` appear in agent tool traces. The C4 slight RCA uplift (33% vs 25%) is within noise and attributable to agent/fault luck, not corpus content.

**Implication:** Tool-call-only RAG requires either stronger prompting (“you MUST query the repo before concluding”), mandatory tool-use policies, or auto-injection of top-k chunks for corpora to matter.

### 3. `compare_telemetry` is underused

Only **12/48 runs (25%)** invoked `compare_telemetry`, though it is always available in scenario A. Agents still default to `search_logs` (48/48 runs) which surfaces **chronic cluster noise** (load-generator Chromium crashes, email restarts) rather than post-T0 deltas.

| Tool | Calls (across 48 runs) |
|------|------------------------|
| `search_logs` | 48 |
| `get_pod_status` | 44 |
| `compare_telemetry` | 12 |
| `scale_deployment` | 7 |
| `delete_pod` | 5 |

### 4. Qwen3 total failure — empty model responses

All 12 Qwen3 runs show `ai_rounds: 1`, **0 prompt/completion tokens**, and **0 tool calls**. The agent exits immediately with `detected: false`. This is an integration/model availability issue, not a context-engineering result. Qwen3 should be excluded from cross-agent comparisons until the MaaS endpoint is fixed.

### 5. Fault difficulty was uniform

| Fault | Detection | RCA | Remediation correct |
|-------|-----------|-----|---------------------|
| `scale_zero` | 10/16 (62%) | 5/16 (31%) | 3/16 (19%) |
| `config_corruption` | 11/16 (69%) | 4/16 (25%) | 2/16 (12%) |
| `kill_pod` | 10/16 (62%) | 4/16 (25%) | 4/16 (25%) |

No fault was reliably easy; `scale_zero` had the best RCA (31%) but agents still often scaled/restarted the wrong service.

### 6. Detection ≠ diagnosis (still true)

65% detection vs 27% RCA means agents frequently notice *something wrong* in a noisy cluster but fail to attribute it to the **injected cart fault**. The June finding stands: **log volume ≠ injected fault signal**.

---

## Detection Heatmap (scenario A, all contexts)

Rows = agent, columns = fault. Cell = detected on all context runs for that fault.

| Agent | scale_zero (×4 ctx) | config_corruption (×4) | kill_pod (×4) |
|-------|---------------------|------------------------|---------------|
| Llama Scout | ✓✓✓✓ | ✓✓✓✓ | ✓✓✓✓ |
| Nemotron | ✓✓✓✓ | ✓✓✓✓ | ✓✓✓✓ |
| DeepSeek | ✓✓✗✗ | ✓✓✓✓ | ✓✓✗✗ |
| Qwen3 | ✗✗✗✗ | ✗✗✗✗ | ✗✗✗✗ |

## RCA Heatmap (agent × context corpus)

| Agent | C1 source | C2 docs | C3 arch | C4 deps |
|-------|-----------|---------|---------|---------|
| **Llama Scout** | **3/3** | 2/3 | 1/3 | 2/3 |
| Nemotron | 0/3 | 1/3 | 0/3 | 2/3 |
| DeepSeek | 0/3 | 0/3 | 2/3 | 0/3 |
| Qwen3 | 0/3 | 0/3 | 0/3 | 0/3 |

---

## Runs with correct RCA (13/48)

| Agent | Context | Fault | MTTD |
|-------|---------|-------|------|
| Llama Scout | C1 | config_corruption | 25s |
| Llama Scout | C1 | kill_pod | 28s |
| Llama Scout | C1 | scale_zero | 25s |
| Llama Scout | C2 | kill_pod | 25s |
| Llama Scout | C2 | scale_zero | 65s |
| Nemotron | C2 | scale_zero | 51s |
| DeepSeek | C3 | config_corruption | 276s |
| Llama Scout | C3 | kill_pod | 72s |
| DeepSeek | C3 | scale_zero | 63s |
| Llama Scout | C4 | config_corruption | 27s |
| Nemotron | C4 | config_corruption | 70s |
| Nemotron | C4 | kill_pod | 46s |
| Llama Scout | C4 | scale_zero | 28s |

---

## Comparison to June 2026 Matrix

| Dimension | June 2026 | July 2026 (this run) |
|-----------|-----------|----------------------|
| Scenarios | 0, A, B, C | **A only** |
| Faults | 8 | 3 (cart-focused) |
| Context model | L0–L2 (auto-inject + tools) | **C0–C4 (tool-call RAG only)** |
| Best RCA | 12% (any agent) | **67% (Llama Scout)** |
| RAG usage | Some `rag_search` at L2 | **0% — no RAG calls** |
| Judge | Declared RCA only | **Hybrid oracle judge** |
| Key blocker | Chronic log noise | Still log noise + RAG unused |

The RCA improvement is primarily **model selection (Llama Scout)** and **stricter cart-oracle scoring**, not context engineering. The C-model infrastructure is in place (per-corpus indexes, MLflow `context_c` tags, `artifacts/context_c.json`) but agents did not consume it.

---

## Recommendations

1. **Default detector: Llama Scout** on scenario A for speed (27s MTTD) and RCA (67%). Use Nemotron for remediation attempts when detection hands off.

2. **Fix Qwen3 integration** — investigate MaaS `qwen3-14b` empty responses before including in future matrices.

3. **Force delta-first investigation** — prompt or policy requiring `compare_telemetry` (or pre/post log diff) before `search_logs` to reduce noise-driven misdiagnosis.

4. **Make RAG actionable** — options:
   - Require at least one `rag_search_*` call when `CONTEXT_C≠0`
   - Inject a short “service map” snippet at session start (hybrid)
   - Add tool descriptions that tie corpora to RCA (“use C4 to find cart’s deployment name”)

5. **Re-run matrix with C0 baseline** — set `INCLUDE_C0=1` to quantify whether any corpus beats pure telemetry+K8s.

6. **Extend to scenarios B/C with best agent** — once RAG usage is enforced, test whether domain/signal specialists benefit more from C2 docs or C4 dependencies.

7. **Recovery verification** — 0% `recovery_verified` indicates post-remediation checks are not passing; tighten harness verification loop.

---

## Artifacts & Reproduction

```bash
# Re-run the matrix (scenario A, C1–C4, 48 runs)
./scripts/run_c_matrix.sh

# Include C0 baseline (60 runs)
INCLUDE_C0=1 ./scripts/run_c_matrix.sh

# Summarize JSON outputs
python scripts/analyze_context_matrix.py --out-dir out --report docs/CONTEXT_ENGINEERING_EVAL_REPORT.md
```

**Configuration references:**

- Scenarios: `config/scenarios.yaml`
- Context corpora: `config/context_c_levels.yaml`
- Matrix script: `scripts/run_c_matrix.sh`

**Related reports:**

- [EVALUATION_RESULTS_23June26.md](EVALUATION_RESULTS_23June26.md) — full 128-run scenario matrix
- [CONTEXT_ENGINEERING_EVAL_REPORT.md](CONTEXT_ENGINEERING_EVAL_REPORT.md) — auto-generated per-run table (includes legacy L-model runs)

---

*Generated 7 July 2026 from 48 `out/context_C*_sa_*.json` harness results.*
