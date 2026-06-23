# Full Matrix Evaluation Results — 23 June 2026

## Run Summary

- **128 runs** completed (4 agents × 8 faults × 4 scenarios)
- **132 result files** recovered (some scenarios produce extra runs)
- **200 MLflow runs** logged at `http://localhost:5050`
- **Runtime**: ~12 hours overnight

## Top-Line Results

| Metric | Value |
|---|---|
| Overall detection rate | 108/132 (82%) |
| Undetectable fault (readiness_probe_fail) | 0/16 (0%) — no agent can catch it |
| Best detection rate (excluding undetectable) | replica_overload 100%, scale_zero 95% |
| Fastest agent (avg MTTD) | **Llama Scout: 46s** |
| Most reliable fixer (MTTR) | **Nemotron: 8 autonomous fixes** |
| Best scenario for detection | **Scenario A (telemetry+K8s): 89%** |

## Key Insights

### 1. Speed and action are inversely correlated

| Agent | Avg MTTD | Fixes executed | Insight |
|---|---|---|---|
| Llama Scout | **46s** | 0 | Fastest detector, never acts |
| Nemotron | 140s | **8** | Slow but always fixes what it finds |
| Qwen3 | 138s | 4 | Moderate speed, sometimes fixes |
| DeepSeek | 171s | 3 | Slowest, occasionally fixes |

No single model excels at both speed AND action. A multi-model pipeline (Llama Scout for detection → Nemotron for remediation) would be optimal.

### 2. Scenario A (telemetry + K8s tools) wins

| Scenario | Detection Rate | Why |
|---|---|---|
| **A** (telemetry + K8s) | **89%** | Full toolset = best detection |
| C (domain specialists) | 84% | Multi-agent helps but adds coordination overhead |
| 0 (telemetry only) | 81% | Can still detect most faults from logs/traces alone |
| B (signal specialists) | 72% | Splitting by signal type hurts — no single signal is sufficient |

Giving one agent ALL tools outperforms splitting tools across specialists.

### 3. readiness_probe_fail is invisible to all agents

0% detection across all 4 agents and all 4 scenarios. This fault produces minimal telemetry signal — the pod runs but isn't marked ready, so traffic stops routing silently. Detecting this requires:
- A tool that explicitly checks endpoint readiness
- Or historical traffic baselines (no requests = anomaly)

### 4. Qwen3 struggles in multi-agent scenarios

| Scenario | Qwen3 detection |
|---|---|
| 0 (single, telemetry) | 7/8 (88%) |
| A (single, full tools) | 8/9 (89%) |
| B (multi, specialists) | **3/8 (38%)** |
| C (multi, domain) | 6/8 (75%) |

Qwen3 drops significantly in scenario B where it only gets one signal type per role. It needs broad tool access to reason effectively.

### 5. DeepSeek's prompt-based tool calling limits it to 1 tool per run

In nearly every run, DeepSeek calls only `search_logs` — never progressing to traces, metrics, or K8s tools. The prompt-based tool calling regex approach means the model produces one tool call, gets results, and immediately concludes. Despite this handicap, it still achieves 85% detection by making good use of that single log search.

### 6. Remediation is unreliable across all agents

Total autonomous fixes across 128 runs: only **15** (12%). Most detections result in suggestions without action. The agents detect and diagnose but resist executing write operations. This is a fundamental LLM behavior pattern — models are cautious about side effects despite explicit instructions to act.

## Detection Heatmap

| Fault | nemotron | qwen3 | deepseek | llama_scout | Total |
|---|---|---|---|---|---|
| scale_zero | ✓✓✓✓✓✓✓✓ | ✓✓✓✓✓✗✓✓ | ✓✓✓✓✓✓✓✗ | ✓✓✓✓✓✓✓✗ | 95% |
| kill_pod | ✓✓✓✓ | ✓✓✓✗ | ✓✓✓✓ | ✓✓✓✓ | 88% |
| memory_limit | ✓✓✓✓ | ✓✓✓✓ | ✓✓✓✗ | ✓✓✓✓ | 94% |
| network_partition | ✓✓✓✓ | ✓✓✓✓ | ✓✓✓✗ | ✓✓✓✓ | 94% |
| config_corruption | ✓✓✓✓ | ✓✓✗✓ | ✓✓✓✓ | ✓✓✓✗ | 88% |
| dependency_removal | ✓✓✓✓ | ✓✓✓✓ | ✓✓✓✗ | ✓✓✓✓ | 94% |
| replica_overload | ✓✓✓✓ | ✓✓✓✓ | ✓✓✓✓ | ✓✓✓✓ | 100% |
| readiness_probe_fail | ✗✗✗✗ | ✗✗✗✗ | ✗✗✗✗ | ✗✗✗✗ | 0% |

## Recommendations

1. **Production pipeline**: Use Llama Scout (22s MTTD) as the fast first-responder detector, then hand off to Nemotron for diagnosis + fix execution.

2. **Add readiness check tool**: A `check_endpoints(service)` tool that queries the K8s Endpoints API would make readiness_probe_fail detectable.

3. **Fix DeepSeek's tool usage**: Either switch to native tool calling (requires a different model variant) or add explicit "call at least 3 tools" enforcement in the prompt-based parser.

4. **Don't split tools across agents** (scenario B): A single agent with all tools consistently outperforms specialized agents with limited views.

5. **Enforce remediation execution**: The 12% fix rate needs improvement. Options: stronger prompting, code-level auto-execution of suggested fixes, or a dedicated remediation agent that receives the diagnosis and always acts.
