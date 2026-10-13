# Evaluation Results

## Full Scenario Comparison (June 2026)

All agents evaluated against `scale_zero` fault (cart deployment scaled to 0 replicas) across all four scenarios.

### Detection Results by Scenario

| Scenario | Agent | Detected | MTTD | MTTR |
|---|---|---|---|---|
| **0** (telemetry only) | nemotron-nano-3 | Yes | 139.3s | - |
| **0** | maas_deepseek | Yes | 152.0s | - |
| **0** | maas_qwen3 | No | - | - |
| **0** | maas_llama-scout | Yes | 56.4s | - |
| **a** (telemetry + K8s) | nemotron-nano-3 | Yes | 66.1s | 66.1s |
| **a** | maas_deepseek | Yes | 69.5s | - |
| **a** | maas_qwen3 | Yes | 67.5s | - |
| **a** | maas_llama-scout | Yes | 60.2s | - |
| **b** (signal specialists) | nemotron-nano-3 | No | - | - |
| **b** | maas_deepseek | Yes | 188.5s | - |
| **b** | maas_qwen3 | No | - | - |
| **b** | maas_llama-scout | Yes | 29.8s | - |
| **c** (domain specialists) | nemotron-nano-3 | Yes | 200.9s | - |
| **c** | maas_deepseek | Yes | 188.7s | - |
| **c** | maas_qwen3 | Yes | 242.0s | - |
| **c** | maas_llama-scout | Yes | 34.3s | - |

### Key Findings

1. **Scenario A (telemetry + K8s) achieves 100% detection** — all agents detect when given both telemetry and Kubernetes tools.

2. **Llama Scout is consistently fastest** at detection (29-60s) across all scenarios despite being the smallest model.

3. **Scenario B (signal specialists) is weakest** — splitting tools across specialist agents hurts detection. Only 2/4 agents detect.

4. **K8s tools significantly improve detection** — Scenario A detects reliably; Scenario 0 (telemetry only) misses 1/4 because `scale_zero` produces minimal log noise.

5. **Nemotron is the only agent that executed a real fix** (MTTR = 66.1s in scenario A) — other agents detected but didn't call write tools.

### Multi-Fault Campaign (Nemotron, Scenario A)

| Fault Type | Detected | MTTD | MTTR | Fix Executed |
|---|---|---|---|---|
| scale_zero | Yes | 65.1s | 52.3s | restart_deployment |
| kill_pod | Yes | 64.3s | 54.6s | restart_deployment |
| memory_limit | Yes | 58.3s | 44.6s | restart_deployment |
| network_partition | Yes | 99.2s | 37.1s | restart_deployment |
| readiness_probe_fail | No | - | - | - |
| config_corruption | Yes | 78.8s | 73.4s | restart_deployment |
| dependency_removal | Yes | 40.5s | 36.4s | scale_deployment |
| replica_overload | Yes | 64.9s | 45.1s | scale_deployment |

**7/8 faults detected and remediated autonomously.** Only `readiness_probe_fail` missed (pod runs but isn't ready — minimal telemetry signal).

### AI Performance Metrics (Nemotron-Nano-3)

| Metric | Value |
|---|---|
| Average TTFT | 1.0-1.2s |
| Tokens/second | 177 |
| Average rounds per detection | 5-8 |
| Average total tokens per run | 30-50K |
| Tool calls per detection | 5-7 |

### Metric Definitions

- **MTTD** (Mean Time To Detect): `fault_injection → agent decides there is a fault` (timestamp recorded just before first write tool call or at agent conclusion)
- **MTTR** (Mean Time To Repair): `fault_injection → system fixed` (timestamp recorded when write tool API call succeeds)
- **MTTR = None**: Agent detected but did not execute a fix (only suggested remediations)

### Scenarios

| ID | Name | Tools Available | Purpose |
|---|---|---|---|
| 0 | baseline_telemetry | search_logs, search_traces, search_metrics, query_clickhouse | Baseline — can telemetry alone detect? |
| a | multimodal_telemetry_k8s | All telemetry + get_pod_status, get_events, restart_deployment, scale_deployment, delete_pod | Full autonomous — detect + diagnose + fix |
| b | signal_specialists | Multi-agent: logs_only / traces_only / metrics_only | Does signal specialization help? |
| c | domain_specialists | Multi-agent: hardware / platform / application | Does domain specialization help? |

### Agents

| Agent | Model | Provider | Tool Calling |
|---|---|---|---|
| nemotron-nano-3 | NVIDIA Nemotron-3-Nano | vLLM/NIM | Native + reasoning |
| maas_qwen3 | Qwen3-14B | LiteLLM | Native |
| maas_deepseek | DeepSeek R1 Distill 14B | LiteLLM | Prompt-based |
| maas_llama-scout | Llama Scout 17B | LiteLLM | Native |
