# Agentic AIOps Architectures

Framework for comparing agentic AIOps approaches: MTTD (Mean Time To Detect), MTTR (Mean Time To Resolve), and remediation suggestion quality.

## Architecture

- **Telemetry**: OTEL Demo app → Collector → ClickHouse (logs, traces, metrics)
- **Experiment tracking**: MLflow
- **Agents**: ollama_qwen (Ollama), maas_deepseek (DeepSeek R1), maas_qwen3 (Qwen3-14B), maas_llama-scout (Llama Scout 17B)
- **Harness**: Chaos injection (flagd) → agent polling → MTTD/MTTR + MLflow logging

### Agentic Framework

We use a **minimal custom approach** — no LangChain, LlamaIndex, AutoGen, or CrewAI. Both agents use an **agentic tool-calling loop**: the LLM is given tools (`query_clickhouse`, `search_logs`, `search_traces`, `search_metrics`, `run_remediation`) and iteratively calls them to gather telemetry, reason, then output detection + remediations. `run_remediation` logs remediation steps to MLflow (no execution yet).

## Quick Start

### 1. Deploy on OpenShift

```bash
# ClickHouse (already in otel-demo)
oc apply -f manifests/clickhouse.yaml

# Patch OTEL collector for ClickHouse exporter (if not done)
python scripts/patch-otel-collector-clickhouse.py

# MLflow
oc apply -f manifests/mlflow.yaml
```

### 2. Verify telemetry flow

```bash
# Port-forward ClickHouse
oc port-forward -n otel-demo svc/clickhouse 8123:8123 &

# After load generator produces traffic (visit frontend, enable loadgen)
curl -s 'http://localhost:8123/?query=SELECT%20count()%20FROM%20otel.otel_logs'
curl -s 'http://localhost:8123/?query=SELECT%20count()%20FROM%20otel.otel_traces'
```

### 3. Run harness

```bash
# Set flagd URLs (see docs/WORKING_URLS.md)
# OpenShift: use the direct flagd-ui-api route for reliable read/write:
export FLAGD_READ_URL="https://$(oc get route flagd-ui-api -n otel-demo -o jsonpath='{.spec.host}')/api/read"
export FLAGD_WRITE_URL="https://$(oc get route flagd-ui-api -n otel-demo -o jsonpath='{.spec.host}')/api/write"
# Or via frontend: .../feature/api/read and .../feature/api/write
# For self-signed cluster certs: export FLAGD_VERIFY_SSL=false

# Port-forward ClickHouse (MLflow uses the OpenShift Route URL, not a local server)
oc port-forward -n otel-demo svc/clickhouse 8123:8123 &

export CLICKHOUSE_HTTP="http://localhost:8123"
export MLFLOW_TRACKING_URI="https://$(oc get route mlflow -n agentic-aiops -o jsonpath='{.spec.host}')"
export MLFLOW_TRACKING_INSECURE_TLS=true

# Run single experiment
python code/harness/run_harness.py --flag cartFailure --variant on -o out/run.json
```

### 4. View MLflow

Open the **OpenShift Route** for MLflow in your browser (same host as `MLFLOW_TRACKING_URI`). Example:

`https://mlflow-agentic-aiops.apps.sno1gpu.localdomain`

Adjust the host to match your cluster (`oc get route mlflow -n agentic-aiops`).

## Components

| Component | Purpose |
|-----------|---------|
| `manifests/clickhouse.yaml` | ClickHouse for OTEL data |
| `manifests/mlflow.yaml` | MLflow for experiment tracking |
| `scripts/patch-otel-collector-clickhouse.py` | Add ClickHouse to collector pipelines |
| `code/agents/ollama_qwen2/agent.py` | LLM agent (Ollama Qwen 2.5) — native tool calling |
| `code/agents/deepseek_agent/agent.py` | LLM agent (DeepSeek R1 Distill 14B) — prompt-based tool calling |
| `code/agents/qwen3_agent/agent.py` | LLM agent (Qwen3-14B) — native tool calling |
| `code/agents/llama_scout_agent/agent.py` | LLM agent (Llama Scout 17B) — native tool calling |
| `code/tools/agent_tools.py` | Shared tools: query_clickhouse, search_logs, search_traces, search_metrics, run_remediation |
| [docs/CLICKHOUSE_QUERIES.md](docs/CLICKHOUSE_QUERIES.md) | Tables, schema summary, telemetry & correlation SQL (TraceId, time windows) |
| `code/harness/run_harness.py` | MTTD/MTTR harness with MLflow logging |
| `data/event_ledger.jsonl` | Event log (fault_injection, first_alert, fault_recovery) |

## Agents

All agents use the same tools (`search_logs`, `search_traces`, `search_metrics`, `query_clickhouse`, `run_remediation`) and return structured detection + remediation JSON.

| Agent | Model | Tool calling | Env vars |
|-------|-------|-------------|----------|
| `ollama_qwen2/` | Qwen 2.5 (local Ollama) | Native | `OPENAI_API_BASE`, `OPENAI_MODEL` |
| `deepseek_agent/` | DeepSeek R1 Distill 14B | Prompt-based | `DEEPSEEK_API_KEY` |
| `qwen3_agent/` | Qwen3-14B | Native | `QWEN3_API_KEY` |
| `llama_scout_agent/` | Llama Scout 17B | Native | `LLAMA_SCOUT_API_KEY` |

API keys go in `config/.env` (gitignored) — `run_harness.sh` sources it automatically.

```bash
# Run all agents
./scripts/run_harness.sh --flag cartFailure --variant on

# Run a specific agent
./scripts/run_harness.sh --classifier code/agents/qwen3_agent/agent.py --flag cartFailure --variant on
```

## Harness Events

- `fault_injection`: When flag is set
- `first_alert`: When agent first detects anomaly
- `fault_recovery`: When flag is cleared

MTTD = first_alert_time - fault_injection_time  
MTTR = fault_recovery_time - first_alert_time  

## Docs

- [ARCHITECTURE_PROPOSAL.md](docs/ARCHITECTURE_PROPOSAL.md) — Full design
- [LLM_CREDENTIALS.md](docs/LLM_CREDENTIALS.md) — API key setup and MLflow logging
- [OPENSHIFT_OTEL_DEPLOYMENT.md](docs/OPENSHIFT_OTEL_DEPLOYMENT.md) — OpenShift setup
