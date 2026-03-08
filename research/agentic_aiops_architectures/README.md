# Agentic AIOps Architectures

Framework for comparing agentic AIOps approaches: MTTD (Mean Time To Detect), MTTR (Mean Time To Resolve), and remediation suggestion quality.

## Architecture

- **Telemetry**: OTEL Demo app → Collector → ClickHouse (logs, traces, metrics)
- **Experiment tracking**: MLflow
- **Reference agent**: Rule-based detection via ClickHouse queries
- **Harness**: Chaos injection (flagd) → agent polling → MTTD/MTTR + MLflow logging

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

# Port-forward ClickHouse + MLflow
oc port-forward -n otel-demo svc/clickhouse 8123:8123 &
oc port-forward -n agentic-aiops svc/mlflow 5000:5000 &

export CLICKHOUSE_HTTP="http://localhost:8123"
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Run single experiment
python code/harness/run_harness.py --flag cartFailure --variant on -o out/run.json
```

### 4. View MLflow

```bash
# MLflow UI: https://mlflow-agentic-aiops.apps.sno1gpu.localdomain
# Or port-forward: oc port-forward -n agentic-aiops svc/mlflow 5000:5000
```

## Components

| Component | Purpose |
|-----------|---------|
| `manifests/clickhouse.yaml` | ClickHouse for OTEL data |
| `manifests/mlflow.yaml` | MLflow for experiment tracking |
| `scripts/patch-otel-collector-clickhouse.py` | Add ClickHouse to collector pipelines |
| `code/agents/baseline_classifier/classifier.py` | Baseline single-shot classifier (threshold-based) |
| `code/agents/reference_agent/agent.py` | LLM-based single-shot (optional) |
| `code/harness/run_harness.py` | MTTD/MTTR harness with MLflow logging |
| `data/event_ledger.jsonl` | Event log (fault_injection, first_alert, fault_recovery) |

## Baseline Classifier (single-shot)

**`code/agents/baseline_classifier/`** — Rule-based, non-agentic baseline:
- Fetches telemetry summary from ClickHouse (error counts, top services, high-latency spans)
- Applies thresholds: 5+ errors or 3+ high-latency spans (>5s) → detected
- Maps top error service to remediation steps via lookup table
- No LLM, no tool calls, no iteration

```bash
CLICKHOUSE_HTTP=http://localhost:8123 python code/agents/baseline_classifier/classifier.py --since 2025-01-01T00:00:00Z --json
```

## Reference Agent (LLM-based, optional)

**`code/agents/reference_agent/`** — LLM-assisted single-shot (also non-agentic):
- Same telemetry fetch, sends context to OpenAI-compatible LLM
- Use with `--classifier path/to/reference_agent/agent.py` if desired

## Harness Events

- `fault_injection`: When flag is set
- `first_alert`: When agent first detects anomaly
- `fault_recovery`: When flag is cleared

MTTD = first_alert_time - fault_injection_time  
MTTR = fault_recovery_time - first_alert_time  

## Docs

- [ARCHITECTURE_PROPOSAL.md](docs/ARCHITECTURE_PROPOSAL.md) — Full design
- [OPENSHIFT_OTEL_DEPLOYMENT.md](docs/OPENSHIFT_OTEL_DEPLOYMENT.md) — OpenShift setup
