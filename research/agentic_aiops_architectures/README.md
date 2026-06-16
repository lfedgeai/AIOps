# Agentic AIOps Architectures

Framework for comparing agentic AIOps approaches: MTTD (Mean Time To Detect), MTTR (Mean Time To Resolve), and remediation suggestion quality.

## Architecture

- **Telemetry**: OTEL Demo app → Collector → ClickHouse (logs, traces, metrics)
- **Experiment tracking**: MLflow
- **Agents**: ollama_qwen (Ollama), maas_deepseek (DeepSeek R1), maas_qwen3 (Qwen3-14B), maas_llama-scout (Llama Scout 17B)
- **Harness**: Fault injection (flagd or Kubernetes API) → agent polling → MTTD/MTTR + MLflow logging

### Agentic Framework

We use a **minimal custom approach** — no LangChain, LlamaIndex, AutoGen, or CrewAI. All agents use an **agentic tool-calling loop**: the LLM is given telemetry tools (`search_logs`, `search_traces`, `search_metrics`, `query_clickhouse`) and Kubernetes tools (`get_pod_status`, `get_events`, `restart_deployment`, `scale_deployment`, …) and iteratively calls them to detect faults, diagnose root cause, and remediate. `run_remediation` logs what the agent did to MLflow; other tools can execute fixes on the cluster.

## Quick Start

### 0. Prerequisites

```bash
oc login <your-cluster-api-url>
cp config/.env.example config/.env   # add MaaS API keys (see docs/LLM_CREDENTIALS.md)
```

### 1. Deploy on OpenShift

```bash
./scripts/deploy-openshift-stack.sh
# Optional: wire EvalHub pod env with Flagd / ClickHouse / harness MLflow URLs
./scripts/patch-evalhub-lfedge-aiops-env.sh
```

The script installs the **OpenTelemetry Demo** Helm chart into **`otel-demo`**, applies **ClickHouse**, patches the demo **otel-collector** to export **logs, metrics, and traces** into ClickHouse, deploys **MLflow** in **`agentic-aiops`**, and exposes the **flagd** API via **Route** (`flagd-ui-api`).  
By default **`HELM_WAIT=false`** so the install does not block on every demo pod (some components may stay `CrashLoopBackOff` on small clusters while the collector + ClickHouse path still works).

Manual variant:

```bash
# ClickHouse (otel-demo)
oc apply -f manifests/clickhouse.yaml

# Patch OTEL collector for ClickHouse exporter (if not done)
python scripts/patch-otel-collector-clickhouse.py

# MLflow
oc apply -f manifests/mlflow.yaml

# Flagd API route (for harness FLAGD_* URLs)
oc apply -f manifests/flagd-ui-route.yaml
```

### 2. Verify stack (after deploy or cluster rebuild)

```bash
oc get deploy -n otel-demo flagd clickhouse otel-collector
oc get deploy -n agentic-aiops mlflow
oc get route -n otel-demo flagd-ui-api
oc get route -n agentic-aiops mlflow

# Port-forward ClickHouse
oc port-forward -n otel-demo svc/clickhouse 8123:8123 &

# Generate traffic (demo frontend route or load generator), then confirm ingest:
curl -s 'http://localhost:8123/?query=SELECT%20count()%20FROM%20otel.otel_logs'
curl -s 'http://localhost:8123/?query=SELECT%20count()%20FROM%20otel.otel_traces'
```

Copy `config/openshift.local.yaml.example` to `config/openshift.local.yaml` (gitignored) or set `OPENSHIFT_CONTEXT` / `OPENSHIFT_API_URL` for your cluster. Inspect merged settings with `python -m code.harness.openshift_config`.

### 3. Run harness

Prefer **`./scripts/run_harness.sh`** — it sources `config/.env`, port-forwards ClickHouse if needed, discovers MLflow/flagd routes, and enables the OpenShift ConfigMap flagd workaround when `flagd-config` exists.

```bash
# Default: K8s platform fault (scale deployment to zero)
./scripts/run_harness.sh --classifier code/agents/qwen3_agent/agent.py --flag scale_zero --variant cart

# Scenario b: logs / traces / metrics specialists (3 agents orchestrated)
./scripts/run_harness.sh --scenario b --classifier code/agents/qwen3_agent/agent.py --flag scale_zero --variant cart

# All agents x all scenarios (campaign)
./scripts/run_scenario_campaign.sh

# Optional: application fault via flagd (non-default)
./scripts/run_harness.sh --use-flagd --flag cartFailure --variant on --classifier code/agents/qwen3_agent/agent.py
```

Manual env (if not using `run_harness.sh`):

```bash
# flagd HTTP API (optional — run_harness.sh uses ConfigMap patch when flagd-config exists)
export FLAGD_READ_URL="https://$(oc get route flagd-ui-api -n otel-demo -o jsonpath='{.spec.host}')/api/read"
export FLAGD_WRITE_URL="https://$(oc get route flagd-ui-api -n otel-demo -o jsonpath='{.spec.host}')/api/write"
export FLAGD_VERIFY_SSL=false   # self-signed cluster routes

oc port-forward -n otel-demo svc/clickhouse 8123:8123 &
export CLICKHOUSE_HTTP="http://localhost:8123"
export MLFLOW_TRACKING_URI="https://$(oc get route mlflow -n agentic-aiops -o jsonpath='{.spec.host}')"
export MLFLOW_TRACKING_INSECURE_TLS=true

source config/.env
python code/harness/run_harness.py --classifier code/agents/qwen3_agent/agent.py --scenario a --flag scale_zero --variant cart -o out/run.json
```

### 4. View MLflow

Open the **OpenShift Route** for MLflow in your browser (same host as `MLFLOW_TRACKING_URI`):

```bash
oc get route mlflow -n agentic-aiops -o jsonpath='https://{.spec.host}{"\n"}'
```

### 5. EvalHub integration

- **`scripts/patch-evalhub-lfedge-aiops-env.sh`** adds connection hints to the **`EvalHub`** CR in **`rhods-notebooks`**: `LFEDGE_AIOPS_FLAGD_READ_URL`, `LFEDGE_AIOPS_FLAGD_WRITE_URL`, `LFEDGE_AIOPS_CLICKHOUSE_HTTP`, `LFEDGE_AIOPS_MLFLOW_URI`, `LFEDGE_AIOPS_OTEL_NAMESPACE`. BYOF jobs or future adapters can read these; the EvalHub UI still uses **Develop & train → Evaluations** (not a separate `/` HTML app).
- **`evalhub/lfedge-aiops-comparison.collection.yaml`** is a sample **collection** definition for **`lm_evaluation_harness`** (e.g. GSM8K) so you can register a baseline leaderboard task next to harness runs logged in MLflow.
- **`evalhub/lfedge_aiops_provider/`** — EvalHub **Kubernetes adapter** that reports **`mttd_seconds`**, **`mttr_seconds`**, and ClickHouse row counts (benchmark **`mttd_mttr_clickhouse`** under provider id **`lfedge_aiops`**). Deploy with **`scripts/deploy-evalhub-lfedge-aiops-provider.sh`** (builds the image in **`rhods-notebooks`**, applies the provider ConfigMap to **`redhat-ods-applications`** with a **TrustyAI `ownerReference`**, patches **`EvalHub.spec.providers`**). Submit a smoke job with **`scripts/submit-evalhub-lfedge-aiops-job.sh`** (requires **`X-Tenant`** — the script uses the notebook namespace). API calls also need **`Authorization: Bearer $(oc whoami -t)`** against the **`evalhub`** Route.
- **“To use evaluations, enable the evaluation service using the TrustyAI Operator”** — The **eval-hub-ui** health API (`/eval-hub/api/v1/evalhub/health`) looks for an **`EvalHub` CR named `evalhub` in `redhat-ods-applications`**, not only in the workbench namespace. If EvalHub exists only in **`rhods-notebooks`**, health returns **`cr-not-found`** / **`available: false`** and the UI shows this banner. Run **`scripts/enable-rhoai-evaluations-ui.sh`** (labels the workbench namespace `evalhub.trustyai.opendatahub.io/tenant=true`, ensures **`spec.database`** on the tenant CR, and creates the platform **`EvalHub`** in **`redhat-ods-applications`**). Your **tenant** jobs still use **`rhods-notebooks/evalhub`** (and **`X-Tenant`** on the API).
- **`harness-engineering` in the console** — DSC **`workbenchNamespace`** is still **`rhods-notebooks`**, so pick the tenant via URL query (hard refresh after fixes):
  - Evaluations: `https://rh-ai.apps.<cluster>/develop-train/evaluations?namespace=harness-engineering`
  - MLflow: `https://rh-ai.apps.<cluster>/develop-train/mlflow/experiments?workspace=harness-engineering`
- **Evaluations (Federated Mode) UI — “Error loading components”** — **What the page should show:** for the selected workbench namespace (e.g. **`rhods-notebooks`**), the federated Evaluations experience should load **evaluation jobs** (table with status, name, timestamps), **providers** and **collections**, and flows to **create / inspect** runs—same data you see when calling the EvalHub API with a tenant context. **Why it breaks today (two separate mismatches):** (1) The dashboard **`eval-hub-ui`** BFF calls **`GET /api/v1/evaluations/jobs?namespace=…`** **without** the **`X-Tenant`** header; EvalHub’s shipped **`auth.yaml`** (from the TrustyAI operator, see [configmap.go `generateAuthConfigData`](https://github.com/trustyai-explainability/trustyai-service-operator/blob/main/controllers/evalhub/configmap.go)) maps tenancy only from **`X-Tenant`**, so authorization fails with **`required header X-Tenant is missing`**. (2) Even if auth were relaxed to treat **`namespace`** as the tenant for GET/HEAD, the EvalHub **HTTP layer** still validates OpenAPI query parameters and returns **`query_bad_parameter`** because **`namespace` is not an allowed query parameter** for list jobs (allowed parameters are along the lines of **`limit`**, **`offset`**, **`status`**, **`name`**, **`tags`**, **`owner`**, **`experiment_id`**). So the UI and server contract must be fixed together in **RHOAI / EvalHub / TrustyAI** (BFF should send **`X-Tenant`** and/or EvalHub should accept **`namespace`** for federated list in both auth and API validation). **Workaround:** use **`curl`** or scripts against the **`evalhub`** Route with **`Authorization: Bearer $(oc whoami -t)`** and **`X-Tenant: your-notebook-namespace`** (see **`scripts/submit-evalhub-lfedge-aiops-job.sh`**). **Not durable on-cluster:** patching **`deployment/rhods-dashboard`** is reverted within seconds by the **Dashboard** controller; **`OdhDashboardConfig`** has no EvalHub BFF URL field in releases checked here. Optional lab-only nginx proxy (does not fix the query-parameter issue by itself): **`scripts/patch-rhods-dashboard-evalhub-ui-tenant-proxy.sh`** and **`evalhub/manifests/evalhub-tenant-header-proxy.*.yaml`**. Reference auth shape (incomplete alone): **`evalhub/manifests/evalhub-config-auth-federated-ui.data.yaml`**.

## Components

| Component | Purpose |
|-----------|---------|
| `manifests/clickhouse.yaml` | ClickHouse for OTEL data (4Gi limit, ConfigMap caps server RAM to avoid OOM on ingest) |
| `manifests/mlflow.yaml` | MLflow for experiment tracking |
| `manifests/flagd-ui-route.yaml` | OpenShift Route for flagd HTTP API (harness) |
| `scripts/deploy-openshift-stack.sh` | One-shot Helm + manifests + collector patch |
| `scripts/patch-evalhub-lfedge-aiops-env.sh` | Patch EvalHub CR with AIOps env vars |
| `scripts/deploy-evalhub-lfedge-aiops-provider.sh` | Build adapter image + register `lfedge_aiops` EvalHub provider (operator namespace + EvalHub CR) |
| `scripts/submit-evalhub-lfedge-aiops-job.sh` | POST a smoke evaluation job (`mttd_mttr_clickhouse`) |
| `scripts/enable-rhoai-evaluations-ui.sh` | Fix “enable evaluation service” banner (platform **`EvalHub`** in **`redhat-ods-applications`** + tenant label) |
| `scripts/fix-evalhub-tenant-rbac.sh` | Platform **`evalhub-service`** ConfigMap create in tenant NS (operator cross-wire workaround) |
| `scripts/patch-rhods-dashboard-evalhub-ui-tenant-proxy.sh` | Optional: deploy nginx **`X-Tenant`** proxy only (no durable UI fix; see README §5) |
| `evalhub/manifests/evalhub-config-auth-federated-ui.data.yaml` | Reference only: alternate **`auth.yaml`** shape for RH support (not sufficient alone; see README §5) |
| `evalhub/manifests/trustyai-operator-evalhub-provider-lfedge-aiops.configmap.yaml` | TrustyAI provider ConfigMap template (`__IMAGE__` placeholder) |
| `evalhub/lfedge_aiops_provider/` | EvalHub adapter container (`main.py` + `Dockerfile`) |
| `evalhub/lfedge-aiops-comparison.collection.yaml` | Sample EvalHub collection (LM baseline) |
| `scripts/patch-otel-collector-clickhouse.py` | Add ClickHouse to collector pipelines |
| `scripts/set-flag-openshift.py` | Patch `flagd-config` ConfigMap + restart flagd (OpenShift fault injection) |
| `scripts/run_scenario_campaign.sh` | Run all agents across scenarios a, b, c |
| `config/scenarios.yaml` | Scenario definitions (tool profiles per scenario) |
| `config/fault_ground_truth.yaml` | Expected root cause per fault (RCA scoring) |
| `code/tools/tool_profiles.py` | Tool gating per scenario role |
| `scripts/run_harness.sh` | Wrapper: sources `.env`, port-forward, route discovery, runs harness |
| `code/agents/scenario_orchestrator/agent.py` | Multi-agent orchestrator for scenarios b and c |
| `code/agents/ollama_qwen2/agent.py` | LLM agent (Ollama Qwen 2.5) — native tool calling |
| `code/agents/deepseek_agent/agent.py` | LLM agent (DeepSeek R1 Distill 14B) — prompt-based tool calling |
| `code/agents/qwen3_agent/agent.py` | LLM agent (Qwen3-14B) — native tool calling |
| `code/agents/llama_scout_agent/agent.py` | LLM agent (Llama Scout 17B) — native tool calling |
| `code/tools/agent_tools.py` | Shared tools: ClickHouse search, Kubernetes inspect/remediate, MLflow logging |
| [docs/CLICKHOUSE_QUERIES.md](docs/CLICKHOUSE_QUERIES.md) | Tables, schema summary, telemetry & correlation SQL (TraceId, time windows) |
| `code/harness/run_harness.py` | MTTD/MTTR harness with MLflow logging |
| `data/event_ledger.jsonl` | Event log (fault_injection, first_alert, fault_recovery) |

## Fault injection

The harness supports two fault modes (see `code/harness/run_harness.py`):

| `--flag` | Mechanism | `--variant` meaning | Example |
|----------|-----------|---------------------|---------|
| `scale_zero`, `kill_pod`, `memory_limit` | **Kubernetes API** (default) — harness patches deployments/pods in `otel-demo` | target deployment name | `--flag scale_zero --variant cart` |
| `cartFailure`, etc. | **flagd** (optional, `--use-flagd`) — app-level chaos in microservices | flagd variant (`on`, `off`, …) | `--use-flagd --flag cartFailure --variant on` |

On OpenShift, app faults usually go through **`scripts/set-flag-openshift.py`** (patches `flagd-config` ConfigMap + restarts `deployment/flagd`) because the flagd-ui HTTP write API often returns 404. `run_harness.sh` sets `FLAGD_USE_OPENSHIFT_CM=1` automatically when that ConfigMap exists.

Generate traffic before running experiments (visit the demo frontend or enable the load generator) so faults produce logs/traces/metrics in ClickHouse.

## Experiment scenarios

Same fault injection runs across three agent topologies (`config/scenarios.yaml`):

| Scenario | Mode | Tool access |
|----------|------|-------------|
| **a** | Single agent | logs + traces + metrics (no K8s API) |
| **b** | Three specialists | logs-only, traces-only, metrics-only agents (orchestrated) |
| **c** | Three domain agents | hardware metrics, platform K8s, application logs/traces |

Harness flags: `--scenario a|b|c`, `--all-scenarios`, `--use-flagd` (optional flagd faults).

RCA accuracy is scored against `config/fault_ground_truth.yaml` and logged to MLflow as `rca_correct` and `remediation_correct`.

## Agents

Agents read `TOOL_PROFILE` from the harness and only receive matching tools (`code/tools/tool_profiles.py`).

| Agent | Model | Tool calling | Env vars |
|-------|-------|-------------|----------|
| `ollama_qwen2/` | Qwen 2.5 (local Ollama) | Native | `OPENAI_API_BASE`, `OPENAI_MODEL` |
| `deepseek_agent/` | DeepSeek R1 Distill 14B | Prompt-based | `DEEPSEEK_API_KEY` |
| `qwen3_agent/` | Qwen3-14B | Native | `QWEN3_API_KEY` |
| `llama_scout_agent/` | Llama Scout 17B | Native | `LLAMA_SCOUT_API_KEY` |

API keys go in `config/.env` (gitignored) — `run_harness.sh` sources it automatically.

```bash
# Run all agents (default K8s fault)
./scripts/run_harness.sh --flag scale_zero --variant cart

# Run a specific agent + scenario
./scripts/run_harness.sh --scenario a --classifier code/agents/qwen3_agent/agent.py --flag scale_zero --variant cart
```

## Harness events and metrics

Events are appended to `data/event_ledger.jsonl` and logged to MLflow:

| Event | When |
|-------|------|
| `fault_injection` | Fault injected (flagd flag set or K8s action applied) |
| `first_alert` | Agent returns `detected: true` |
| `fault_recovery` | Fault cleared (flag off or K8s recovery) |

**MTTD** (Mean Time To Detect):

```
MTTD = first_alert_time - fault_injection_time
```

**MTTR** (Mean Time To Remediate) — as implemented in the harness today:

```
MTTR = remediation_suggested_time - fault_injection_time
```

`remediation_suggested_time` is recorded when the agent returns non-empty `suggested_remediations` (typically on the same poll as detection). This measures time from fault injection until the agent proposes a fix, not time from detection until the cluster is healthy. See [ARCHITECTURE_PROPOSAL.md](docs/ARCHITECTURE_PROPOSAL.md) for alternative MTTR definitions (detection → recovery).

## Docs

- [ARCHITECTURE_PROPOSAL.md](docs/ARCHITECTURE_PROPOSAL.md) — Full design
- [LLM_CREDENTIALS.md](docs/LLM_CREDENTIALS.md) — API key setup and MLflow logging
- [OPENSHIFT_OTEL_DEPLOYMENT.md](docs/OPENSHIFT_OTEL_DEPLOYMENT.md) — OpenShift setup
