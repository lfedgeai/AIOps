# Standalone Agentic AIOps Evaluation Harness (OCP / RHOAI)

This repo is a self-contained extraction of the LF Edge AIOps research harness. It measures how well LLM agents **detect, diagnose, and remediate** faults in the OpenTelemetry Demo, reporting **MTTD**, **MTTR**, RCA accuracy, and remediation success.

It runs on **OpenShift Container Platform** and optionally integrates with **Red Hat OpenShift AI 3.4+** (workbenches, KServe/vLLM or MaaS model endpoints, MLflow, and the TrustyAI **EvalHub** Evaluations UI).

Everything needed to deploy the evaluation stack is in this repo: the OTEL Demo Kubernetes manifest, ClickHouse, MLflow, harness RBAC, an optional vLLM endpoint, and a real EvalHub provider.

---

## What it measures

One **cell** = one fault injected into the OTEL Demo + one agent configuration + one context corpus.

| Metric | Meaning |
|---|---|
| `detected` | Agent reported a fault inside the detection window |
| `mttd_seconds` | Fault injection → agent's first detection claim |
| `mttr_seconds` | Fault injection → first effective remediation write |
| `mttr_verified_seconds` | Fault injection → target workload verified healthy |
| `rca_correct` | Predicted root cause matches ground truth |
| `remediation_executed` | A targeted, action-appropriate write succeeded |
| `recovery_verified` | Harness verified recovery before its own auto-recovery |

Each run also records TTFT, tokens/sec, tool calls, and LLM rounds in MLflow.

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │  Harness (Python CLI / Job / Workbench)       │
                    │  fault inject → poll agent → score → MLflow   │
                    └───────┬───────────────────┬──────────────────┘
                            │                   │
              Kubernetes API│                   │OpenAI-compatible
                            ▼                   ▼
        ┌───────────────────────────┐   ┌───────────────────────────┐
        │ OTEL Demo (otel-demo)     │   │ Models (MaaS / KServe /   │
        │ services + load generator │   │ raw vLLM)                 │
        └───────────┬───────────────┘   └───────────────────────────┘
                    │ OTLP
                    ▼
        ┌───────────────────────────┐   ┌───────────────────────────┐
        │ OTEL Collector            │──▶│ ClickHouse                │
        │ (patched exporter)        │   │ (agentic-aiops)           │
        └───────────────────────────┘   └───────────┬───────────────┘
                                                    │ SQL over HTTP
                                                    ▼
                                        ┌───────────────────────────┐
                                        │ Harness tools             │
                                        │ logs / traces / metrics   │
                                        └───────────────────────────┘
```

Namespaces:
- `otel-demo` — OTEL Demo, load generator, collector, flagd.
- `agentic-aiops` — ClickHouse, MLflow, harness Job/SA, optional vLLM.
- RHOAI namespaces (configurable in `.env`; defaults `redhat-ods-operator`, `redhat-ods-applications`, `rhods-notebooks`) — optional EvalHub.

---

## What is included

- `code/` — harness, agents, tool profiles, RAG context engineering.
- `config/` — scenarios, fault ground truth, context levels, harness defaults.
- `manifests/otel-demo.yaml` — vendored upstream OTEL Demo Kubernetes manifest (no Helm dependency).
- `manifests/clickhouse.yaml` — ClickHouse + PVC + memory cap.
- `manifests/mlflow.yaml` — standalone MLflow + PVC + Route.
- `manifests/vllm.yaml` — optional raw vLLM OpenAI-compatible endpoint.
- `manifests/rbac/` — namespace-scoped fault-injection RBAC; optional node RBAC.
- `manifests/harness-job.yaml` — one-shot in-cluster harness Job.
- `context/opentelemetry-demo/` — vendored source used by RAG corpora C1–C4.
- `evalhub/` — real EvalHub provider that runs one harness cell per job.
- `scripts/` — deploy, smoke test, matrix runner, EvalHub wiring.

The original research notes and dated evaluation reports are intentionally not copied. The code and scoring logic are preserved.

---

## Prerequisites

- OpenShift 4.x cluster, `oc` (or `kubectl`), cluster-admin or enough namespace/SCC rights.
- Python 3.11+ for local/Workbench runs:
  `python3 -m pip install -r requirements.txt -r code/harness/requirements.txt`.
- For the EvalHub UI: RHOAI **3.4+** with `dashboard` and `trustyai` components **Managed**. The cluster global pull secret must include credentials for `registry.redhat.io`; run `./scripts/check-pull-secret.sh`.
- For in-cluster models: RHOAI KServe/vLLM, an external MaaS/LiteLLM gateway, or the included raw vLLM fallback with GPU nodes.
- An AWS OCP cluster with a default StorageClass (usually `gp3-csi`) and enough worker capacity for the OTEL Demo (~6 GB RAM) plus ClickHouse/MLflow. Cluster inputs are captured in [docs/AWS-OCP.md](docs/AWS-OCP.md).

---

## Quick start

### 1. Fill in `.env`

```bash
cp .env.example .env   # .env is gitignored
$EDITOR .env
```

At minimum set:
- cluster access (`KUBECONFIG`, or `OCP_API_URL` + `OCP_TOKEN`)
- `STORAGE_CLASS`
- the model endpoints/keys for the agents you want to run
- `HARNESS_IMAGE` once you build/push it

Use `./scripts/00-preflight.sh` to validate cluster access and storage. If you are installing/using RHOAI, also run `./scripts/check-pull-secret.sh`; the cluster pull secret must have `registry.redhat.io` credentials.

### 2. Build and push the harness image

```bash
# set HARNESS_IMAGE=quay.io/<you>/aiops-harness:latest in .env first
./scripts/build-image.sh
# or: CONTAINER_ENGINE=docker ./scripts/build-image.sh
# or: PUSH=0 ./scripts/build-image.sh   # build only
```

The image contains the harness, all agents, the vendored context repo, and the Python dependencies. It runs with the pod service account — no `oc` or kubeconfig is needed in-cluster.

### 3. Deploy the stack

```bash
./scripts/deploy.sh
```

This runs:
1. preflight (cluster, storage class, optional RHOAI/GPU checks)
2. namespaces + harness RBAC
3. vendored OTEL Demo + SCC bindings + Routes
4. ClickHouse + PVC
5. collector patch → ClickHouse
6. MLflow + PVC + Route

Optional:

```bash
VLLM_ENABLED=1 ./scripts/deploy.sh          # raw vLLM fallback
ENABLE_EVALHUB=1 ./scripts/deploy.sh        # RHOAI EvalHub wiring
```

### 4. Create model credentials

```bash
./scripts/create-secrets.sh
```

This creates `agentic-aiops/aiops-model-credentials` from the `*_API_KEY`/`*_API_BASE` values in `.env`; the harness Job consumes it with `envFrom`.

### 5. Smoke test

```bash
./scripts/smoke_test.sh
# or:
./scripts/run_harness.sh --scenario a --flag scale_zero --variant cart \
  --agents rhdp_qwen3-14b --detection-timeout 600 --fault-duration 120
```

### 6. Run a matrix

```bash
./scripts/run_matrix.sh context        # scenario A × agents × 3 faults × C1–C4
./scripts/run_matrix.sh full           # agents × 8 faults × scenarios 0/a/b/c
./scripts/run_matrix.sh permutations   # all CONTEXT_C subsets
```

Results land in `out/`, events in `data/event_ledger.jsonl`, and metrics/artifacts in MLflow.

---

## Running in-cluster (recommended)

Apply the Job with the helper (it renders the manifest and recreates the Job):

```bash
HARNESS_ARGS="--scenario a --flag scale_zero --variant cart --agents rhdp_qwen3-14b" \
  ./scripts/run_job.sh
```

The Job:
- runs as ServiceAccount `aiops-harness`
- mounts a PVC for `out/`, `data/`, `mlartifacts/`, `logs/`
- reads model credentials from the `aiops-model-credentials` Secret
- talks to ClickHouse and MLflow over cluster DNS

For long matrices, prefer a Job or RHOAI Pipeline over a Workbench: RHOAI Workbench idle-culling will kill a 10–14 hour run.

---

## RHOAI / EvalHub

The `evalhub/lfedge_aiops_provider` is a **real** provider: it runs one harness cell, parses the harness JSON, and reports `mttd_seconds`, `mttr_seconds`, `detected`, `rca_correct`, `remediation_executed`, `recovery_verified`, and `agent_error`.

Registration:

```bash
ENABLE_EVALHUB=1 ./scripts/deploy.sh
# or:
./scripts/70-evalhub.sh
```

The provider needs the same RBAC as the harness in `otel-demo`. The EvalHub job service account must be bound to the `aiops-harness-faults` Role (see the message printed by `70-evalhub.sh`). Set `LFEDGE_MODEL_API_KEY` in `.env` to the key for the model under test, or pass `api_key` in the benchmark parameters.

If RHOAI is installed with non-default namespaces, set `RHOAI_OPERATOR_NAMESPACE`, `RHOAI_APPLICATIONS_NAMESPACE`, `EVALHUB_NAMESPACE`, and `EVALHUB_PLATFORM_NAMESPACE` in `.env`. The scripts no longer assume `redhat-ods-applications` or `rhods-notebooks`.

Known RHOAI 3.4 caveats (from the original study):
- The dashboard federated Evaluations UI cannot be durably patched; use the EvalHub Route with `Authorization` + `X-Tenant` (the scripts do this).
- The tenant EvalHub CR requires `spec.database.type=sqlite`.
- A platform EvalHub in `EVALHUB_PLATFORM_NAMESPACE` (defaults to `RHOAI_APPLICATIONS_NAMESPACE`) is needed for the UI health check.
- EvalHub job scheduling may cross-wire tenant RoleBindings; `fix-evalhub-tenant-rbac.sh` adds stable bindings.

---

## Model endpoints

The harness speaks **OpenAI-compatible** `/v1/chat/completions` and requires tool calling. Two supported patterns:

1. **MaaS / LiteLLM gateway** (recommended) — set each agent's `*_API_BASE`, `*_API_KEY`, `*_MODEL`.
2. **RHOAI KServe/vLLM or raw vLLM** — serve with `--enable-auto-tool-choice` and a compatible `--tool-call-parser` (`hermes` for Qwen). Set the agent's `*_API_BASE` to `https://<route>/v1`.

Default lineup (6 agents, 5 distinct models):
- DeepSeek R1 Distill 14B
- Qwen3-14B
- Qwen3.8-27B (RHDP)
- Qwen3.8-27B (LiteMaaS; the code's `qwen36` slot)
- Granite 3.2 8B Instruct
- GLM-5.3 Flash

The `vllm.yaml` fallback is sized for a single small model (default Qwen2.5-7B). The full lineup needs MaaS or multiple/adequate GPUs.

---

## Faults

All faults are injected through the Kubernetes API in `K8S_FAULT_NAMESPACE` (default `otel-demo`):

| Flag | Effect |
|---|---|
| `scale_zero` | Scale deployment to 0 |
| `kill_pod` | Delete target pods |
| `memory_limit` | Set a 10Mi memory limit (OOMKill) |
| `network_partition` | Block ingress with a NetworkPolicy |
| `readiness_probe_fail` | Set an always-failing readiness probe |
| `config_corruption` | Inject an invalid `DATABASE_HOST` |
| `dependency_removal` | Scale a backing service to 0 |
| `replica_overload` | Scale target to 0 + load generator to 3 |
| `node_taint` | Taint a node with NoExecute (**cluster-scoped; dangerous**) |
| `pvc_full` | Fill 500 MB in a pod's `/tmp` |

`--variant` is the target deployment, e.g. `cart`.

---

## Scenarios and context

| Scenario | Mode | Question |
|---|---|---|
| `0` | Single agent, telemetry only | Can observability alone detect? |
| `a` | Single agent, telemetry + K8s | Full detect → diagnose → fix |
| `b` | Multi-agent (logs/traces/metrics) | Does signal specialization help? |
| `c` | Multi-agent (hardware/platform/application) | Does domain specialization help? |

Context corpora (orthogonal): `C0` none, `C1` source, `C2` docs, `C3` architecture, `C4` dependencies. The vendored `context/opentelemetry-demo` makes C1–C4 work without GitHub egress.

---

## Security notes

- The harness has **write access** to `otel-demo` (scale/restart/delete/patch, pod exec, NetworkPolicies). Never bind it to RHOAI/application namespaces.
- `node_taint` is cluster-scoped and can evict model servers/workbenches. Leave `ENABLE_NODE_FAULTS=0` on shared clusters.
- The ClickHouse manifest has no authentication. Keep it cluster-internal; do not expose a Route.
- Model keys live in `.env`/the `aiops-model-credentials` Secret. Do not bake `.env` into the image.
- The EvalHub provider uses an in-pod unverified TLS context for its sidecar callback; that is local to the job pod.

---

## Troubleshooting

- **ClickHouse unreachable** — `oc get pods -n agentic-aiops -l app=clickhouse`; check the PVC/StorageClass; from a laptop set `CLICKHOUSE_PORT_FORWARD=1`. From a laptop also set `MLFLOW_MODE=local` (or point `MLFLOW_TRACKING_URI` at the MLflow Route).
- **Collector not exporting** — `oc logs -n otel-demo deploy/otel-collector`; re-run `./scripts/40-collector.sh`; verify `SHOW TABLES FROM otel` in ClickHouse.
- **OTEL demo pods Pending/CrashLoop** — check privileged SCC bindings (`scripts/20-otel-demo.sh`) and memory; the full demo needs ~6 GB.
- **Agent API preflight fails** — check the `*_API_BASE`/`*_API_KEY`; set `HARNESS_ALLOW_MISSING_AGENTS=1` to run only configured agents.
- **No remediation writes** — confirm the harness SA can patch deployments in `otel-demo`; run `pre_run_health_gate.py`.
- **MLflow unreachable** — `oc get route mlflow -n agentic-aiops`; the in-cluster URI is `http://mlflow.agentic-aiops.svc.cluster.local:5000`.

---

## Repo layout

```
code/                harness, agents, tools, RAG
config/              scenarios, fault ground truth, context levels
context/             vendored OTEL demo source for RAG
docs/                architecture and harness design notes
evalhub/             real EvalHub provider + manifests
manifests/           OTEL demo, ClickHouse, MLflow, vLLM, RBAC, Job
scripts/             deploy, smoke, matrix, EvalHub wiring
.env.example         configuration template (copy to .env)
```

---

## Attribution

The harness, agents, scoring, and research design come from the LF Edge AIOps `research/agentic_aiops_architectures` study. This repo repackages them for a standalone OCP/RHOAI deployment: vendored OTEL Demo, consistent namespaces, in-cluster MLflow/ClickHouse, portable Python, RBAC, and a real EvalHub provider.
