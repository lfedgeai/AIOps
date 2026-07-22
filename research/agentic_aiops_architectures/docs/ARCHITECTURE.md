# Agentic AIOps Evaluation Architecture

Framework for comparing LLM agents on autonomous fault detection and remediation against the OpenTelemetry Demo on OpenShift. Measures MTTD, MTTR, RCA accuracy, and remediation success across models, scenarios, fault types, and context corpora.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         EXPERIMENT HARNESS                                       │
│  run_harness.py — fault inject → poll agent → score → MLflow + event ledger      │
└─────────────────────────────────────────────────────────────────────────────────┘
          │                    │                         │
          ▼                    ▼                         ▼
┌──────────────────┐  ┌────────────────────┐  ┌──────────────────────────┐
│ Fault injection  │  │ Agents under test  │  │ Evaluation               │
│ K8s API (default)│  │ (single Python     │  │ MTTD/MTTR, RCA,          │
│ flagd (optional) │  │  tool-calling loop)│  │ remediation judge        │
└──────────────────┘  └────────────────────┘  └──────────────────────────┘
          │                    │
          ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  OTEL Demo (`otel-demo`)          │  ClickHouse (`agentic-aiops`)                 │
│  Microservices + load generator   │  logs / traces / metrics via OTLP collector  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Design choices in production today:**

- No LangChain / AutoGen — each agent is a single Python file with a native tool-calling loop
- Telemetry store is **ClickHouse** (not OpenSearch/Prometheus for agent queries)
- Default faults are **Kubernetes API** mutations; flagd remains optional (`--use-flagd`)
- Experiment tracking is **local MLflow** (`agentic_aiops_mttd_mttr`)

---

## 2. Experiment Flow

One harness run (`./scripts/run_harness.sh` → `code/harness/run_harness.py`):

1. **Resolve config** — scenario (`0|a|b|c`), agent, fault flag/variant, optional `CONTEXT_C` corpora
2. **Start MLflow run** — tags and params for approach, scenario, fault, context
3. **Inject fault** — K8s API (default) or flagd; append `fault_injection` to `data/event_ledger.jsonl`
4. **Wait briefly** — allow OTLP → ClickHouse to catch up (~15s live)
5. **Poll agent** — invoke classifier until `detected` or detection timeout
6. **Record first alert** — MTTD from fault injection → detection
7. **Parse remediation signals** — successful write tools → MTTR
8. **Verify recovery** — poll cluster state on the correct target (`fault_verification.py`)
9. **Auto-recover** — harness clears the injected fault after `fault_duration`
10. **Score** — RCA / remediation against `config/fault_ground_truth.yaml`; optional oracle LLM judge
11. **Log artifacts** — agent output, tool calls, prompts, judge JSON, metrics to MLflow and `out/`

Campaign scripts:

| Script | Purpose |
|--------|---------|
| `scripts/run_harness.sh` | Single run |
| `scripts/run_full_matrix.sh` | Agents × faults on scenario `a` |
| `scripts/run_context_matrix.sh` / `run_c_matrix.sh` / `run_full_c_matrix.sh` | Context corpora (C0–C4) matrices |
| `scripts/run_scenario_campaign.sh` | Scenario sweep |

---

## 3. Telemetry Platform

```
OTEL Demo (otel-demo ns)
        │ OTLP
        ▼
OTEL Collector ──patch──► ClickHouse (agentic-aiops ns)
                              │
                              ▼
                    agent tools: search_logs / search_traces /
                                 search_metrics / query_clickhouse /
                                 compare_telemetry
```

- ClickHouse manifests: `manifests/clickhouse.yaml`
- Collector patch: `scripts/patch-otel-collector-clickhouse.py`
- Query reference: `docs/CLICKHOUSE_QUERIES.md`
- Local access typically via `oc port-forward -n agentic-aiops svc/clickhouse 38123:8123`

---

## 4. Fault Injection

### 4.1 Kubernetes faults (default)

Injected by `inject_k8s_fault()` in `run_harness.py` against deployments in `otel-demo` (override with `K8S_FAULT_NAMESPACE`).

| Flag | Effect | Typical recovery signal |
|------|--------|-------------------------|
| `scale_zero` | Scale deployment to 0 | Scale back ≥1 |
| `kill_pod` | Delete target pods | Self-heal / restart |
| `memory_limit` | Tiny memory limit (OOM) | Remove limit |
| `network_partition` | Blocking NetworkPolicy | Delete policy |
| `readiness_probe_fail` | Always-fail readiness | Remove probe |
| `config_corruption` | Bad env (e.g. `DATABASE_HOST`) | Remove env |
| `dependency_removal` | Scale backing service to 0 | Scale back |
| `replica_overload` | Target 0 + load-generator up | Restore both |
| `node_taint` | NoExecute taint | Remove taint |
| `pvc_full` | Fill pod `/tmp` | Remove fill file |

Usage: `--flag <fault> --variant <deployment>` (e.g. `--flag scale_zero --variant cart`).

Ground truth for RCA/remediation scoring: `config/fault_ground_truth.yaml`.

### 4.2 flagd (optional)

Legacy application-level chaos via Open Feature / flagd. Enable with `--use-flagd`. On OpenShift, ConfigMap patch is preferred when flagd-ui write returns 404 (`config/harness.yaml` → `flagd.use_openshift_cm`).

---

## 5. Scenarios and Tool Profiles

Defined in `config/scenarios.yaml`. Same fault, different topology / tool access.

| Scenario | Mode | Tools | Question |
|----------|------|-------|----------|
| **0** | Single | Telemetry only | Can observability alone detect? |
| **a** | Single | Telemetry + K8s read/write | Full detect → diagnose → fix |
| **b** | Multi | logs / traces / metrics specialists | Does signal specialization help? |
| **c** | Multi | hardware / platform / application | Does domain specialization help? |

Tool allow-lists live in `code/tools/tool_profiles.py`. Profiles gate OpenAI tool definitions so agents cannot call tools outside their role.

Shared tools (`code/tools/agent_tools.py`):

- **Telemetry:** `search_logs`, `search_traces`, `search_metrics`, `query_clickhouse`, `compare_telemetry`
- **K8s read:** `get_pod_status`, `get_pod_logs`, `get_events`
- **K8s write:** `restart_deployment`, `scale_deployment`, `delete_pod` (+ related patches where used)
- **Audit:** `log_action` (also feeds MLflow during the run)

Multi-agent scenarios use `code/agents/scenario_orchestrator/`.

---

## 6. Agents

Each agent under `code/agents/<name>/agent.py` implements the same contract: tool-calling loop → JSON with detection / RCA / remediation signals. Credentials in `config/.env` (see `docs/LLM_CREDENTIALS.md`).

| Agent | Model | Tool calling |
|-------|-------|--------------|
| `nemotron_agent` | NVIDIA Nemotron-3-Nano | Native + reasoning |
| `qwen3_agent` | Qwen3-14B | Native + streaming |
| `deepseek_agent` | DeepSeek R1 Distill 14B | Prompt-based |
| `llama_scout_agent` | Llama Scout 17B | Native |
| `gpt_oss_agent` | GPT-OSS 120B | Native |

Shared helpers: `ai_metrics.py` (TTFT, tokens/sec), `mlflow_agent_logging.py`, `remediation_signals.py`.

---

## 7. Context Engineering (orthogonal axis)

Independent of scenario topology. Controlled by `CONTEXT_C` / `config/context_c_levels.yaml`.

| Corpus | Tool | Content |
|--------|------|---------|
| **C0** | (none) | Live telemetry/tools only |
| **C1** | `rag_search_source` | App source (`src/**`) |
| **C2** | `rag_search_docs` | Docs / README / changelog |
| **C3** | `rag_search_architecture` | Architecture / mermaid material |
| **C4** | `rag_search_dependencies` | Compose + K8s manifests |

Default RAG repo: OpenTelemetry Demo (`open-telemetry/opentelemetry-demo`). Implementation: `code/tools/context_engineering.py`, `rag_context.py`, `rag_policy.py`. Eval write-up: `docs/CONTEXT_ENGINEERING_EVAL_REPORT.md`.

---

## 8. Metrics and Evaluation

### 8.1 Timing

| Metric | Definition |
|--------|------------|
| **MTTD** | `fault_injection → detection` (agent declares fault / effective detection via remediation) |
| **MTTR** | `fault_injection → first successful write tool` |
| **Remediation time** | `detection → fix` (MTTR − MTTD) |
| **MTTR verified** | `fault_injection → recovery verified on correct target` |
| **MTTR = None** | Detected but no successful write |

Events are appended to `data/event_ledger.jsonl` (`fault_injection`, `first_alert`, recovery-related events).

### 8.2 Quality scores

Implemented in `code/harness/evaluation.py`:

- **RCA accuracy** — predicted root cause vs ground truth (normalized component match)
- **Remediation declared** — text mentions expected component
- **Remediation executed** — successful write tool matches fault type / target
- **Hybrid score** — combines declared + executed signals; scenario role scoring for multi-agent

### 8.3 Oracle judge

`code/harness/remediation_judge.py` — optional second LLM (prefer different provider than agent under test) scores RCA/remediation against ground truth. Enabled in `config/harness.yaml` (`judge.enabled`).

### 8.4 AI performance

Per run: TTFT, tokens/sec, total tokens, tool-call count, LLM rounds (logged as MLflow metrics/artifacts).

---

## 9. Experiment Tracking (MLflow)

| Concept | Mapping |
|---------|---------|
| Experiment | `agentic_aiops_mttd_mttr` |
| Run | One (agent × scenario × fault × variant × context) |
| Params / tags | approach, scenario, tool_profiles, fault_*, context_c, injection mode |
| Metrics | `mttd_seconds`, `mttr_seconds`, `mttr_verified_seconds`, `rca_correct`, remediation flags, AI metrics |
| Artifacts | agent output, prompts, tool calls, thinking, judge JSON, harness summary |

Default tracking URI: `http://localhost:5050` (local SQLite `mlflow_local.db` + `mlartifacts/`).

---

## 10. Directory Layout

```
research/agentic_aiops_architectures/
├── code/
│   ├── harness/          # run_harness, evaluation, fault_verification, remediation_judge, scenarios
│   ├── agents/           # One directory per model + scenario_orchestrator
│   └── tools/            # agent_tools, tool_profiles, context_engineering, RAG
├── config/               # harness.yaml, scenarios.yaml, fault_ground_truth.yaml, context_c_levels.yaml, .env
├── scripts/              # run_harness, matrix runners, OpenShift/ClickHouse helpers
├── manifests/            # ClickHouse and related OpenShift YAML
├── data/                 # event_ledger.jsonl
├── out/                  # Per-run JSON summaries (gitignored)
├── mlartifacts/          # MLflow artifacts
├── evalhub/              # TrustyAI / EvalHub provider wiring
└── docs/                 # Architecture, credentials, ClickHouse queries, eval reports
```

---

## 11. Deployment Topology

| Namespace | Role |
|-----------|------|
| `otel-demo` | OTEL Demo microservices, load generator, flagd |
| `agentic-aiops` | ClickHouse observability backend |

OpenShift notes: `docs/OPENSHIFT_OTEL_DEPLOYMENT.md`, `docs/OPENSHIFT_CREDENTIALS.md`.

---

## 12. Related Documents

- [README.md](../README.md) — quick start and operational reference
- [CONTEXT_ENGINEERING_EVAL_REPORT.md](CONTEXT_ENGINEERING_EVAL_REPORT.md) — C0–C4 matrix results
- [EVALUATION_RESULTS_07July26.md](EVALUATION_RESULTS_07July26.md) / [EVALUATION_RESULTS_23June26.md](EVALUATION_RESULTS_23June26.md) — campaign results
- [CLICKHOUSE_QUERIES.md](CLICKHOUSE_QUERIES.md) — SQL against `otel.*` tables
- [LLM_CREDENTIALS.md](LLM_CREDENTIALS.md) — API keys and endpoints
