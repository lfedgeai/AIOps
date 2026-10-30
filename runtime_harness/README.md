# AIOps Runtime Harness

Self-adopting, closed-loop AIOps runtime: **observes any OTLP-emitting system, detects anomalies (Granite), correlates signals and root-causes (GLM-5.3-Flash), and remediates behind safety gates (GLM-5.3-Flash)** — with its **own ClickHouse telemetry instance**, a live scheduler, and honest MTTD/MTTR metrics.

Evidence base: [EVALUATION_RESULTS_Oct26.md](../research/agentic_aiops_architectures/docs/EVALUATION_RESULTS_Oct26.md) (Run A 288 cells + Run B 40 cells, fixed metric pipeline).

## Architecture

```
otel-demo app ──OTLP──► otel-collector ──┬──► eval ClickHouse (agentic-aiops)
                                          └──► runtime ClickHouse (clickhouse-rt) ← OWN instance
                                                       │
   scheduler (UI-settable interval) ──► cycle: enrich → telemetry (self-adopt) → DETECT (Granite)
                                                       → CORRELATE/RCA (GLM-5.3) → REMEDIATE (GLM-5.3, gated) → verify
                                                       │
                                          MTTD/MTTR from T0 (injected | telemetry-evidence) ──► trends UI
```

- **No fabricated evidence**: healthy systems report healthy. Fault labels only appear when a drill fault is declared (`POST /cycle {fault: {...}}`).
- **Honest T0**: drill mode uses `injected_at` (exact); production uses first telemetry evidence (lower bound, labelled). MTTD = detection − T0; MTTR = remediation − T0.
- **The harness owns execution**: models propose; writes are gated (`dry_run`, `human_in_the_loop`, MCP write tools refused unless `AIOPS_RT_ENABLE_WRITE_TOOLS=1`).
- **No agent framework** — see [the design rationale](../blog/2026-10-27-no-framework-required.md). Pure Python: FastAPI, pydantic, httpx, kubernetes client, FastMCP.

## Quick start

```bash
cd runtime_harness
# deps on system python (or your venv):
pip install -e ".[mcp,k8s]"   # fastapi uvicorn httpx pydantic pyyaml rich + mcp + kubernetes

# 1. Model credentials (Granite detect, GLM-5.3 correlate/RCA/remediate):
export GRANITE_API_BASE=... GRANITE_API_KEY=... GRANITE_MODEL=...
export GLM_API_BASE=...     GLM_API_KEY=...     GLM_MODEL=...
# (research tree has these in research/agentic_aiops_architectures/config/.env)

# 2. Reach the runtime's own ClickHouse (port-forward to clickhouse-rt):
oc port-forward -n agentic-aiops svc/clickhouse-rt 38124:8123 &

# 3. Serve the UI + API:
PYTHONPATH=. python3 -m uvicorn runtime_harness.api:app --port 8085
# open http://127.0.0.1:8085/
```

The UI provides: **scheduler timing control** (interval + enable/disable), **live cycle progress** (stage pipeline, incident, MTTD/MTTR), **telemetry status of the own ClickHouse** (table freshness, adopted services, ingestion rate), and **MTTD/MTTR trend charts**.

### API

| Endpoint | Purpose |
|---|---|
| `GET /` | Live UI (status, scheduler, telemetry, trends) |
| `GET /status` | `RuntimeStatus` JSON (stage, health, incident, MTTD/MTTR) |
| `GET /assembly` · `GET\|PUT /models` · `POST /models/reset` | Model lineup + overrides |
| `GET\|PUT /prompts` · `POST /prompts/reset` | Editable prompts |
| `GET\|PUT /scheduler` | Continuous cycle control `{enabled, interval_s}` |
| `GET\|PUT /context` | Active C1–C4 RAG context sets (BM25) |
| `GET\|PUT /triggers` | Trigger config: mode (`scheduler`\|`alertmanager`), watcher toggle |
| `POST /alert` | Alertmanager webhook (push mode) — firing alerts → targeted cycles |
| `POST /trigger` | Generic trigger `{target, t0, source}` |
| `GET /telemetry-status` | Own-ClickHouse health (freshness, services, rate) |
| `GET /trends` | Cycle history (MTTD/MTTR series) |
| `POST /cycle` | One cycle; optional `{target, fault: {type, target, injected_at}}` for drills |
| `POST /cycle` (fault types) | `scale_zero` · `kill_pod` · `config_corruption` |

### MCP tooling (eval-harness parity)

```bash
aiops-runtime-mcp --transport stdio     # or: aiops-runtime mcp
```

Reads: `list_services`, `search_logs`, `search_traces`, `search_metrics`, `get_deployment_config`.
Writes (gated by `AIOPS_RT_ENABLE_WRITE_TOOLS=1`): `scale_deployment`, `restart_deployment`, `delete_pod`, `patch_deployment` — with the Oct-26 guardrails: `set_env` may only override existing vars, no-op patches fail, 409s retry with backoff.

### One cycle (CLI)

```bash
PYTHONPATH=. python3 -m runtime_harness.cli cycle --target cart
```

### Deploying the own ClickHouse + telemetry fork

```bash
oc apply -f clickhouse-rt.yaml          # PVC + Deployment + Service in agentic-aiops
# patch otel-demo's otel-collector: add a `clickhouse/rt` exporter
# (see /tmp/opencode/relay-new.yaml pattern — exporter clickhouse/rt + pipeline refs)
oc rollout restart deploy/otel-collector -n otel-demo
oc port-forward -n agentic-aiops svc/clickhouse-rt 38124:8123 &
```

### Querying the telemetry in ClickHouse

The runtime's own instance is reachable at `http://127.0.0.1:38124` (port-forward to `clickhouse-rt`), database `otel`. ClickHouse's HTTP interface takes SQL as the POST body:

```bash
# recent logs (newest first)
curl -s -X POST http://127.0.0.1:38124 --data-binary \
  "SELECT Timestamp, ServiceName, substring(Body, 1, 120) FROM otel.otel_logs
   ORDER BY Timestamp DESC LIMIT 5 FORMAT Vertical"

# which services are erroring right now (the detector's first tool call)
curl -s -X POST http://127.0.0.1:38124 --data-binary \
  "SELECT ServiceName, count() AS errors FROM otel.otel_logs
   WHERE Timestamp >= now() - INTERVAL 30 MINUTE
     AND (positionCaseInsensitive(SeverityText, 'ERROR') > 0
          OR positionCaseInsensitive(Body, 'error') > 0)
     AND ServiceName != ''
   GROUP BY ServiceName ORDER BY errors DESC LIMIT 10 FORMAT TabSeparated"

# failed spans for a service, since a timestamp (RCA correlation input)
curl -s -X POST http://127.0.0.1:38124 --data-binary \
  "SELECT Timestamp, SpanName, Duration/1000000 AS ms FROM otel.otel_traces
   WHERE ServiceName = 'checkout' AND StatusCode = 'Error'
     AND Timestamp >= '2026-10-29 00:00:00'
   ORDER BY Timestamp DESC LIMIT 10 FORMAT TabSeparated"

# service inventory (self-adoption view, 2h window)
curl -s -X POST http://127.0.0.1:38124 --data-binary \
  "SELECT ServiceName, count() FROM otel.otel_logs
   WHERE Timestamp >= now() - INTERVAL 2 HOUR AND ServiceName != ''
   GROUP BY ServiceName ORDER BY 2 DESC FORMAT TabSeparated"

# a service's recent metric values
curl -s -X POST http://127.0.0.1:38124 --data-binary \
  "SELECT TimeUnix, MetricName, ServiceName, Value FROM otel.otel_metrics_gauge
   WHERE ServiceName = 'cart' ORDER BY TimeUnix DESC LIMIT 10 FORMAT TabSeparated"
```

With the `clickhouse-client` binary instead of curl (if installed):

```bash
clickhouse-client --host 127.0.0.1 --port 38124 --query \
  "SELECT count() FROM otel.otel_logs"
```

Notes: timestamps in `WHERE` clauses use `YYYY-MM-DD HH:MM:SS` (UTC); `TabSeparated` / `Vertical` / `JSONEachRow` are convenient output formats. Tables: `otel_logs`, `otel_traces`, `otel_metrics_gauge`, `otel_metrics_sum` (plus histogram/summary variants).

## Layout

```
runtime_harness/
  config/default_assembly.yaml   # Granite/Glm lineup, own-CH URL, scheduler defaults
  runtime_harness/
    schemas.py                   # DetectDecision / RcaDecision / RemediationPlan (response contract)
    agent.py                     # bounded tool-loop: full telemetry query access for detect/RCA
    rag.py                       # BM25 C1–C4 corpora (source/docs/architecture/deps)
    faults.py                    # fault registry + honest T0 resolution
    providers/llm.py             # MaaS-only, structured output, fail-loud (no mock)
    orchestrator.py              # staged loop, T0-based MTTD/MTTR
    scheduler.py                 # continuous cycles (UI-settable interval, no overlap)
    history.py                   # persisted cycle history (trends)
    telemetry/clickhouse.py      # own-CH client + telemetry_status()
    mcp/server.py                # eval-parity tool surface (reads + gated writes)
    api.py                       # FastAPI + UI
    cli.py
    slots/                       # pluggable stage interfaces
```

## Remediation — three mutually exclusive modes

One active mode at a time (`GET|PUT /remediation`, UI buttons, YAML `remediation.mode`):

| Mode | What the model does | What executes | Safety basis |
|---|---|---|---|
| **hil** (default) | Proposes an action string | Nothing — a human decides | Human review |
| **governed** | **Selects** a playbook from the reviewed catalog (`playbooks/`) + params | The harness runs it via ansible-runner | Organizational trust in reviewed, versioned playbooks — the model never writes commands; catalog is enforced harness-side |
| **direct** | **Selects** a guarded write tool (`scale/restart/delete/patch`) + args | The harness executes via `write_tools.py` (G1–G3 guardrails) | Structured selection + allowlist + guardrails; **requires `AIOPS_RT_ENABLE_WRITE_TOOLS=1` to arm** |

Fault → playbook defaults: `scale_zero → scale_restore`, `kill_pod → restart_deployment`, `config_corruption → patch_remove_env`. A selection outside the catalog is refused and recorded as a proposal.

## Status / known gaps

- **Tool-loop agents**: detect and RCA investigate with full telemetry query
  access (logs/metrics/traces + `since_ts` + `count_errors_by_service` +
  `get_deployment_config`), bounded at 8 rounds (eval median was 18 — raise
  via `max_tool_iterations` if needed).
- **Real verifier**: recovery is checked against live cluster state
  (`readyReplicas` via the kubernetes client); proposal-mode fallback only
  when cluster access is unavailable.
- **Execution stays harness-gated by design**: models *select*, never write
  commands. Three mutually exclusive modes (HIL / Governed-Ansible / Direct-
  MCP); direct additionally requires `AIOPS_RT_ENABLE_WRITE_TOOLS=1`.
- **Detection shares its suspect**: `DetectDecision.component` names the
  anomalous service; RCA starts from that suspicion.
