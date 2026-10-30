"""FastAPI status UI + control plane — scheduler, telemetry, trends.

Purpose
-------
Operator surface for the runtime harness:
- live cycle status / progress (stage pipeline, incident, MTTD/MTTR)
- scheduler timing control (interval, enable/disable) — continuous cycles
- telemetry status of the runtime's OWN ClickHouse (freshness, services, rate)
- MTTD / MTTR trend charts from persisted cycle history
- model lineup (Granite detect · GLM-5.3 correlate/RCA/remediate)

How it connects
---------------
- Bootstraps ``Assembly`` + ``Orchestrator`` + ``Scheduler`` on startup.
- ``GET|PUT /scheduler`` → live timing control.
- ``GET /telemetry-status`` → ``telemetry/clickhouse.telemetry_status``.
- ``GET /trends`` → ``history.get_history().trends()``.
- ``POST /cycle`` runs one cycle (optional drill fault for T0-exact metrics).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from runtime_harness.assembly import Assembly, load_assembly_config
from runtime_harness.faults import FAULT_TYPES, FaultSpec
from runtime_harness.history import get_history
from runtime_harness.models import RuntimeStatus
from runtime_harness.orchestrator import Orchestrator, bootstrap_status
from runtime_harness.prompts import get_prompt_store
from runtime_harness.scheduler import get_scheduler, init_scheduler
from runtime_harness.stage_models import get_stage_model_store
from runtime_harness.status_store import get_store
from runtime_harness.telemetry.clickhouse import (
    ClickHouseClient,
    clickhouse_http_url_from_config,
    telemetry_status,
)

_assembly: Optional[Assembly] = None
_orch: Optional[Orchestrator] = None


def get_assembly() -> Assembly:
    global _assembly, _orch
    if _assembly is None:
        _assembly = Assembly(load_assembly_config())
        _orch = Orchestrator(_assembly)
        bootstrap_status(_assembly)
    return _assembly


def get_orch() -> Orchestrator:
    get_assembly()
    assert _orch is not None
    return _orch


app = FastAPI(
    title="AIOps Runtime Harness",
    description="Self-adopting closed-loop AIOps runtime (Granite detect · GLM-5.3 RCA/remediate).",
    version="0.2.0",
)


@app.on_event("startup")
def _startup() -> None:
    a = get_assembly()
    sched_cfg = a.config.get("scheduler") or {}
    init_scheduler(
        runner=lambda: get_orch().run_cycle(source="scheduler"),
        enabled=bool(sched_cfg.get("enabled", False)),
        interval_s=int(sched_cfg.get("interval_s", 300)),
        gate=lambda: get_trigger_mode() == "scheduler",
    )
    from runtime_harness.triggers import init_trigger_mode_from_config

    init_trigger_mode_from_config(a.config)
    _init_watcher(a)
    _init_am_poller(a)


def _init_watcher(assembly: Assembly):
    global _watcher
    from runtime_harness.telemetry.clickhouse import ClickHouseClient
    from runtime_harness.triggers import StatisticalWatcher, record_trigger
    from runtime_harness.faults import T0

    w_cfg = (assembly.config.get("triggers") or {}).get("watcher") or {}

    def fire(target: str, onset, reason: str) -> None:
        record_trigger("watcher", target, reason, t0=onset.isoformat())
        get_scheduler().try_run(
            lambda: get_orch().run_cycle(
                target=target, t0=T0(ts=onset, source=reason), source="watcher"
            )
        )

    _watcher = StatisticalWatcher(
        fire=fire,
        client_factory=ClickHouseClient,
        interval_s=int(w_cfg.get("interval_s", 20)),
        error_window_m=int(w_cfg.get("error_window_m", 5)),
        error_threshold=int(w_cfg.get("error_threshold", 25)),
        silence_s=int(w_cfg.get("silence_s", 90)),
        cooldown_s=int(w_cfg.get("cooldown_s", 300)),
    )
    _watcher.enabled = bool(w_cfg.get("enabled", False))
    _watcher.start()


_watcher = None
_am_poller = None


def _init_am_poller(assembly: Assembly):
    global _am_poller
    from runtime_harness.triggers import AlertmanagerPoller, record_trigger

    t_cfg = assembly.config.get("triggers") or {}
    am_cfg = t_cfg.get("alertmanager") or {}

    def fire(target, t0, alertname, severity):
        record_trigger(
            "alertmanager-poll",
            target,
            f"{alertname} ({severity})",
            t0=t0.ts.isoformat() if t0 else None,
        )
        get_scheduler().try_run(
            lambda: get_orch().run_cycle(target=target, t0=t0, source="alertmanager")
        )

    _am_poller = AlertmanagerPoller(
        fire=fire,
        url=str(am_cfg.get("url", "http://127.0.0.1:9093")),
        interval_s=int(am_cfg.get("poll_interval_s", 15)),
    )
    _am_poller.start()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class FaultRequest(BaseModel):
    type: str = Field(description=f"One of {FAULT_TYPES}")
    target: str = Field(description="Workload name, e.g. 'cart'")
    injected_at: Optional[str] = Field(
        default=None, description="ISO timestamp; defaults to now"
    )


class CycleRequest(BaseModel):
    target: Optional[str] = Field(
        default=None, description="Optional focus service; omit to adopt from telemetry"
    )
    namespace: str = "otel-demo"
    context_c: Optional[str] = None
    fault: Optional[FaultRequest] = Field(
        default=None, description="Drill mode: injected fault (exact T0 for MTTD/MTTR)"
    )


class PromptUpdate(BaseModel):
    system_prompt: Optional[str] = None
    user_prompt_template: Optional[str] = None
    detect_system: Optional[str] = None
    rca_system: Optional[str] = None
    remediate_system: Optional[str] = None
    notes: Optional[str] = None


class SchedulerUpdate(BaseModel):
    enabled: Optional[bool] = None
    interval_s: Optional[int] = Field(default=None, ge=30, le=86400)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"ok": "true"}


@app.get("/status", response_model=RuntimeStatus)
def status() -> RuntimeStatus:
    return get_store().get()


@app.get("/assembly")
def assembly() -> dict[str, Any]:
    a = get_assembly()
    return {
        "name": a.name,
        "components": [c.model_dump() for c in a.component_refs()],
        "evidence_report": a.evidence_report(),
        "config": a.config,
    }


@app.get("/models")
def models() -> dict[str, Any]:
    cfg = get_stage_model_store().get().to_dict()
    from runtime_harness.providers.llm import wiring_info

    cfg["wiring"] = wiring_info(
        [
            cfg["anomaly_detection"]["primary"],
            cfg["rca"]["primary"],
            cfg["remediation"]["primary"],
        ]
    )
    return cfg


@app.put("/models")
def models_update(body: dict[str, Any]) -> dict[str, Any]:
    store = get_stage_model_store()
    store.update(
        anomaly_detection=body.get("anomaly_detection"),
        rca=body.get("rca"),
        remediation=body.get("remediation"),
    )
    return store.get().to_dict()


@app.post("/models/reset")
def models_reset() -> dict[str, Any]:
    return get_stage_model_store().reset().to_dict()


@app.get("/prompts")
def prompts() -> dict[str, Any]:
    return get_prompt_store().get().to_dict()


@app.put("/prompts")
def prompts_update(body: PromptUpdate) -> dict[str, Any]:
    store = get_prompt_store()
    store.update(
        system_prompt=body.system_prompt,
        user_prompt_template=body.user_prompt_template,
        detect_system=body.detect_system,
        rca_system=body.rca_system,
        remediate_system=body.remediate_system,
        notes=body.notes,
    )
    return store.get().to_dict()


@app.post("/prompts/reset")
def prompts_reset() -> dict[str, Any]:
    return get_prompt_store().reset().to_dict()


@app.get("/scheduler")
def scheduler_status() -> dict[str, Any]:
    return get_scheduler().status()


@app.put("/scheduler")
def scheduler_update(body: SchedulerUpdate) -> dict[str, Any]:
    get_scheduler().configure(enabled=body.enabled, interval_s=body.interval_s)
    return get_scheduler().status()


@app.get("/trends")
def trends(limit: int = 100) -> dict[str, Any]:
    return {"cycles": get_history().trends(limit=limit)}


@app.get("/context")
def context_sets() -> dict[str, Any]:
    from runtime_harness.rag import CONTEXT_GLOBS, get_active_sets

    return {
        "active": get_active_sets(),
        "available": {
            k: {"label": v["label"], "globs": v["include"]}
            for k, v in CONTEXT_GLOBS.items()
        },
    }


@app.put("/context")
def context_sets_update(body: dict[str, Any]) -> dict[str, Any]:
    from runtime_harness.rag import get_active_sets, set_active_sets

    sets = body.get("sets")
    if not isinstance(sets, list):
        raise ValueError("body must be {\"sets\": [\"1\",\"2\",...]}")
    set_active_sets(sets)
    return {"active": get_active_sets()}


@app.get("/remediation")
def remediation_status() -> dict[str, Any]:
    from runtime_harness.remediation import mode_status

    return mode_status()


@app.get("/triggers")
def triggers_status() -> dict[str, Any]:
    from runtime_harness.triggers import (
        TRIGGER_MODES,
        get_trigger_mode,
        get_trigger_namespaces,
        trigger_history,
    )

    return {
        "mode": get_trigger_mode(),
        "modes": list(TRIGGER_MODES),
        "namespaces": get_trigger_namespaces(get_assembly().config),
        "scheduler": get_scheduler().status(),
        "watcher": _watcher.status() if _watcher else {},
        "alertmanager": _am_poller.status() if _am_poller else {},
        "history": trigger_history(10),
    }


@app.put("/triggers")
def triggers_update(body: dict[str, Any]) -> dict[str, Any]:
    from runtime_harness.triggers import set_trigger_mode

    mode = body.get("mode")
    if mode is not None:
        set_trigger_mode(mode)
        if _am_poller:
            _am_poller.enabled = mode == "alertmanager"
    if "watcher_enabled" in body:
        if _watcher:
            _watcher.enabled = bool(body["watcher_enabled"])
    return triggers_status()


@app.post("/alert")
def alertmanager_webhook(payload: dict[str, Any]) -> dict[str, Any]:
    """Alertmanager webhook receiver: firing alerts → targeted cycles.

    Trust-but-verify: the alert is a cue (target hint + T0 from startsAt);
    the detector still investigates independently.
    """
    import threading

    from runtime_harness.faults import T0
    from runtime_harness.triggers import parse_alertmanager_payload, record_trigger

    events = parse_alertmanager_payload(payload)
    accepted = []
    for ev in events:
        record_trigger(
            "alertmanager",
            ev.target,
            f"{ev.alertname} ({ev.severity})",
            namespace=ev.namespace,
            t0=ev.starts_at.isoformat() if ev.starts_at else None,
        )
        accepted.append(
            {"target": ev.target, "alertname": ev.alertname, "severity": ev.severity}
        )
        t0 = T0(ts=ev.starts_at, source="alert") if ev.starts_at else None

        def _run(ev=ev, t0=t0):
            get_scheduler().try_run(
                lambda: get_orch().run_cycle(
                    target=ev.target, t0=t0, source="alertmanager"
                )
            )

        threading.Thread(target=_run, daemon=True).start()
    return {"accepted": accepted, "dropped": len((payload.get("alerts") or [])) - len(accepted)}


@app.post("/trigger")
def generic_trigger(body: dict[str, Any]) -> dict[str, Any]:
    """Generic trigger: {target, t0 (ISO), source} → targeted cycle."""
    import threading

    from datetime import datetime as _dt
    from runtime_harness.faults import T0
    from runtime_harness.triggers import record_trigger

    target = body.get("target")
    if not target:
        raise ValueError("target is required")
    source = str(body.get("source", "external"))
    t0 = None
    if body.get("t0"):
        ts = _dt.fromisoformat(str(body["t0"]).replace("Z", "+00:00"))
        t0 = T0(ts=ts, source=source)
    record_trigger(source, target, body.get("detail", ""), t0=body.get("t0"))

    def _run():
        get_scheduler().try_run(
            lambda: get_orch().run_cycle(target=target, t0=t0, source=source)
        )

    threading.Thread(target=_run, daemon=True).start()
    return {"accepted": True, "target": target, "source": source}


@app.put("/remediation")
def remediation_update(body: dict[str, Any]) -> dict[str, Any]:
    from runtime_harness.remediation import RemediationMode, mode_status, set_mode

    mode = body.get("mode")
    if mode not in ("hil", "governed", "direct"):
        raise ValueError("mode must be one of: hil, governed, direct")
    set_mode(RemediationMode(mode))
    return mode_status()


@app.get("/telemetry-status")
def telemetry() -> dict[str, Any]:
    a = get_assembly()
    cfg = a.config.get("telemetry") or {}
    client = ClickHouseClient(
        base_url=clickhouse_http_url_from_config(cfg),
        database=str(cfg.get("database", "otel")),
    )
    return telemetry_status(client)


@app.post("/cycle", response_model=RuntimeStatus)
def run_cycle(body: CycleRequest) -> RuntimeStatus:
    fault: Optional[FaultSpec] = None
    if body.fault is not None:
        if body.fault.type not in FAULT_TYPES:
            raise ValueError(f"unknown fault type: {body.fault.type}")
        ts = body.fault.injected_at or datetime.now(timezone.utc).isoformat()
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        fault = FaultSpec(type=body.fault.type, target=body.fault.target, injected_at=parsed)
    return get_orch().run_cycle(
        target=body.target or "",
        namespace=body.namespace,
        context_c=body.context_c,
        fault=fault,
    )


@app.get("/", response_class=HTMLResponse)
def status_page() -> str:
    return _STATUS_HTML


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_STATUS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AIOps Runtime Harness</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --border: #30363d; --fg: #e6edf3;
    --dim: #8b949e; --accent: #58a6ff; --green: #3fb950; --red: #f85149;
    --yellow: #d29922; --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--fg); font: 14px/1.5 -apple-system, sans-serif; padding: 20px; }
  h1 { font-size: 18px; margin-bottom: 4px; }
  .sub { color: var(--dim); font-size: 12px; margin-bottom: 18px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; max-width: 1200px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
  .card h2 { font-size: 13px; color: var(--dim); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 10px; }
  .wide { grid-column: 1 / -1; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-family: var(--mono); border: 1px solid var(--border); margin-right: 6px; }
  .b-healthy { color: var(--green); border-color: var(--green); }
  .b-incident { color: var(--red); border-color: var(--red); }
  .b-degraded { color: var(--yellow); border-color: var(--yellow); }
  .b-error { color: var(--red); border-color: var(--red); }
  .b-unknown { color: var(--dim); }
  .b-accent { color: var(--accent); border-color: var(--accent); }
  .row { display: flex; justify-content: space-between; padding: 3px 0; font-size: 13px; }
  .row .k { color: var(--dim); }
  .row .v { font-family: var(--mono); font-size: 12px; text-align: right; }
  .stages { display: flex; gap: 4px; margin: 10px 0; }
  .stage { flex: 1; text-align: center; font-size: 10px; padding: 6px 2px; border-radius: 4px; background: #21262d; color: var(--dim); border: 1px solid var(--border); }
  .stage.active { background: var(--accent); color: #000; font-weight: 600; border-color: var(--accent); }
  .stage.done { background: #1f2b1f; color: var(--green); border-color: #2ea04340; }
  input, select, button {
    background: #0d1117; color: var(--fg); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 10px; font-size: 13px; font-family: var(--mono);
  }
  button { cursor: pointer; }
  button:hover { border-color: var(--accent); }
  button.primary { background: #1f6feb; border-color: #1f6feb; color: #fff; }
  button.danger { background: transparent; border-color: var(--red); color: var(--red); }
  button.ok { background: transparent; border-color: var(--green); color: var(--green); }
  table { width: 100%; border-collapse: collapse; font-size: 12px; font-family: var(--mono); }
  th { text-align: left; color: var(--dim); font-weight: 500; padding: 4px 6px; border-bottom: 1px solid var(--border); }
  td { padding: 4px 6px; border-bottom: 1px solid #21262d; }
  canvas { width: 100%; height: 120px; }
  .controls { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; align-items: center; }
  .msg { font-size: 12px; color: var(--dim); margin-top: 8px; min-height: 16px; }
  .err { color: var(--red); }
  .ok-text { color: var(--green); }
  .chart-title { font-size: 11px; color: var(--dim); margin: 8px 0 2px; }
</style>
</head>
<body>
<h1>AIOps Runtime Harness</h1>
<div class="sub" id="subtitle">self-adopting closed-loop AIOps · own ClickHouse · Granite detect · GLM-5.3 correlate/RCA/remediate</div>

<div class="grid">
  <div class="card">
    <h2>Runtime status</h2>
    <div><span class="badge" id="health">unknown</span><span class="badge b-accent" id="assembly">—</span><span class="badge" id="dryrun">dry-run</span></div>
    <div class="stages" id="stages"></div>
    <div class="row"><span class="k">message</span><span class="v" id="message">—</span></div>
    <div class="row"><span class="k">target / component</span><span class="v" id="target">—</span></div>
    <div class="row"><span class="k">fault hypothesis</span><span class="v" id="fault">—</span></div>
    <div class="row"><span class="k">remediation</span><span class="v" id="remediation">—</span></div>
    <div class="row"><span class="k">MTTD / MTTR</span><span class="v" id="mtts">— / —</span></div>
    <div class="controls">
      <input id="target-input" placeholder="target (optional)" style="width:150px">
      <select id="drill-fault"><option value="">no drill fault</option><option>scale_zero</option><option>kill_pod</option><option>config_corruption</option></select>
      <button class="primary" onclick="runCycle()">Run cycle</button>
    </div>
    <div class="msg" id="cycle-msg"></div>
  </div>

  <div class="card">
    <h2>Triggers</h2>
    <div class="controls">
      <button id="trig-scheduler" onclick="setTrigMode('scheduler')">Scheduler (interval)</button>
      <button id="trig-alertmanager" onclick="setTrigMode('alertmanager')">Alertmanager (webhook)</button>
      <button id="trig-watcher" onclick="toggleWatcher()">Watcher: off</button>
    </div>
    <div class="row"><span class="k">primary mode</span><span class="v" id="trig-mode">—</span></div>
    <div class="row"><span class="k">namespaces</span><span class="v" id="trig-ns">—</span></div>
    <div class="row"><span class="k">scheduler</span><span class="v" id="sched-state">—</span></div>
    <div class="controls">
      <input id="interval-input" type="number" min="30" step="10" style="width:80px"> s
      <button class="primary" onclick="setInterval2()">Set interval</button>
      <button class="ok" id="toggle-btn" onclick="toggleScheduler()">Enable</button>
    </div>
    <div class="row"><span class="k">watcher checks</span><span class="v" id="watch-checks">—</span></div>
    <div class="row"><span class="k">last breaches</span><span class="v" id="watch-breaches">—</span></div>
    <div class="chart-title">recent triggers</div>
    <table id="trig-history"><tr><th>time</th><th>source</th><th>target</th><th>detail</th></tr></table>
    <div class="msg">Primary mode is exclusive: scheduler polls on an interval; alertmanager waits for POST /alert webhooks (T0 from startsAt). The watcher is an independent cheap sensor (error bursts + service silence) that wakes full cycles. Triggers are cues — the detector still verifies independently.</div>
  </div>

  <div class="card">
    <h2>Telemetry — own ClickHouse</h2>
    <div class="row"><span class="k">instance</span><span class="v" id="ch-url">—</span></div>
    <div class="row"><span class="k">reachable</span><span class="v" id="ch-ok">—</span></div>
    <div class="row"><span class="k">logs (5 min)</span><span class="v" id="ch-rate">—</span></div>
    <table id="ch-tables"><tr><th>table</th><th>last event</th><th>age</th><th>rows 1h</th></tr></table>
    <div class="chart-title">adopted services (2h window)</div>
    <table id="ch-services"><tr><th>service</th><th>signals</th></tr></table>
  </div>

  <div class="card">
    <h2>Context sets (RAG · BM25)</h2>
    <div class="controls" id="ctx-buttons">
      <button id="ctx-1" onclick="toggleCtx('1')">C1 source</button>
      <button id="ctx-2" onclick="toggleCtx('2')">C2 docs</button>
      <button id="ctx-3" onclick="toggleCtx('3')">C3 architecture</button>
      <button id="ctx-4" onclick="toggleCtx('4')">C4 dependencies</button>
    </div>
    <div class="msg">Active sets feed the enrichment stage (BM25 top-k chunks into detect/RCA prompts). C0 (none) weakens RCA for diagnosers — Oct-26 finding.</div>
    <div class="row"><span class="k">active</span><span class="v" id="ctx-active">—</span></div>
    <div class="row"><span class="k">anomaly detection</span><span class="v" id="m-detect">—</span></div>
    <div class="row"><span class="k">signal correlation / RCA</span><span class="v" id="m-rca">—</span></div>
    <div class="row"><span class="k">remediation</span><span class="v" id="m-rem">—</span></div>
    <div class="chart-title">endpoint wiring (keys never shown)</div>
    <table id="m-wiring"><tr><th>agent</th><th>served model</th><th>endpoint</th><th>key</th></tr></table>
    <div class="chart-title">evidence</div>
    <div class="msg" id="m-evidence">—</div>
  </div>

  <div class="card">
    <h2>Remediation mode</h2>
    <div class="controls">
      <button id="rem-hil" onclick="setRemMode('hil')">HIL — proposals</button>
      <button id="rem-governed" onclick="setRemMode('governed')">Governed — Ansible</button>
      <button id="rem-direct" onclick="setRemMode('direct')">Direct — MCP tools</button>
    </div>
    <div class="row"><span class="k">active</span><span class="v" id="rem-active">—</span></div>
    <div class="row"><span class="k">direct armed</span><span class="v" id="rem-armed">—</span></div>
    <div class="row"><span class="k">playbook catalog</span><span class="v" id="rem-playbooks">—</span></div>
    <div class="row"><span class="k">fault → playbook</span><span class="v" id="rem-map">—</span></div>
    <div class="msg">Mutually exclusive. HIL: model proposes, human decides. Governed: model selects a reviewed playbook; harness runs it (model never writes commands). Direct: guarded K8s writes (G1–G3) — requires AIOPS_RT_ENABLE_WRITE_TOOLS=1.</div>
  </div>

  <div class="card wide">
    <h2>MTTD / MTTR trends</h2>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px;">
      <div><div class="chart-title">MTTD (s) — detection latency from T0</div><canvas id="mttd-chart" width="560" height="120"></canvas></div>
      <div><div class="chart-title">MTTR (s) — remediation latency from T0</div><canvas id="mttr-chart" width="560" height="120"></canvas></div>
    </div>
    <table id="cycles-table">
      <tr><th>time</th><th>target</th><th>t0 source</th><th>detected</th><th>RCA</th><th>MTTD s</th><th>MTTR s</th><th>health</th></tr>
    </table>
  </div>
</div>

<script>
const STAGES = ["enrich","telemetry","detect","rca","remediate","verify"];
let trendsCache = [];

async function jget(u) { const r = await fetch(u); return r.json(); }

function fmtAge(s) {
  if (s == null) return "—";
  if (s < 90) return s.toFixed(0) + "s";
  if (s < 5400) return (s/60).toFixed(1) + "m";
  return (s/3600).toFixed(1) + "h";
}

async function refreshStatus() {
  const s = await jget("/status");
  const h = document.getElementById("health");
  h.textContent = s.health; h.className = "badge b-" + s.health;
  document.getElementById("assembly").textContent = s.assembly;
  document.getElementById("dryrun").textContent =
    "mode: " + (s.remediation_mode || "hil") + (s.human_in_the_loop ? " · gated" : "");
  const order = STAGES.indexOf(s.stage.value === "complete" || s.stage.value === "error" ? "" : s.stage.value);
  document.getElementById("stages").innerHTML = STAGES.map((st, i) => {
    let cls = "stage";
    if (i < order || (s.stage.value === "complete" && s.health !== "error")) cls += " done";
    if (i === order) cls += " active";
    return `<div class="${cls}">${st}</div>`;
  }).join("");
  document.getElementById("message").textContent = s.message;
  const i = s.incident;
  document.getElementById("target").textContent = i.component || "—";
  document.getElementById("fault").textContent = i.fault_hypothesis || "—";
  document.getElementById("remediation").textContent = (i.remediation_proposed || "—") + (i.remediation_executed ? " [executed]" : "");
  document.getElementById("mtts").textContent =
    (i.mttd_s != null ? i.mttd_s.toFixed(1) : "—") + " / " +
    (i.mttr_s != null ? i.mttr_s.toFixed(1) : "—");
}

async function refreshScheduler() {
  const t = await jget("/triggers");
  document.getElementById("trig-mode").textContent = t.mode;
  document.getElementById("trig-ns").textContent = (t.namespaces || []).join(", ");
  for (const m of ["scheduler","alertmanager"]) {
    const btn = document.getElementById("trig-" + m);
    btn.className = (t.mode === m) ? "primary" : "";
    btn.style.opacity = (t.mode === m) ? "1" : "0.55";
  }
  const wb = document.getElementById("trig-watcher");
  const w = t.watcher || {};
  wb.textContent = "Watcher: " + (w.enabled ? "on" : "off");
  wb.className = w.enabled ? "primary" : "";
  document.getElementById("watch-checks").textContent = w.checks_run != null ? w.checks_run : "—";
  const br = (w.breaches || []).slice(-2).map(b => (b.service || "?") + " " + (b.reason || b.error || "")).join(" · ");
  document.getElementById("watch-breaches").textContent = br || "—";
  const s = t.scheduler || {};
  document.getElementById("sched-state").textContent =
    (s.enabled && t.mode === "scheduler" ? "enabled" : s.enabled ? "paused (mode=" + t.mode + ")" : "disabled");
  document.getElementById("sched-state").style.color = (s.enabled && t.mode === "scheduler") ? "var(--green)" : "var(--dim)";
  if (!document.getElementById("interval-input").value)
    document.getElementById("interval-input").value = s.interval_s;
  const btn = document.getElementById("toggle-btn");
  btn.textContent = s.enabled ? "Disable" : "Enable";
  btn.className = s.enabled ? "danger" : "ok";
  const th = document.getElementById("trig-history");
  th.innerHTML = "<tr><th>time</th><th>source</th><th>target</th><th>detail</th></tr>" +
    (t.history || []).slice().reverse().map(h =>
      `<tr><td>${(h.ts||"").slice(11,19)}</td><td>${h.source}</td><td>${h.target || "—"}</td><td>${(h.detail||"").slice(0,40)}</td></tr>`).join("");
}

async function setTrigMode(m) {
  await fetch("/triggers", { method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify({ mode: m }) });
  refreshScheduler();
}

async function toggleWatcher() {
  const t = await jget("/triggers");
  const on = !(t.watcher || {}).enabled;
  await fetch("/triggers", { method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify({ watcher_enabled: on }) });
  refreshScheduler();
}

async function refreshTelemetry() {
  const t = await jget("/telemetry-status");
  document.getElementById("ch-url").textContent = t.url;
  const ok = document.getElementById("ch-ok");
  ok.textContent = t.reachable ? "yes" : "NO";
  ok.style.color = t.reachable ? "var(--green)" : "var(--red)";
  document.getElementById("ch-rate").textContent = t.logs_last_5m != null ? t.logs_last_5m : "—";
  const tt = document.getElementById("ch-tables");
  tt.innerHTML = "<tr><th>table</th><th>last event</th><th>age</th><th>rows 1h</th></tr>" +
    (t.tables || []).map(r =>
      `<tr><td>${r.table}</td><td>${r.last_event}</td><td>${fmtAge(r.age_s)}</td><td>${r.rows_1h}</td></tr>`).join("");
  const st = document.getElementById("ch-services");
  st.innerHTML = "<tr><th>service</th><th>signals</th></tr>" +
    (t.services || []).slice(0, 8).map(s =>
      `<tr><td>${s.service}</td><td>${s.signal_count}</td></tr>`).join("");
}

async function refreshModels() {
  const m = await jget("/models");
  document.getElementById("m-detect").textContent = m.anomaly_detection.primary + (m.anomaly_detection.secondary ? " + " + m.anomaly_detection.secondary : "");
  document.getElementById("m-rca").textContent = m.rca.primary + (m.rca.confirmer ? " + " + m.rca.confirmer : "");
  document.getElementById("m-rem").textContent = m.remediation.primary;
  const wt = document.getElementById("m-wiring");
  wt.innerHTML = "<tr><th>agent</th><th>served model</th><th>endpoint</th><th>key</th></tr>" +
    (m.wiring || []).map(w =>
      `<tr><td>${w.id}</td><td>${w.model}</td>` +
      `<td title="${w.env.base}">${w.base_url ? w.base_url.replace(/^https?:\\/\\//,'').slice(0,42) : "NOT SET"}</td>` +
      `<td style="color:${w.api_key_set ? "var(--green)" : "var(--red)"}">${w.api_key_set ? "✓" : "✗"}</td></tr>`).join("");
  const a = await jget("/assembly");
  document.getElementById("m-evidence").textContent = a.evidence_report || "—";
}

async function refreshTrends() {
  const t = await jget("/trends");
  trendsCache = t.cycles || [];
  drawChart("mttd-chart", trendsCache.map(c => c.mttd_s), "#58a6ff");
  drawChart("mttr-chart", trendsCache.map(c => c.mttr_s), "#d29922");
  const tb = document.getElementById("cycles-table");
  tb.innerHTML = "<tr><th>time</th><th>target</th><th>t0 source</th><th>detected</th><th>RCA</th><th>MTTD s</th><th>MTTR s</th><th>health</th></tr>" +
    trendsCache.slice(-10).reverse().map(c =>
      `<tr><td>${(c.ts||"").slice(11,19)}</td><td>${c.target}</td><td>${c.t0_source}</td>` +
      `<td>${c.detected ? "✓" : "—"}</td><td>${c.rca_component || "—"}</td>` +
      `<td>${c.mttd_s != null ? c.mttd_s.toFixed(1) : "—"}</td>` +
      `<td>${c.mttr_s != null ? c.mttr_s.toFixed(1) : "—"}</td><td>${c.health}</td></tr>`).join("");
}

function drawChart(id, values, color) {
  const cv = document.getElementById(id);
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const vals = values.filter(v => v != null);
  if (!vals.length) {
    ctx.fillStyle = "#8b949e"; ctx.font = "12px sans-serif";
    ctx.fillText("no cycles yet", 10, 60); return;
  }
  const max = Math.max(...vals) * 1.15, min = 0;
  const n = values.length;
  ctx.strokeStyle = "#30363d"; ctx.beginPath();
  ctx.moveTo(0, 119); ctx.lineTo(560, 119); ctx.stroke();
  values.forEach((v, i) => {
    if (v == null) return;
    const x = (i / Math.max(1, n - 1)) * 550 + 5;
    const y = 115 - ((v - min) / (max - min)) * 105;
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(x, y, 2.5, 0, 7); ctx.fill();
  });
  // connect detected points
  ctx.strokeStyle = color + "80"; ctx.beginPath(); let started = false;
  values.forEach((v, i) => {
    if (v == null) { started = false; return; }
    const x = (i / Math.max(1, n - 1)) * 550 + 5;
    const y = 115 - ((v - min) / (max - min)) * 105;
    if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#8b949e"; ctx.font = "10px monospace";
  ctx.fillText("max " + max.toFixed(0) + "s", 4, 10);
  ctx.fillText("n=" + vals.length, 500, 10);
}

async function runCycle() {
  const msg = document.getElementById("cycle-msg");
  msg.textContent = "cycle running…"; msg.className = "msg";
  const body = {};
  const t = document.getElementById("target-input").value.trim();
  if (t) body.target = t;
  const f = document.getElementById("drill-fault").value;
  if (f) body.fault = { type: f, target: t || "cart" };
  const r = await fetch("/cycle", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body) });
  const s = await r.json();
  msg.textContent = r.ok ? s.message : ("error: " + (s.detail || s.message));
  msg.className = "msg " + (r.ok && s.health !== "error" ? "" : "err");
  refreshAll();
}

async function setInterval2() {
  const v = parseInt(document.getElementById("interval-input").value);
  if (!v || v < 30) { alert("interval must be >= 30s"); return; }
  await fetch("/scheduler", { method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify({ interval_s: v }) });
  refreshScheduler();
}

async function toggleScheduler() {
  const s = await jget("/scheduler");
  await fetch("/scheduler", { method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify({ enabled: !s.enabled }) });
  refreshScheduler();
}

async function refreshContext() {
  const c = await jget("/context");
  document.getElementById("ctx-active").textContent = "C" + (c.active || []).join("+") || "none";
  for (const i of ["1","2","3","4"]) {
    const btn = document.getElementById("ctx-" + i);
    const on = (c.active || []).includes(i);
    btn.className = on ? "primary" : "";
    btn.style.opacity = on ? "1" : "0.55";
  }
}

async function toggleCtx(i) {
  const c = await jget("/context");
  let sets = c.active || [];
  sets = sets.includes(i) ? sets.filter(x => x !== i) : sets.concat(i);
  await fetch("/context", { method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify({ sets }) });
  refreshContext();
}

async function refreshRemediation() {
  const r = await jget("/remediation");
  document.getElementById("rem-active").textContent = r.mode;
  document.getElementById("rem-armed").textContent = r.direct_armed ? "yes" : "NO";
  document.getElementById("rem-armed").style.color = r.direct_armed ? "var(--green)" : "var(--dim)";
  document.getElementById("rem-playbooks").textContent = (r.playbooks || []).join(", ") || "—";
  document.getElementById("rem-map").textContent = Object.entries(r.fault_playbook_map || {}).map(([k,v]) => k+"→"+v).join(" · ") || "—";
  for (const m of ["hil","governed","direct"]) {
    const btn = document.getElementById("rem-" + m);
    btn.className = (r.mode === m) ? "primary" : "";
    btn.style.opacity = (r.mode === m) ? "1" : "0.55";
  }
}

async function setRemMode(m) {
  const r = await fetch("/remediation", { method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify({ mode: m }) });
  if (!r.ok) { const e = await r.json(); alert(e.detail || "mode change failed"); }
  refreshRemediation();
}

function refreshAll() {
  refreshStatus(); refreshScheduler(); refreshTelemetry(); refreshModels(); refreshTrends(); refreshContext(); refreshRemediation();
}
refreshAll();
setInterval(refreshAll, 3000);
</script>
</body>
</html>
"""
