#!/usr/bin/env bash
# Seed OpenClaw workspace skills, MCP, capture-proxy, AGENTS.md (v18).
# Presenter docs: PRESENTER-RUNBOOK.md · SECURITY-DEMO.md
# Agent captures flows in the Control UI via netobserv-capture-proxy — no pre-seeded evidence.
#   - harden seed-openclaw init for power-cycles (NETOBSERV_SEED_HARDENED_v5; Control UI logo + @openclaw/slack)
#   - RBAC, heal MCP/proxy, optional RH MCP (ENABLE_OPENSHIFT_MCP=1), sandbox flatten
#   - pin lab openclaw.json (skills, remote mode, scope=agent, primary-model thinking policy, compaction)
#   - stage workspace skills, TOOLS.md, AGENTS.md v18; clears stale evidence/
#   - patch OpenShell upload nesting + reload gateway (same EmptyDir)
#   - recreate OpenShell sandboxes
# Lab LLM (outside kit): ~/labs/openshell-on-openshift-lab/manifests/openclaw/config.yaml
#   Current bastion: LiteMaaS Qwen3.6-35B-A3B, contextWindow 131072, maxTokens 12288
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
OPENSHELL_NS="${OPENSHELL_NS:-openshell}"
API_SERVER="${API_SERVER:-}"
# Prefer the public OpenShift API hostname: OpenShell network policy matches hosts,
# and agent exec is proxied (kubectl exec is not). In-cluster DNS often gets RST.
API_SERVER_MODE="${API_SERVER_MODE:-external}"  # external | incluster
LAB_CFG="${LAB_CFG:-$HOME/labs/openshell-on-openshift-lab/manifests/openclaw/config.yaml}"
LAB_OPENCLAW_DIR="${LAB_OPENCLAW_DIR:-$(dirname "$LAB_CFG")}"

c_blue=$'\033[1;34m'; c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_reset=$'\033[0m'
step() { printf '%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }

KUBECTL="$(command -v oc || command -v kubectl)"
[[ -n "$KUBECTL" ]] || { echo "oc/kubectl required" >&2; exit 1; }

RHOAI_NS="${RHOAI_NS:-redhat-ods-applications}"
# demo-a-fast / demo-a often run without env flags; auto-wire when RHOAI MLflow is already on-cluster.
if [[ "${ENABLE_RHOAI_PLATFORM:-0}" != "1" && "${ENABLE_OPENCLAW_MLFLOW:-0}" != "1" ]]; then
  if "$KUBECTL" -n "$RHOAI_NS" get deploy/mlflow >/dev/null 2>&1; then
    export ENABLE_RHOAI_PLATFORM=1
    step "Auto-detected RHOAI MLflow in ${RHOAI_NS} — MCP audit will use rhoai backend"
  fi
fi

resolve_openclaw_pod() {
  local name=""
  for _ in $(seq 1 45); do
    name="$("$KUBECTL" -n "$OPENCLAW_NS" get pods \
      -l app.kubernetes.io/name=openclaw \
      --field-selector=status.phase=Running \
      -o json 2>/dev/null | python3 -c '
import json,sys
items=json.load(sys.stdin).get("items") or []
for it in items:
  owners=it.get("metadata",{}).get("ownerReferences") or []
  if any(o.get("kind")=="ReplicaSet" for o in owners):
    ready=(it.get("status") or {}).get("containerStatuses") or []
    if ready and all(c.get("ready") for c in ready):
      print(it["metadata"]["name"]); break
' 2>/dev/null || true)"
    if [[ -n "$name" ]]; then
      POD="$name"
      return 0
    fi
    sleep 2
  done
  return 1
}

step "Apply limited RBAC"
"$KUBECTL" apply -f "$ROOT/openclaw-skills/manifests/openclaw-netobserv-rbac.yaml"

step "Harden OpenClaw seed-openclaw init (power-cycle / CrashLoopBackOff guard)"
chmod +x "$ROOT/scripts/patch-openclaw-seed-idempotent.sh"
# Apply with the later kustomize pass when LAB_CFG exists; otherwise apply now.
if [[ -f "$LAB_CFG" ]]; then
  APPLY_LAB_OPENCLAW=0 LAB_OPENCLAW_DIR="$LAB_OPENCLAW_DIR" \
    "$ROOT/scripts/patch-openclaw-seed-idempotent.sh" || \
    warn "could not patch lab seed-openclaw (CrashLoop risk after power-cycle)"
else
  LAB_OPENCLAW_DIR="$LAB_OPENCLAW_DIR" \
    "$ROOT/scripts/patch-openclaw-seed-idempotent.sh" || \
    warn "could not patch lab seed-openclaw (CrashLoop risk after power-cycle)"
fi

if "$KUBECTL" -n "$OPENCLAW_NS" get secret openclaw-slack-tokens >/dev/null 2>&1; then
  step "Slack tokens detected — ensure @openclaw/slack in seed init (v5)"
  chmod +x "$ROOT/scripts/wire-openclaw-slack.sh"
  RECYCLE_POD=0 SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}" OPENCLAW_NS="$OPENCLAW_NS" \
    "$ROOT/scripts/wire-openclaw-slack.sh" || \
    warn "wire-openclaw-slack failed — see docs/SLACK-PRESENTER-GUIDE.md"
fi

step "Deploy OpenShell sandbox flatten loop (safety net for nested upload skill paths)"
"$KUBECTL" apply -f "$ROOT/openclaw-skills/manifests/netobserv-sandbox-flatten.yaml"
"$KUBECTL" -n "$OPENCLAW_NS" rollout restart deploy/netobserv-sandbox-flatten 2>/dev/null || true
"$KUBECTL" -n "$OPENCLAW_NS" rollout status deploy/netobserv-sandbox-flatten --timeout=120s || \
  warn "sandbox-flatten rollout not ready yet"

step "Deploy in-cluster heal proxy (HTTP fallback for bash skill)"
"$KUBECTL" -n "$OPENCLAW_NS" create configmap netobserv-heal-proxy-scripts \
  --from-file=netobserv-cluster-heal.py="$ROOT/openclaw-skills/netobserv-heal/scripts/netobserv-cluster-heal.py" \
  --from-file=netobserv-heal-proxy.py="$ROOT/openclaw-skills/netobserv-heal/scripts/netobserv-heal-proxy.py" \
  --dry-run=client -o yaml | "$KUBECTL" apply -f -
"$KUBECTL" apply -f "$ROOT/openclaw-skills/manifests/netobserv-heal-proxy.yaml"
"$KUBECTL" -n "$OPENCLAW_NS" rollout restart deploy/netobserv-heal-proxy
"$KUBECTL" -n "$OPENCLAW_NS" rollout status deploy/netobserv-heal-proxy --timeout=120s || \
  warn "heal proxy rollout not ready yet"

step "Deploy NetObserv MCP server (investigation — gateway → in-cluster Streamable HTTP)"
"$KUBECTL" -n "$OPENCLAW_NS" create configmap netobserv-mcp-scripts \
  --from-file=server.py="$ROOT/openclaw-skills/mcp-server/server.py" \
  --from-file=mlflow_tracing.py="$ROOT/openclaw-skills/mcp-server/mlflow_tracing.py" \
  --from-file=requirements.txt="$ROOT/openclaw-skills/mcp-server/requirements.txt" \
  --from-file=netobserv-cluster-heal.py="$ROOT/openclaw-skills/netobserv-heal/scripts/netobserv-cluster-heal.py" \
  --from-file=summarize-evidence.py="$ROOT/openclaw-skills/netobserv-evidence/scripts/summarize-evidence.py" \
  --dry-run=client -o yaml | "$KUBECTL" apply -f -
"$KUBECTL" apply -f "$ROOT/openclaw-skills/manifests/netobserv-mcp.yaml"
"$KUBECTL" -n "$OPENCLAW_NS" rollout restart deploy/netobserv-mcp
"$KUBECTL" -n "$OPENCLAW_NS" rollout status deploy/netobserv-mcp --timeout=240s || \
  warn "heal MCP server rollout not ready yet"

step "Deploy Red Hat OpenShift MCP server (netobserv toolset, read-only)"
if [[ "${SKIP_OPENSHIFT_MCP:-0}" != "1" ]]; then
  chmod +x "$ROOT/scripts/deploy-openshift-mcp-server.sh"
  OPENCLAW_NS="$OPENCLAW_NS" "$ROOT/scripts/deploy-openshift-mcp-server.sh" || \
    warn "openshift-mcp-server deploy failed (heal MCP still available)"
else
  warn "SKIP_OPENSHIFT_MCP=1 — skipping Red Hat MCP chart"
fi

if [[ "${ENABLE_RHOAI_PLATFORM:-0}" == "1" ]]; then
  step "RHOAI platform MLflow enabled (expects install-rhoai-platform-minimal.sh already run)"
elif [[ "${ENABLE_OPENCLAW_MLFLOW:-0}" == "1" ]]; then
  step "Deploy MLflow for OpenClaw audit traces (ENABLE_OPENCLAW_MLFLOW=1)"
  chmod +x "$ROOT/scripts/deploy-openclaw-mlflow.sh" "$ROOT/scripts/wire-openclaw-mlflow.sh"
  OPENCLAW_NS="$OPENCLAW_NS" "$ROOT/scripts/deploy-openclaw-mlflow.sh" || \
    warn "MLflow deploy failed — see docs/OPENCLAW-MLFLOW-DESIGN.md"
else
  step "MLflow audit off (set ENABLE_OPENCLAW_MLFLOW=1 for standalone; RHOAI auto-detected when deploy/mlflow exists)"
fi

# Pin skill allowlist + keep openshell remote in lab config when present
if [[ "${CONFIG_ONLY:-0}" == "1" ]] || [[ -f "$LAB_CFG" ]]; then
  if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
    step "Patch OpenClaw lab config only (ansible-automation MCP)"
    [[ -f "$LAB_CFG" ]] || { warn "LAB_CFG missing: $LAB_CFG"; exit 1; }
  else
    step "Pin agents.defaults.skills allowlist + openshell remote in lab config"
  fi
  if [[ "${ENABLE_RHOAI_PLATFORM:-0}" == "1" || "${ENABLE_OPENCLAW_MLFLOW:-0}" == "1" ]]; then
    export _SEED_ENABLE_MLFLOW=1
  else
    export _SEED_ENABLE_MLFLOW=0
  fi
  if [[ "${ENABLE_OPENCLAW_OTEL:-0}" == "1" ]]; then
    export _SEED_ENABLE_OTEL=1
    export _SEED_OTEL_ENDPOINT="${OTEL_ENDPOINT:-http://openclaw-otel-collector.${OPENCLAW_NS}.svc:4318}"
  else
    export _SEED_ENABLE_OTEL=0
  fi
  CFG_OUT="$(python3 - "$LAB_CFG" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text())
defaults = d.setdefault("agents", {}).setdefault("defaults", {})
wanted = ["netobserv-investigate", "netobserv-evidence", "netobserv-heal", "netobserv-live-flows", "ansible-automation"]
# Qwen thinking: off on legacy 16k (empty replies); on 128k LiteMaaS enable when contextWindow >= 65536.
# Override: ENABLE_QWEN_THINKING=1|0
primary = (defaults.get("model") or {}).get("primary") or "openai/Qwen3.6-35B-A3B"
primary_id = primary.split("/", 1)[1] if "/" in primary else primary
providers_models = (
    d.setdefault("models", {})
    .setdefault("providers", {})
    .setdefault("openai", {})
    .setdefault("models", [])
)
primary_ctx = next(
    (m.get("contextWindow") for m in providers_models if isinstance(m, dict) and m.get("id") == primary_id),
    0,
) or 0
os_en = __import__("os").environ
think_flag = os_en.get("ENABLE_QWEN_THINKING", "").strip().lower()
if think_flag in ("0", "false", "no"):
    enable_thinking = False
elif think_flag in ("1", "true", "yes"):
    enable_thinking = True
else:
    enable_thinking = primary_ctx >= 65536
wanted_think = {"enable_thinking": enable_thinking}
model_params = defaults.setdefault("models", {}).setdefault(primary, {})
cur_params = model_params.setdefault("params", {})
cur_ctk = cur_params.get("chat_template_kwargs") or {}
cur_extra = cur_params.get("extra_body") if isinstance(cur_params.get("extra_body"), dict) else {}
oshell = d.setdefault("plugins", {}).setdefault("entries", {}).setdefault("openshell", {})
cfg = oshell.setdefault("config", {})
mcp_servers = d.setdefault("mcp", {}).setdefault("servers", {})
# Thin domain investigation MCP (custom)
heal_mcp = {
    "url": "http://netobserv-mcp.openclaw.svc.cluster.local:8080/mcp",
    "transport": "streamable-http",
    "enabled": True,
    "connectionTimeoutMs": 45000,
    "requestTimeoutMs": 240000,
}
ansible_mcp_enabled = os_en.get("ENABLE_ANSIBLE_MCP", "").strip().lower() in ("1", "true", "yes")
ansible_mcp = {
    "url": "http://ansible-mcp.openclaw.svc.cluster.local:8080/mcp",
    "transport": "streamable-http",
    "enabled": ansible_mcp_enabled,
    "connectionTimeoutMs": 45000,
    "requestTimeoutMs": 300000,
}
# Red Hat OpenShift MCP (netobserv reads). Auto-on when primary contextWindow >= 65536 (128k LiteMaaS).
# Override: ENABLE_OPENSHIFT_MCP=1|0 ; DISABLE_OPENSHIFT_MCP=1 forces off.
if __import__("os").environ.get("DISABLE_OPENSHIFT_MCP", "").strip().lower() in (
    "1",
    "true",
    "yes",
):
    openshift_mcp_enabled = False
else:
    mcp_flag = __import__("os").environ.get("ENABLE_OPENSHIFT_MCP", "").strip().lower()
    if mcp_flag in ("0", "false", "no"):
        openshift_mcp_enabled = False
    elif mcp_flag in ("1", "true", "yes"):
        openshift_mcp_enabled = True
    else:
        openshift_mcp_enabled = primary_ctx >= 65536
openshift_mcp = {
    "url": "http://openshift-mcp.openclaw.svc.cluster.local:8080/mcp",
    "transport": "streamable-http",
    "enabled": openshift_mcp_enabled,
    "connectionTimeoutMs": 45000,
    "requestTimeoutMs": 240000,
}
# Sandbox tool policy hides MCP unless alsoAllow includes bundle-mcp / server globs
sandbox_defaults = defaults.setdefault("sandbox", {})
sandbox_also = (
    d.setdefault("tools", {})
    .setdefault("sandbox", {})
    .setdefault("tools", {})
    .setdefault("alsoAllow", [])
)
also_wanted = [
    "bundle-mcp",
    "netobserv-openshift__*",
]
if ansible_mcp_enabled:
    also_wanted.append("ansible-automation__*")
if openshift_mcp_enabled:
    also_wanted.append("openshift-mcp__*")
changed = False
# Drop RH glob when MCP is disabled so sandbox does not advertise dead tools
if not openshift_mcp_enabled and "openshift-mcp__*" in sandbox_also:
    sandbox_also[:] = [x for x in sandbox_also if x != "openshift-mcp__*"]
    changed = True
if defaults.get("skills") != wanted:
    defaults["skills"] = wanted
    changed = True
if model_params.get("alias") != primary_id:
    model_params["alias"] = primary_id
    changed = True
# Apply thinking policy via extra_body (openai-completions).
# Drop duplicate top-level chat_template_kwargs to avoid overwrite warnings.
if "chat_template_kwargs" in cur_params:
    del cur_params["chat_template_kwargs"]
    changed = True
extra_ctk = cur_extra.get("chat_template_kwargs") if isinstance(cur_extra.get("chat_template_kwargs"), dict) else {}
if extra_ctk.get("enable_thinking") != enable_thinking or "extra_body" not in cur_params:
    cur_params["extra_body"] = {
        **{k: v for k, v in cur_extra.items() if k != "chat_template_kwargs"},
        "chat_template_kwargs": {**extra_ctk, **wanted_think},
    }
    changed = True
# Compaction: leave room after large evidence turns (else heal hits stopReason=length)
compaction = defaults.setdefault("compaction", {})
wanted_compaction = {
    "mode": "default",
    "reserveTokens": 4096,
    "reserveTokensFloor": 4096,
    "keepRecentTokens": 2048,
    "maxHistoryShare": 0.5,
    "recentTurnsPreserve": 2,
}
for k, v in wanted_compaction.items():
    if compaction.get(k) != v:
        compaction[k] = v
        changed = True
# Provider catalog: pin 16k only for legacy qwen3-14b; sync primary model limits when set in config
primary_model = (defaults.get("model") or {}).get("primary") or ""
primary_id = primary_model.split("/", 1)[1] if "/" in primary_model else primary_model
for m in (
    d.setdefault("models", {})
    .setdefault("providers", {})
    .setdefault("openai", {})
    .setdefault("models", [])
):
    if not isinstance(m, dict):
        continue
    mid = m.get("id")
    if mid == "qwen3-14b":
        if m.get("maxTokens") != 6144:
            m["maxTokens"] = 6144
            changed = True
        if m.get("contextWindow") != 16384:
            m["contextWindow"] = 16384
            changed = True
        mp = m.setdefault("params", {})
        mctk = mp.get("chat_template_kwargs") if isinstance(mp.get("chat_template_kwargs"), dict) else {}
        if mctk.get("enable_thinking") is not False:
            mp["chat_template_kwargs"] = {**mctk, "enable_thinking": False}
            changed = True
    elif primary_id and mid == primary_id:
        # Preserve large-context LiteMaaS / vLLM tuning if already set; default 128k / 12k
        want_ctx = m.get("contextWindow") or 131072
        want_max = m.get("maxTokens") or 12288
        if want_ctx >= 65536:
            pass  # respect operator-configured large window
        elif m.get("contextWindow") != want_ctx:
            m["contextWindow"] = want_ctx
            changed = True
        if m.get("maxTokens") != want_max:
            m["maxTokens"] = want_max
            changed = True
if not oshell.get("enabled", False):
    oshell["enabled"] = True
    changed = True
if cfg.get("mode") != "remote":
    cfg["mode"] = "remote"
    changed = True
# Per-session sandboxes re-seed nested openclaw-openshell-upload-* trees on every
# /new, so skill reads miss /sandbox/.openclaw/sandbox-skills/skills/... Use agent
# scope so one flattened sandbox is reused across Control UI sessions.
if sandbox_defaults.get("scope") != "agent":
    sandbox_defaults["scope"] = "agent"
    changed = True
if sandbox_defaults.get("backend") != "openshell":
    sandbox_defaults["backend"] = "openshell"
    changed = True
if sandbox_defaults.get("mode") != "all":
    sandbox_defaults["mode"] = "all"
    changed = True
if sandbox_defaults.get("workspaceAccess") != "rw":
    sandbox_defaults["workspaceAccess"] = "rw"
    changed = True
for name, wanted_cfg in (
    ("netobserv-openshift", heal_mcp),
    ("ansible-automation", ansible_mcp),
    ("openshift-mcp", openshift_mcp),
):
    cur = mcp_servers.get(name) or {}
    if {k: cur.get(k) for k in wanted_cfg} != wanted_cfg:
        mcp_servers[name] = {**cur, **wanted_cfg}
        changed = True
for item in also_wanted:
    if item not in sandbox_also:
        sandbox_also.append(item)
        changed = True
# @mlflow/mlflow-openclaw npm plugin is broken on OpenClaw 2026.6.11 — always strip stale pins.
plugins_root = d.setdefault("plugins", {})
allow = plugins_root.get("allow") or []
if "mlflow-openclaw" in allow:
    plugins_root["allow"] = [x for x in allow if x != "mlflow-openclaw"]
    changed = True
entries = plugins_root.get("entries") or {}
if "mlflow-openclaw" in entries:
    del entries["mlflow-openclaw"]
    plugins_root["entries"] = entries
    changed = True
# MLflow audit: MCP tool spans on netobserv-mcp (see OPENCLAW-MLFLOW-DESIGN.md).
mlflow_flag = os_en.get("_SEED_ENABLE_MLFLOW", "").strip().lower()
# Phase D OTel (optional — also run install-openclaw-otel-grafana.sh for collector + Grafana).
otel_flag = os_en.get("_SEED_ENABLE_OTEL", "").strip().lower()
otel_endpoint = os_en.get("_SEED_OTEL_ENDPOINT", "http://openclaw-otel-collector.openclaw.svc:4318")
if otel_flag in ("1", "true", "yes"):
    allow_cur = plugins_root.get("allow") or []
    if "diagnostics-otel" not in allow_cur:
        plugins_root["allow"] = allow_cur + ["diagnostics-otel"]
        changed = True
    entries = plugins_root.setdefault("entries", {})
    if entries.get("diagnostics-otel") != {"enabled": True}:
        entries["diagnostics-otel"] = {"enabled": True}
        changed = True
    diag = d.setdefault("diagnostics", {})
    if diag.get("enabled") is not True:
        diag["enabled"] = True
        changed = True
    otel = diag.setdefault("otel", {})
    wanted_otel = {
        "enabled": True,
        "endpoint": otel_endpoint,
        "protocol": "http/protobuf",
        "serviceName": "openclaw-gateway",
        "traces": True,
        "metrics": True,
        "logs": False,
        "sampleRate": 1.0,
        "flushIntervalMs": 15000,
    }
    if {k: otel.get(k) for k in wanted_otel} != wanted_otel:
        otel.update(wanted_otel)
        changed = True
if changed:
    p.write_text(json.dumps(d, indent=2) + "\n")
print("skills allowlist:", defaults.get("skills"))
print("openshell.mode:", cfg.get("mode"))
print("sandbox.scope:", sandbox_defaults.get("scope"))
print("primary model:", primary, "enable_thinking:", enable_thinking, "contextWindow:", primary_ctx)
print("primary maxTokens:", next((m.get("maxTokens") for m in d.get("models",{}).get("providers",{}).get("openai",{}).get("models",[]) if isinstance(m,dict) and m.get("id")==primary_id), None))
print("primary contextWindow:", next((m.get("contextWindow") for m in d.get("models",{}).get("providers",{}).get("openai",{}).get("models",[]) if isinstance(m,dict) and m.get("id")==primary_id), None))
print("compaction.keepRecentTokens:", compaction.get("keepRecentTokens"))
print("mcp.netobserv-openshift:", mcp_servers.get("netobserv-openshift", {}).get("url"))
print("mcp.ansible-automation:", mcp_servers.get("ansible-automation", {}).get("url"),
      "enabled=", mcp_servers.get("ansible-automation", {}).get("enabled"))
print("mcp.openshift-mcp:", mcp_servers.get("openshift-mcp", {}).get("url"),
      "enabled=", mcp_servers.get("openshift-mcp", {}).get("enabled"))
print("tools.sandbox.tools.alsoAllow:", sandbox_also)
print("mlflow_audit:", "mcp-spans" if mlflow_flag in ("1", "true", "yes") else "off")
print("otel_export:", otel_endpoint if otel_flag in ("1", "true", "yes") else "off")
print("config_changed:", changed)
PY
)"
  printf '%s\n' "$CFG_OUT"
  "$KUBECTL" -n "$OPENCLAW_NS" apply -k "$(dirname "$LAB_CFG")"
  # Guard against accidental single-key ConfigMap overwrites (crashes seed-openclaw).
  _cm_json="$("$KUBECTL" -n "$OPENCLAW_NS" get cm openclaw-config -o jsonpath='{.data.openclaw\.json}' 2>/dev/null || true)"
  _cm_pol="$("$KUBECTL" -n "$OPENCLAW_NS" get cm openclaw-config -o jsonpath='{.data.openclaw-managed-policy\.yaml}' 2>/dev/null || true)"
  if [[ -z "$_cm_json" || -z "$_cm_pol" ]]; then
    warn "openclaw-config missing keys after apply — retrying kustomize"
    "$KUBECTL" -n "$OPENCLAW_NS" apply -k "$(dirname "$LAB_CFG")"
  fi
  if grep -q 'config_changed: True' <<<"$CFG_OUT"; then
    if [[ "${ENABLE_OPENCLAW_MLFLOW:-0}" == "1" || "${ENABLE_RHOAI_PLATFORM:-0}" == "1" ]]; then
      step "Lab config changed; defer openclaw restart to wire-openclaw-mlflow.sh"
    else
      step "Lab config changed; restart openclaw"
      "$KUBECTL" -n "$OPENCLAW_NS" rollout restart deploy/openclaw
      "$KUBECTL" -n "$OPENCLAW_NS" rollout status deploy/openclaw --timeout=180s
    fi
  elif [[ "${ENABLE_OPENCLAW_MLFLOW:-0}" == "1" || "${ENABLE_RHOAI_PLATFORM:-0}" == "1" ]]; then
    step "MLflow enabled; defer openclaw restart to wire-openclaw-mlflow.sh"
  else
    step "Lab config unchanged; skip forced restart"
    "$KUBECTL" -n "$OPENCLAW_NS" rollout status deploy/openclaw --timeout=180s 2>/dev/null || true
  fi
  if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
    ok "OpenClaw lab config patched (CONFIG_ONLY)"
    exit 0
  fi
fi

if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
  exit 0
fi

if [[ "${ENABLE_RHOAI_PLATFORM:-0}" == "1" ]]; then
  step "Wire OpenClaw → RHOAI MLflow"
  chmod +x "$ROOT/scripts/wire-openclaw-mlflow.sh"
  OPENCLAW_NS="$OPENCLAW_NS" MLFLOW_BACKEND=rhoai \
    MLFLOW_REMOVE_STANDALONE="${MLFLOW_REMOVE_STANDALONE:-1}" \
    "$ROOT/scripts/wire-openclaw-mlflow.sh" || \
    warn "wire-openclaw-mlflow (rhoai) failed"
elif [[ "${ENABLE_OPENCLAW_MLFLOW:-0}" == "1" ]]; then
  step "Wire OpenClaw gateway → MLflow (before skill staging — restart wipes EmptyDir)"
  chmod +x "$ROOT/scripts/wire-openclaw-mlflow.sh"
  OPENCLAW_NS="$OPENCLAW_NS" "$ROOT/scripts/wire-openclaw-mlflow.sh" || \
    warn "wire-openclaw-mlflow failed — traces may be unavailable until fixed"
fi

# Lab kustomize apply resets fields — reconcile restores kit-owned wiring in lab config.yaml.
if [[ "${SKIP_RECONCILE:-0}" != "1" ]] && [[ -x "$ROOT/scripts/reconcile-openclaw-demo-wiring.sh" ]]; then
  step "Reconcile kit-owned OpenClaw wiring (baseUrl, hooks, slack)"
  RECONCILE_IN_SEED=1 SKIP_METRICS="${SKIP_METRICS:-1}" \
    SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}" "$ROOT/scripts/reconcile-openclaw-demo-wiring.sh" apply || \
    warn "reconcile-openclaw-demo-wiring failed — run manually before demo"
elif [[ "${SKIP_TRUSTYAI_GUARD:-0}" != "1" ]] \
    && "$KUBECTL" -n "${GUARDRAILS_NS:-netobserv-guardrails}" get guardrailsorchestrator guardrails-orchestrator >/dev/null 2>&1; then
  step "Re-wire OpenClaw LLM → TrustyAI guard proxy (seed resets lab baseUrl)"
  chmod +x "$ROOT/scripts/wire-openclaw-trustyai-guardrails.sh"
  OPENCLAW_NS="$OPENCLAW_NS" "$ROOT/scripts/wire-openclaw-trustyai-guardrails.sh" all || \
    warn "wire-openclaw-trustyai-guardrails failed — run before security demo"
fi

resolve_openclaw_pod || { echo "no ready openclaw Deployment pod" >&2; exit 1; }
step "Using pod $POD"

if [[ -z "$API_SERVER" ]]; then
  if [[ "$API_SERVER_MODE" == "external" ]]; then
    API_SERVER="$("$KUBECTL" config view --minify -o jsonpath='{.clusters[0].cluster.server}')"
  else
    API_SERVER="https://kubernetes.default.svc"
  fi
fi
step "API server for sandbox kubeconfig: $API_SERVER"

TOKEN="$("$KUBECTL" -n "$OPENCLAW_NS" create token openclaw-netobserv --duration=24h)"
CA_B64="$("$KUBECTL" config view --raw --minify -o jsonpath='{.clusters[0].cluster.certificate-authority-data}' 2>/dev/null || true)"
if [[ -z "$CA_B64" ]]; then
  CA_FILE="$("$KUBECTL" config view --minify -o jsonpath='{.clusters[0].cluster.certificate-authority}')"
  if [[ -n "$CA_FILE" && -f "$CA_FILE" ]]; then
    CA_B64="$(base64 <"$CA_FILE" | tr -d '\n')"
  fi
fi
# Prefer the in-cluster root CA when targeting kubernetes.default.svc
if [[ "$API_SERVER" == *"kubernetes.default.svc"* ]]; then
  ROOT_CA="$("$KUBECTL" -n "$OPENCLAW_NS" get configmap kube-root-ca.crt -o jsonpath='{.data.ca\.crt}' 2>/dev/null || true)"
  if [[ -n "$ROOT_CA" ]]; then
    CA_B64="$(printf '%s' "$ROOT_CA" | base64 | tr -d '\n')"
  fi
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export _SEED_API_SERVER="$API_SERVER" _SEED_TOKEN="$TOKEN" _SEED_CA_B64="$CA_B64" _SEED_KUBE="$TMP/kubeconfig"
python3 - <<'PY'
import json, os
cluster = {"server": os.environ["_SEED_API_SERVER"]}
ca = os.environ.get("_SEED_CA_B64") or ""
if ca:
    cluster["certificate-authority-data"] = ca
cfg = {
  "apiVersion": "v1",
  "kind": "Config",
  "clusters": [{"name": "cluster", "cluster": cluster}],
  "users": [{"name": "openclaw-netobserv", "user": {"token": os.environ["_SEED_TOKEN"]}}],
  "contexts": [{
    "name": "openclaw-netobserv",
    "context": {"cluster": "cluster", "user": "openclaw-netobserv", "namespace": "default"},
  }],
  "current-context": "openclaw-netobserv",
}
open(os.environ["_SEED_KUBE"], "w").write(json.dumps(cfg, indent=2))
print("kubeconfig written")
PY

step "Deploy in-cluster flow capture proxy (agent-driven NetObserv capture)"
# oc-netobserv needs the bastion client-cert kubeconfig (not the SA token JSON).
CAPTURE_KUBECONFIG="$TMP/capture-admin-kube.json"
"$KUBECTL" config view --minify --flatten -o json >"$CAPTURE_KUBECONFIG"
"$KUBECTL" -n "$OPENCLAW_NS" create secret generic netobserv-capture-kubeconfig \
  --from-file=config="$CAPTURE_KUBECONFIG" \
  --dry-run=client -o yaml | "$KUBECTL" apply -f -
"$KUBECTL" -n "$OPENCLAW_NS" create configmap netobserv-capture-proxy-scripts \
  --from-file=netobserv-cluster-capture.py="$ROOT/openclaw-skills/netobserv-heal/scripts/netobserv-cluster-capture.py" \
  --from-file=netobserv-cluster-heal.py="$ROOT/openclaw-skills/netobserv-heal/scripts/netobserv-cluster-heal.py" \
  --from-file=netobserv-capture-proxy.py="$ROOT/openclaw-skills/netobserv-heal/scripts/netobserv-capture-proxy.py" \
  --dry-run=client -o yaml | "$KUBECTL" apply -f -
"$KUBECTL" apply -f "$ROOT/openclaw-skills/manifests/netobserv-capture-proxy.yaml"
"$KUBECTL" -n "$OPENCLAW_NS" rollout restart deploy/netobserv-capture-proxy 2>/dev/null || true
"$KUBECTL" -n "$OPENCLAW_NS" rollout status deploy/netobserv-capture-proxy --timeout=180s || \
  warn "capture proxy rollout not ready yet"

step "Flatten polluted workspace + stage files"
resolve_openclaw_pod || { echo "openclaw pod disappeared before staging" >&2; exit 1; }
step "Staging into pod $POD"
"$KUBECTL" -n "$OPENCLAW_NS" exec "$POD" -- sh -lc '
  set -euo pipefail
  WS=/opt/openclaw/workspace
  for d in "$WS"/openclaw-openshell-upload-*; do
    [ -d "$d" ] || continue
    echo "flattening $d"
    cp -a "$d"/. "$WS"/ 2>/dev/null || true
    rm -rf "$d"
  done
  mkdir -p "$WS/skills" "$WS/evidence" "$WS/.kube" /opt/openclaw/kube
  rm -f "$WS/evidence/latest.json"
'

for skill in netobserv-investigate netobserv-evidence netobserv-heal netobserv-live-flows ansible-automation; do
  "$KUBECTL" -n "$OPENCLAW_NS" exec "$POD" -- rm -rf "/opt/openclaw/workspace/skills/${skill}"
  "$KUBECTL" -n "$OPENCLAW_NS" cp "$ROOT/openclaw-skills/${skill}" "${POD}:/opt/openclaw/workspace/skills/${skill}"
done

"$KUBECTL" -n "$OPENCLAW_NS" cp \
  "$ROOT/openclaw-skills/analyze-evidence.sh" \
  "${POD}:/opt/openclaw/workspace/analyze-evidence.sh"
"$KUBECTL" -n "$OPENCLAW_NS" cp \
  "$ROOT/openclaw-skills/heal-netobserv.sh" \
  "${POD}:/opt/openclaw/workspace/heal-netobserv.sh"
"$KUBECTL" -n "$OPENCLAW_NS" cp \
  "$ROOT/openclaw-skills/sync-evidence.sh" \
  "${POD}:/opt/openclaw/workspace/sync-evidence.sh"
"$KUBECTL" -n "$OPENCLAW_NS" cp \
  "$ROOT/openclaw-skills/netobserv-evidence/scripts/summarize-evidence.py" \
  "${POD}:/opt/openclaw/workspace/summarize-evidence.py"
"$KUBECTL" -n "$OPENCLAW_NS" cp \
  "$ROOT/openclaw-skills/TOOLS.netobserv.md" \
  "${POD}:/opt/openclaw/workspace/TOOLS.md"

"$KUBECTL" -n "$OPENCLAW_NS" cp "$TMP/kubeconfig" "${POD}:/opt/openclaw/kube/kubeconfig"
"$KUBECTL" -n "$OPENCLAW_NS" cp "$TMP/kubeconfig" "${POD}:/opt/openclaw/workspace/.kube/config"

# Refresh NetObserv rules in AGENTS.md (always rewrite the marked block)
AGENTS_PATCH="$TMP/patch-agents-netobserv.py"
cat >"$AGENTS_PATCH" <<'PY'
from pathlib import Path
ag = Path("/opt/openclaw/workspace/AGENTS.md")
mark = "## NetObserv demo rules"
ver = "netobserv-rules-v20"
block = f"""# AGENTS.md - Your Workspace

{mark}
<!-- {ver} -->

For Network Observability in this lab, treat these as direct orders:

- **Symptom reports** ("todo app is slow", "users complain", "DB timeouts"): use
  **netobserv-investigate** — triage **todo-demo** workloads (netobserv-openshift) → **cluster-wide
  checks on BOTH MCP servers** (`openshift_cluster_health` + openshift-mcp `events_list` or
  `netobserv_get_flow_metrics`) → probe latency OR policy status → capture → analyze → ask
  permission to remediate. Do **not** skip capture on first symptom message.
- **Real-incident narrative:** User-facing replies must sound like production SRE triage. Never
  mention loadgen, loadgen-heavy, synthetic traffic, traffic generators, todo-client consumer
  workloads, demo/lab plumbing, or "we injected load to simulate errors." Triage with `k8s_*` in
  **todo-demo only** during investigation — not todo-client.
- **Never** use the `read` tool on `summarize-evidence.py` or guess evidence paths — analysis
  is MCP `netobserv_analyze_evidence` only (or exec analyze-evidence.sh as fallback).
- If the user asks whether the **OpenShift cluster** is healthy (operators, nodes, ClusterVersion):
  call **`netobserv-openshift`** MCP `openshift_cluster_health` **and** one read-only
  **`openshift-mcp`** tool (e.g. `events_list` or `netobserv_get_flow_metrics`). Do **not**
  answer from `netobserv_status` alone or from app-path probes alone.
- `netobserv_status` / `netobserv_probe_latency` are **todo→DB demo path only**. Never say
  "the OpenShift cluster is healthy" based only on those tools.
- **Evidence analysis is diagnosis-only:** After capture, call MCP `netobserv_analyze_evidence`.
  Optional live context during analysis: **at most one** of `k8s_list_pods` or
  `k8s_list_deployments` in todo-demo — **not** k8s_recent_events, netobserv_status,
  netobserv_list_chaos, or openshift_cluster_health during evidence turns.
- **Evidence reply format:** Impact → Network (NetObserv) → Application → Likely cause class →
  **Suggested follow-up (diagnostic only)** → Confidence. Quote script numbers only.
  Use **Suggested follow-up** — not "Next Steps". Do **not** name MCP tools, heal commands, chaos/batch
  jobs, or fault-injection tooling. Investigation turn may ask once for remediation permission;
  evidence-only turns do **not** ask "would you like to heal?"
- **No spoilers:** Do not mention Kraken, chaos Jobs, loadgen, loadgen-heavy, synthetic traffic,
  network shaping cleanup, ansible_launch_job, or netobserv_list_chaos unless the **user** named
  remediation first.
- **Policy / microsegmentation incidents:** When evidence shows NetworkPolicy ingress does not admit
  todo (connectivity denial, policy-related drops, low RTT with JDBC failures), classify as
  **connectivity failure — not latency**. During evidence turns do **not** call or suggest
  automation. When the user **explicitly** confirms fix/restore in a **new** message after
  such a diagnosis, call **ansible-automation** MCP `ansible_launch_job` with
  job_template=`netobserv-restore-policy` and confirmed=true.
- For **live** NetObserv RTT/flows: **netobserv-live-flows** — `netobserv_get_flow_metrics` first
  (`dataSource=prom`); `netobserv_list_flows` only as narrow fallback (Loki timeouts).
- **Security demo:** Rogue exec/API prompts must fail in sandbox; investigation uses MCP read tools;
  automation requires `confirmed=true` on **ansible-automation** MCP. When **TrustyAI Guardrails** is wired, OpenClaw LLM
  traffic goes through the gateway preset **netobserv-sre** (regex + HAP + prompt injection). See
  **SECURITY-DEMO.md** and **docs/TRUSTYAI-GUARDRAILS-GUIDE.md**.
- If the user **explicitly** confirms heal for a **latency** incident in a **new** message
  (/netobserv-heal, "yes heal" after RTT/path degradation): call **ansible-automation** MCP
  `ansible_launch_job` with job_template=`netobserv-heal-db-path` and confirmed=true.
- NEVER run clawhub, NEVER ask the user to paste find/bash one-liners or kubeconfigs.
- Quote exact tool output; never invent 403/connection diagnoses.
- OpenShell nests files under /sandbox/openclaw-openshell-upload-* — helpers already handle that.

---

"""
body = ag.read_text() if ag.exists() else ""
lines = body.splitlines(True)
out = []
skip = False
for line in lines:
    if line.startswith(mark):
        skip = True
        continue
    if skip:
        if line.strip() == "---":
            skip = False
        continue
    if line.startswith("# AGENTS.md"):
        continue
    out.append(line)
ag.write_text(block + "".join(out).lstrip("\n"))
print(f"patched AGENTS.md ({ver})")
PY
"$KUBECTL" -n "$OPENCLAW_NS" cp "$AGENTS_PATCH" "${POD}:/tmp/patch-agents-netobserv.py"
"$KUBECTL" -n "$OPENCLAW_NS" exec "$POD" -- python3 /tmp/patch-agents-netobserv.py

"$KUBECTL" -n "$OPENCLAW_NS" exec "$POD" -- sh -lc '
  set -euo pipefail
  chmod +x /opt/openclaw/workspace/analyze-evidence.sh \
           /opt/openclaw/workspace/heal-netobserv.sh \
           /opt/openclaw/workspace/sync-evidence.sh \
           /opt/openclaw/workspace/skills/netobserv-*/scripts/* 2>/dev/null || true
  test -f /opt/openclaw/workspace/summarize-evidence.py
  test -f /opt/openclaw/workspace/analyze-evidence.sh
  test -f /opt/openclaw/workspace/heal-netobserv.sh
  test -f /opt/openclaw/workspace/.kube/config
  test -f /opt/openclaw/workspace/TOOLS.md
  if [[ ! -f /opt/openclaw/workspace/evidence/latest.json ]]; then
    echo "[info] evidence/latest.json will be populated by agent capture (netobserv-investigate)" >&2
  fi
  HOME=/opt/openclaw OPENCLAW_CONFIG_PATH=/opt/openclaw/config/openclaw.json \
    node /app/openclaw.mjs skills list | grep netobserv
  HOME=/opt/openclaw OPENCLAW_CONFIG_PATH=/opt/openclaw/config/openclaw.json \
    node /app/openclaw.mjs config get agents.defaults.skills
'

step "Patch OpenShell upload flatten (fixes SKILL.md ENOENT) + reload gateway"
chmod +x "$ROOT/scripts/patch-openshell-upload-flatten.sh"
resolve_openclaw_pod || { echo "openclaw pod disappeared before upload patch" >&2; exit 1; }
OPENCLAW_NS="$OPENCLAW_NS" "$ROOT/scripts/patch-openshell-upload-flatten.sh" || \
  warn "openshell upload flatten patch failed (skill reads may ENOENT until fixed)"
# Reload main container only so EmptyDir (skills + patched plugin) survives.
# Full pod recreate would wipe emptyDir and require another seed.
if "$KUBECTL" -n "$OPENCLAW_NS" exec "$POD" -c openclaw -- sh -lc '
  f=$(ls /opt/openclaw/config/npm/projects/*/node_modules/@openclaw/openshell-sandbox/dist/index.js 2>/dev/null | head -1)
  grep -q NETOBSERV_FLATTEN_UPLOAD "$f"
'; then
  step "Restart openclaw container to load upload-flatten patch (keep EmptyDir)"
  "$KUBECTL" -n "$OPENCLAW_NS" exec "$POD" -c openclaw -- sh -lc 'kill 1' >/dev/null 2>&1 || true
  # Wait until the main container is Ready again (kill 1 briefly drops readiness).
  for _ in $(seq 1 90); do
    ready="$("$KUBECTL" -n "$OPENCLAW_NS" get pod "$POD" -o jsonpath='{.status.containerStatuses[?(@.name=="openclaw")].ready}' 2>/dev/null || true)"
    phase="$("$KUBECTL" -n "$OPENCLAW_NS" get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    if [[ "$phase" == "Running" && "$ready" == "true" ]]; then
      # Brief settle so exec/API is usable again
      sleep 3
      if "$KUBECTL" -n "$OPENCLAW_NS" exec "$POD" -c openclaw -- true >/dev/null 2>&1; then
        break
      fi
    fi
    sleep 2
  done
  resolve_openclaw_pod || warn "openclaw not ready after container reload"
  ok "Gateway reloaded with upload-flatten patch on $POD"
else
  warn "upload-flatten marker missing — gateway not reloaded"
fi

step "Recreate sandboxes"
"$KUBECTL" -n "$OPENCLAW_NS" exec "$POD" -- sh -lc '
  HOME=/opt/openclaw OPENCLAW_CONFIG_PATH=/opt/openclaw/config/openclaw.json
  node /app/openclaw.mjs sandbox recreate --all --force 2>&1 || true
' || true
"$KUBECTL" -n "$OPENSHELL_NS" get sandboxes -o name 2>/dev/null | \
  xargs -r "$KUBECTL" -n "$OPENSHELL_NS" delete --wait=false 2>/dev/null || true

# Best-effort immediate flatten for any sandbox that comes up during seed
step "Flatten nested OpenShell uploads in live sandboxes (best-effort)"
for _ in $(seq 1 12); do
  pods="$("$KUBECTL" -n "$OPENSHELL_NS" get pods -o name 2>/dev/null | grep openclaw-agent || true)"
  [[ -n "$pods" ]] || { sleep 2; continue; }
  while IFS= read -r p; do
    [[ -n "$p" ]] || continue
    "$KUBECTL" -n "$OPENSHELL_NS" exec "${p#pod/}" -c agent -- sh -c '
      flatten_root() {
        root="$1"
        for d in "$root"/openclaw-openshell-upload-*; do
          [ -d "$d" ] || continue
          cp -a "$d"/. "$root"/ 2>/dev/null || true
          rm -rf "$d"
        done
      }
      flatten_root /sandbox
      mkdir -p /sandbox/.openclaw/sandbox-skills
      flatten_root /sandbox/.openclaw/sandbox-skills
      test -f /sandbox/.openclaw/sandbox-skills/skills/netobserv-evidence/SKILL.md
    ' 2>/dev/null && ok "flattened ${p#pod/}" && break 2
  done <<<"$pods"
  sleep 2
done || warn "no live sandbox to flatten yet (upload patch + flatten-loop cover first /new)"

if [[ "${ENABLE_OPENCLAW_OTEL:-0}" == "1" ]]; then
  step "Phase D — deploy OTel collector + wire gateway export"
  chmod +x "$ROOT/scripts/wire-openclaw-otel.sh"
  OPENCLAW_NS="$OPENCLAW_NS" "$ROOT/scripts/wire-openclaw-otel.sh" all || \
    warn "wire-openclaw-otel failed — run ./scripts/install-openclaw-otel-grafana.sh all"
fi

ok "Seed complete on $POD"
cat <<'EOF'

Next in Control UI:
  1. /new  →  send a message (sandbox pod spawns on first turn — not on /new alone)
  2. Investigate:  "Users say the todo app is really slow talking to the database."
     (dual MCP triage → capture ~60s → netobserv_analyze_evidence)
  3. /new  →  remediate:
     Scenario A:  "Yes, heal the DB path"
     Scenario B:  "Yes, restore the network policy"

Presenter docs (print):
  PRESENTER-RUNBOOK.md      — full timeline
  SECURITY-DEMO.md          — optional sandbox + rogue prompts (~5 min)
  SECURITY-PROOF-GUIDE.md   — customer pre-read (architecture + proof commands)

Bastion checks:
  ./scripts/netobserv-e2e-openclaw-test.sh demo-a-fast   # inject + Grafana metrics sync (demo-a for cold start + seed)
  ./scripts/pre-demo-sanity.sh quick                # optional extra probes
  ./scripts/openshell-sandbox-proof.sh wait         # after first UI message
  ./scripts/netobserv-e2e-openclaw-test.sh status
  ./scripts/deploy-platform-merge.sh status     # OpenShift AI + RHCL + Grafana (if installed)
  ./scripts/netobserv-e2e-openclaw-test.sh mlflow-check
EOF
if [[ "${ENABLE_OPENCLAW_MLFLOW:-0}" == "1" || "${ENABLE_RHOAI_PLATFORM:-0}" == "1" ]]; then
  : # mlflow-check already listed above
elif oc get deploy/mlflow -n redhat-ods-applications >/dev/null 2>&1; then
  cat <<'EOF'
  ./scripts/netobserv-e2e-openclaw-test.sh mlflow-check   # RHOAI MLflow detected
EOF
fi
if [[ "${ENABLE_OPENCLAW_MLFLOW:-0}" == "1" ]]; then
  echo "  (legacy standalone MLflow enabled)"
fi
cat <<'EOF'

Notes:
  - Inject fault BEFORE UI:  ./scripts/netobserv-e2e-openclaw-test.sh demo-a
  - Agent heal ≠ bastion cleanup (Scenario A):  ./scripts/netobserv-krkn-fault.sh restore
  - Full cleanup:  ./scripts/netobserv-e2e-openclaw-test.sh restore
EOF
