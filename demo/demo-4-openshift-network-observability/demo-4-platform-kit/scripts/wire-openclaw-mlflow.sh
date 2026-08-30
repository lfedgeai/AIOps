#!/usr/bin/env bash
# Wire OpenClaw + netobserv-mcp to MLflow (standalone or RHOAI).
#
# MLFLOW_BACKEND:
#   standalone (default) — openclaw/mlflow HTTP :5000
#   rhoai                — redhat-ods-applications MLflow HTTPS + kubernetes auth
#
# Gateway @mlflow/mlflow-openclaw plugin is broken on OpenClaw 2026.6.11;
# MCP tool Runs (netobserv-mcp) provide the audit trail.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
GUARDRAILS_NS="${GUARDRAILS_NS:-netobserv-guardrails}"
RHOAI_NS="${RHOAI_NS:-redhat-ods-applications}"
KUBECTL="$(command -v oc || command -v kubectl)"
MLFLOW_BACKEND="${MLFLOW_BACKEND:-standalone}"
MLFLOW_EXP="${MLFLOW_EXPERIMENT_NAME:-openclaw-netobserv}"
MLFLOW_WORKSPACE="${MLFLOW_WORKSPACE:-openclaw}"
MLFLOW_REMOVE_STANDALONE="${MLFLOW_REMOVE_STANDALONE:-0}"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
step() { printf '\n%s==>%s %s\n' "$c_green" "$c_reset" "$*"; }

[[ -n "$KUBECTL" ]] || { echo "oc/kubectl required" >&2; exit 1; }

apply_mlflow_tracing_configmap() {
  local ns="$1"
  "$KUBECTL" -n "$ns" create configmap netobserv-mlflow-tracing \
    --from-file=mlflow_tracing.py="$ROOT/openclaw-skills/mcp-server/mlflow_tracing.py" \
    --dry-run=client -o yaml | "$KUBECTL" apply -f -
}

sync_mcp_scripts_configmap() {
  # MCP pod mounts /opt/mcp from netobserv-mcp-scripts (not netobserv-mlflow-tracing).
  "$KUBECTL" -n "$OPENCLAW_NS" create configmap netobserv-mcp-scripts \
    --from-file=server.py="$ROOT/openclaw-skills/mcp-server/server.py" \
    --from-file=mlflow_tracing.py="$ROOT/openclaw-skills/mcp-server/mlflow_tracing.py" \
    --from-file=requirements.txt="$ROOT/openclaw-skills/mcp-server/requirements.txt" \
    --from-file=netobserv-cluster-heal.py="$ROOT/openclaw-skills/netobserv-heal/scripts/netobserv-cluster-heal.py" \
    --from-file=summarize-evidence.py="$ROOT/openclaw-skills/netobserv-evidence/scripts/summarize-evidence.py" \
    --dry-run=client -o yaml | "$KUBECTL" apply -f -
}

step "Publish shared mlflow_tracing.py ConfigMap"
apply_mlflow_tracing_configmap "$OPENCLAW_NS"
if "$KUBECTL" get ns "$GUARDRAILS_NS" >/dev/null 2>&1; then
  apply_mlflow_tracing_configmap "$GUARDRAILS_NS"
else
  warn "Namespace ${GUARDRAILS_NS} absent — skip guardrails ConfigMap (install guardrails phase later)"
fi
sync_mcp_scripts_configmap
ok "netobserv-mlflow-tracing + netobserv-mcp-scripts mlflow_tracing.py synced"

# Auto-detect RHOAI when standalone mlflow in openclaw ns is absent.
if [[ "${MLFLOW_BACKEND}" == "standalone" && "${ENABLE_OPENCLAW_MLFLOW:-0}" != "1" ]]; then
  if "$KUBECTL" -n "$RHOAI_NS" get deploy/mlflow >/dev/null 2>&1; then
    MLFLOW_BACKEND=rhoai
    step "Auto-detected RHOAI MLflow in ${RHOAI_NS}"
  fi
fi

case "$MLFLOW_BACKEND" in
  rhoai)
    MLFLOW_URI="${MLFLOW_TRACKING_URI:-https://mlflow.${RHOAI_NS}.svc.cluster.local:8443}"
    MLFLOW_DEPLOY_NS="$RHOAI_NS"
    MLFLOW_DEPLOY_NAME="mlflow"
    MLFLOW_AUTH_ENV=(
      MLFLOW_TRACKING_INSECURE_TLS=true
      MLFLOW_WORKSPACE="$MLFLOW_WORKSPACE"
      MLFLOW_TRACKING_AUTH=kubernetes-namespaced
    )
    ;;
  standalone|*)
    MLFLOW_URI="${MLFLOW_TRACKING_URI:-http://mlflow.${OPENCLAW_NS}.svc.cluster.local:5000}"
    MLFLOW_DEPLOY_NS="$OPENCLAW_NS"
    MLFLOW_DEPLOY_NAME="mlflow"
    MLFLOW_AUTH_ENV=()
    ;;
esac

step "Backend=$MLFLOW_BACKEND URI=$MLFLOW_URI experiment=$MLFLOW_EXP"

if [[ "$MLFLOW_BACKEND" == "rhoai" ]]; then
  step "Apply openclaw → RHOAI MLflow RBAC"
  "$KUBECTL" apply -f "$ROOT/manifests/platform-merge/07-openclaw-mlflow-rbac.yaml"
fi

step "Resolve MLflow experiment id for '$MLFLOW_EXP'"
EXP_ID=""
if [[ "$MLFLOW_BACKEND" == "rhoai" ]]; then
  MLFLOW_TOKEN="$("$KUBECTL" -n "$OPENCLAW_NS" create token openclaw-netobserv --duration=30m 2>/dev/null || true)"
  if [[ -n "$MLFLOW_TOKEN" ]]; then
    EXP_ID="$("$KUBECTL" -n "$OPENCLAW_NS" run mlflow-exp-bootstrap --rm -i --restart=Never \
      --image=registry.access.redhat.com/ubi9/python-312:latest \
      --overrides='{"spec":{"serviceAccountName":"openclaw-netobserv"}}' \
      --command -- bash -lc "
pip install -q 'mlflow>=3.1,<4' && python3 -c \"
import os, mlflow
os.environ['MLFLOW_TRACKING_URI']='${MLFLOW_URI}'
os.environ['MLFLOW_TRACKING_TOKEN']='${MLFLOW_TOKEN}'
os.environ['MLFLOW_TRACKING_INSECURE_TLS']='true'
os.environ['MLFLOW_WORKSPACE']='${MLFLOW_WORKSPACE}'
mlflow.set_tracking_uri('${MLFLOW_URI}')
if hasattr(mlflow, 'set_workspace'):
    mlflow.set_workspace('${MLFLOW_WORKSPACE}')
exp = mlflow.get_experiment_by_name('${MLFLOW_EXP}')
if exp is None:
    exp = mlflow.set_experiment('${MLFLOW_EXP}')
print(exp.experiment_id)
\"
" 2>/dev/null | grep -E '^[0-9a-f-]+$' | tail -1)" || true
  fi
else
  EXP_ID="$("$KUBECTL" -n "$MLFLOW_DEPLOY_NS" exec "deploy/${MLFLOW_DEPLOY_NAME}" -- python3 -c "
import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
exp = mlflow.get_experiment_by_name('${MLFLOW_EXP}')
if exp is None:
    exp = mlflow.set_experiment('${MLFLOW_EXP}')
    print(exp.experiment_id)
else:
    print(exp.experiment_id)
" 2>/dev/null)" || true
fi
[[ -n "$EXP_ID" ]] || warn "Could not resolve experiment id (first MCP tool call may create it)"

step "Set MLFLOW_* env on deploy/openclaw (future gateway plugin)"
OPENCLAW_ENV=(
  MLFLOW_TRACKING_URI="$MLFLOW_URI"
  MLFLOW_EXPERIMENT_NAME="$MLFLOW_EXP"
)
if [[ -n "$EXP_ID" ]]; then
  OPENCLAW_ENV+=(MLFLOW_EXPERIMENT_ID="$EXP_ID")
fi
if [[ ${#MLFLOW_AUTH_ENV[@]} -gt 0 ]]; then
  OPENCLAW_ENV+=("${MLFLOW_AUTH_ENV[@]}")
fi
"$KUBECTL" -n "$OPENCLAW_NS" set env deploy/openclaw "${OPENCLAW_ENV[@]}" 2>/dev/null || true

step "Wire netobserv-mcp → MLflow ($MLFLOW_BACKEND)"
MCP_ENV=(
  MLFLOW_TRACKING_URI="$MLFLOW_URI"
  MLFLOW_EXPERIMENT_NAME="$MLFLOW_EXP"
  MLFLOW_TRACE_CONTEXT_URL="http://netobserv-grafana-bridge.${OPENCLAW_NS}.svc.cluster.local:8080/mlflow/trace-context"
)
if [[ "$MLFLOW_BACKEND" == "rhoai" ]]; then
  MCP_ENV+=(MLFLOW_WORKSPACE="$MLFLOW_WORKSPACE")
fi
if [[ ${#MLFLOW_AUTH_ENV[@]} -gt 0 ]]; then
  MCP_ENV+=("${MLFLOW_AUTH_ENV[@]}")
fi
"$KUBECTL" -n "$OPENCLAW_NS" set env deploy/netobserv-mcp "${MCP_ENV[@]}"
"$KUBECTL" -n "$OPENCLAW_NS" rollout restart deploy/netobserv-mcp
"$KUBECTL" -n "$OPENCLAW_NS" rollout status deploy/netobserv-mcp --timeout=300s

step "MCP self-test trace"
MCP_POD="$("$KUBECTL" -n "$OPENCLAW_NS" get pods -l app.kubernetes.io/name=netobserv-mcp \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
if [[ -n "$MCP_POD" ]]; then
  MCP_TEST_ENV=(MLFLOW_TRACKING_URI="$MLFLOW_URI" MLFLOW_EXPERIMENT_NAME="$MLFLOW_EXP")
  if [[ ${#MLFLOW_AUTH_ENV[@]} -gt 0 ]]; then
    MCP_TEST_ENV+=("${MLFLOW_AUTH_ENV[@]}")
  fi
  "$KUBECTL" -n "$OPENCLAW_NS" exec "$MCP_POD" -- env "${MCP_TEST_ENV[@]}" python3 -c "
import os, sys
sys.path.insert(0, '/opt/mcp')
from mlflow_tracing import init_mlflow, trace_mcp_tool
@trace_mcp_tool('wire_selftest')
def _t():
    return 'ok'
assert init_mlflow()
assert _t() == 'ok'
print('mcp_trace_selftest_ok')
" && ok "MCP → MLflow Traces write OK"
else
  warn "netobserv-mcp pod not found for self-test"
fi

step "Wire netobserv-grafana-bridge → MLflow Traces ($MLFLOW_BACKEND)"
BRIDGE_ENV=(
  MLFLOW_TRACKING_URI="$MLFLOW_URI"
  MLFLOW_EXPERIMENT_NAME="$MLFLOW_EXP"
  MLFLOW_TRACE_CONTEXT_TTL_SEC=3600
)
if [[ "$MLFLOW_BACKEND" == "rhoai" ]]; then
  BRIDGE_ENV+=(MLFLOW_WORKSPACE="$MLFLOW_WORKSPACE")
fi
if [[ ${#MLFLOW_AUTH_ENV[@]} -gt 0 ]]; then
  BRIDGE_ENV+=("${MLFLOW_AUTH_ENV[@]}")
fi
"$KUBECTL" -n "$OPENCLAW_NS" set env deploy/netobserv-grafana-bridge "${BRIDGE_ENV[@]}" 2>/dev/null || \
  warn "netobserv-grafana-bridge not deployed — run wire-openclaw-hooks.sh"
"$KUBECTL" -n "$OPENCLAW_NS" rollout restart deploy/netobserv-grafana-bridge 2>/dev/null || true
"$KUBECTL" -n "$OPENCLAW_NS" rollout status deploy/netobserv-grafana-bridge --timeout=180s 2>/dev/null || \
  warn "grafana-bridge rollout slow"

step "Wire netobserv-llm-guard-proxy → MLflow Traces ($MLFLOW_BACKEND)"
if "$KUBECTL" -n "$GUARDRAILS_NS" get deploy/netobserv-llm-guard-proxy >/dev/null 2>&1; then
  PROXY_ENV=(
    MLFLOW_TRACKING_URI="$MLFLOW_URI"
    MLFLOW_EXPERIMENT_NAME="$MLFLOW_EXP"
    MLFLOW_TRACE_CONTEXT_URL="http://netobserv-grafana-bridge.${OPENCLAW_NS}.svc.cluster.local:8080/mlflow/trace-context"
  )
  if [[ "$MLFLOW_BACKEND" == "rhoai" ]]; then
    PROXY_ENV+=(MLFLOW_WORKSPACE="$MLFLOW_WORKSPACE")
  fi
  if [[ ${#MLFLOW_AUTH_ENV[@]} -gt 0 ]]; then
    PROXY_ENV+=("${MLFLOW_AUTH_ENV[@]}")
  fi
  "$KUBECTL" -n "$GUARDRAILS_NS" set env deploy/netobserv-llm-guard-proxy "${PROXY_ENV[@]}"
  "$KUBECTL" -n "$GUARDRAILS_NS" rollout restart deploy/netobserv-llm-guard-proxy
  "$KUBECTL" -n "$GUARDRAILS_NS" rollout status deploy/netobserv-llm-guard-proxy --timeout=180s
  ok "LLM guard proxy wired for llm_call spans"
else
  warn "netobserv-llm-guard-proxy not deployed — run install-trustyai-guardrails.sh"
fi

if [[ "$MLFLOW_BACKEND" == "rhoai" && "$MLFLOW_REMOVE_STANDALONE" == "1" ]]; then
  step "Remove standalone MLflow in $OPENCLAW_NS (avoid duplicate UIs)"
  "$KUBECTL" -n "$OPENCLAW_NS" delete route mlflow-openclaw --ignore-not-found
  "$KUBECTL" -n "$OPENCLAW_NS" delete deploy,svc,pvc mlflow --ignore-not-found
  ok "Standalone mlflow removed from openclaw ns"
fi

DASH_HOST="$("$KUBECTL" -n "$RHOAI_NS" get route rhods-dashboard -o jsonpath='{.spec.host}' 2>/dev/null || true)"
RH_AI_HOST="$("$KUBECTL" -n "$RHOAI_NS" get route -o jsonpath='{range .items[?(@.metadata.name=~"rh-ai")].spec.host}{"\n"}{end}' 2>/dev/null | head -1 || true)"
STANDALONE_ROUTE="$("$KUBECTL" -n "$OPENCLAW_NS" get route mlflow-openclaw -o jsonpath='{.spec.host}' 2>/dev/null || true)"

ok "MLflow tracking URI: $MLFLOW_URI"
[[ -n "$EXP_ID" ]] && ok "MLflow experiment id: $EXP_ID"
[[ -n "$DASH_HOST" ]] && ok "OpenShift AI dashboard: https://${DASH_HOST}/"
[[ -n "$RH_AI_HOST" ]] && ok "rh-ai route: https://${RH_AI_HOST}/"
[[ -n "$STANDALONE_ROUTE" ]] && ok "Standalone MLflow UI: https://${STANDALONE_ROUTE}/"

cat <<EOF

Audit path: MLflow Traces in experiment '${MLFLOW_EXP}' (backend=${MLFLOW_BACKEND}).
  investigate_session (grafana-bridge) → llm_call (guard proxy) → tool.* (netobserv-mcp)
LiteMaaS Qwen unchanged — OpenClaw LLM config not modified.

Verify: ./scripts/netobserv-e2e-openclaw-test.sh mlflow-check
EOF
