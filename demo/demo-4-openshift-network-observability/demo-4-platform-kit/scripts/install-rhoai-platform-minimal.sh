#!/usr/bin/env bash
# Install minimal Red Hat OpenShift AI: dashboard + MLflow (no KServe / workbenches).
# Usage: ./scripts/install-rhoai-platform-minimal.sh [status|install|wire-prep]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFESTS="$ROOT/manifests/platform-merge"
KUBECTL="$(command -v oc || command -v kubectl)"
CMD="${1:-install}"
RHOAI_CHANNEL="${RHOAI_CHANNEL:-stable-3.4}"
RHOAI_CSV="${RHOAI_CSV:-rhods-operator.3.4.3}"
RHOAI_NS="${RHOAI_NS:-redhat-ods-applications}"
MLFLOW_EXP="${MLFLOW_EXPERIMENT_NAME:-openclaw-netobserv}"
WAIT_RHOAI_SEC="${WAIT_RHOAI_SEC:-2400}"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -n "$KUBECTL" ]] || die "oc/kubectl required"

wait_rhods_csv() {
  step "Wait rhods-operator CSV in redhat-ods-operator (timeout ${1:-1200}s)"
  local timeout="${1:-1200}"
  local end=$((SECONDS + timeout))
  while (( SECONDS < end )); do
    local csv phase
    csv="$("$KUBECTL" get csv -n redhat-ods-operator -o json 2>/dev/null | python3 -c "
import json,sys
data=json.load(sys.stdin)
for item in data.get('items',[]):
    name=item.get('metadata',{}).get('name','')
    if name.startswith('rhods-operator.'):
        print(name)
        break
" 2>/dev/null || true)"
    if [[ -n "$csv" ]]; then
      phase="$("$KUBECTL" get csv "$csv" -n redhat-ods-operator -o jsonpath='{.status.phase}' 2>/dev/null || true)"
      if [[ "$phase" == "Succeeded" ]]; then
        ok "CSV $csv Succeeded"
        return 0
      fi
      printf '  waiting csv=%s phase=%s\n' "$csv" "${phase:-Pending}"
    else
      printf '  waiting for rhods-operator CSV...\n'
    fi
    sleep 15
  done
  warn "rhods-operator CSV not Succeeded within ${timeout}s"
  "$KUBECTL" get csv -n redhat-ods-operator 2>/dev/null || true
  return 1
}

wait_dsc_ready() {
  step "Wait DataScienceCluster default-dsc Ready (timeout ${WAIT_RHOAI_SEC}s)"
  local end=$((SECONDS + WAIT_RHOAI_SEC))
  while (( SECONDS < end )); do
    local phase ready
    phase="$("$KUBECTL" get dsc default-dsc -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    ready="$("$KUBECTL" get dsc default-dsc -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)"
    if [[ "$phase" == "Ready" || "$ready" == "True" ]]; then
      ok "default-dsc Ready"
      return 0
    fi
    printf '  dsc phase=%s ready=%s (%ds)\n' "${phase:-unknown}" "${ready:-unknown}" "$SECONDS"
    sleep 20
  done
  warn "default-dsc not Ready within ${WAIT_RHOAI_SEC}s — check: oc get dsc,dsci -o yaml"
  "$KUBECTL" get dsc default-dsc -o yaml 2>/dev/null | tail -40 || true
  return 1
}

wait_mlflow_deploy() {
  step "Wait RHOAI MLflow deployment"
  "$KUBECTL" -n "$RHOAI_NS" wait --for=condition=Available deploy/mlflow --timeout=600s 2>/dev/null || \
    "$KUBECTL" -n "$RHOAI_NS" rollout status deploy/mlflow --timeout=600s
  ok "MLflow deployment available"
}

print_status() {
  step "Platform merge status"
  "$KUBECTL" get sub,csv -n redhat-ods-operator 2>/dev/null || true
  "$KUBECTL" get dsci,dsc 2>/dev/null || true
  "$KUBECTL" get mlflow -n "$RHOAI_NS" 2>/dev/null || true
  "$KUBECTL" get deploy,pods,route -n "$RHOAI_NS" 2>/dev/null | grep -E 'mlflow|dashboard|rh-ai|NAME' || true
  local dash mlflow_rh
  dash="$("$KUBECTL" -n "$RHOAI_NS" get route rhods-dashboard -o jsonpath='{.spec.host}' 2>/dev/null || true)"
  mlflow_rh="$("$KUBECTL" -n "$RHOAI_NS" get route -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.host}{"\n"}{end}' 2>/dev/null | grep -i mlflow || true)"
  [[ -n "$dash" ]] && ok "OpenShift AI dashboard: https://${dash}/"
  [[ -n "$mlflow_rh" ]] && printf '%s\n' "$mlflow_rh"
}

install_rhoai() {
  step "Apply RHOAI operator subscription (channel $RHOAI_CHANNEL)"
  "$KUBECTL" apply -f "$MANIFESTS/00-namespace-rhods-operator.yaml"
  "$KUBECTL" apply -f "$MANIFESTS/01-rhods-operatorgroup.yaml"
  if [[ "$RHOAI_CHANNEL" != "stable-3.4" ]]; then
    sed "s/channel: stable-3.4/channel: ${RHOAI_CHANNEL}/" "$MANIFESTS/02-rhods-subscription.yaml" | "$KUBECTL" apply -f -
  else
    "$KUBECTL" apply -f "$MANIFESTS/02-rhods-subscription.yaml"
  fi

  wait_rhods_csv 1200 || die "rhods-operator CSV failed"

  step "Apply DSCInitialization + minimal DataScienceCluster"
  "$KUBECTL" apply -f "$MANIFESTS/03-dscinitialization.yaml"
  sleep 5
  "$KUBECTL" apply -f "$MANIFESTS/04-dsc-minimal.yaml"
  wait_dsc_ready || warn "DSC not fully Ready — continuing if MLflow operator is up"

  step "Workspace namespace + MLflow CR + RBAC"
  "$KUBECTL" apply -f "$MANIFESTS/06-namespace-netobserv-demo.yaml"
  "$KUBECTL" apply -f "$MANIFESTS/05-mlflow-cr.yaml"
  "$KUBECTL" apply -f "$MANIFESTS/07-openclaw-mlflow-rbac.yaml"

  wait_mlflow_deploy

  step "Ensure MLflow experiment '$MLFLOW_EXP'"
  "$KUBECTL" -n "$RHOAI_NS" exec deploy/mlflow -- python3 -c "
import mlflow, os
uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
mlflow.set_tracking_uri(uri)
exp = mlflow.set_experiment('${MLFLOW_EXP}')
print(f'experiment_id={exp.experiment_id} name=${MLFLOW_EXP}')
" 2>/dev/null || warn "Could not create experiment inside mlflow pod — wire step will retry"

  print_status
  cat <<EOF

RHOAI platform core installed.
  OpenShift AI dashboard → Applications → MLflow
  Experiment: ${MLFLOW_EXP}

Next:
  MLFLOW_BACKEND=rhoai ./scripts/wire-openclaw-mlflow.sh
  # or full merge:
  ./scripts/deploy-platform-merge.sh wire

LiteMaaS Qwen unchanged — OpenClaw still uses external LLM.
EOF
}

case "$CMD" in
  status) print_status ;;
  install) install_rhoai ;;
  wire-prep)
    "$KUBECTL" apply -f "$MANIFESTS/07-openclaw-mlflow-rbac.yaml"
    ok "RBAC applied"
    ;;
  *)
    die "usage: $0 [status|install|wire-prep]"
    ;;
esac
