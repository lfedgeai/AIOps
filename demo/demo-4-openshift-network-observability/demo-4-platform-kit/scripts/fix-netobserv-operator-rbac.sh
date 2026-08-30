#!/usr/bin/env bash
# Grant NetObserv operator SA access to Loki / netobserv Secrets (console plugin + pipeline).
# Required after fresh install and after cluster restart if FlowCollector shows WebConsoleError.
#
# Usage:
#   ./scripts/fix-netobserv-operator-rbac.sh
#   ./scripts/fix-netobserv-operator-rbac.sh status
#
# Env:
#   OPERATOR_NS=openshift-netobserv-operator
#   LOKI_NS=netobserv-loki
#   NETOBSERV_NS=netobserv
set -euo pipefail

OPERATOR_NS="${OPERATOR_NS:-openshift-netobserv-operator}"
LOKI_NS="${LOKI_NS:-netobserv-loki}"
NETOBSERV_NS="${NETOBSERV_NS:-netobserv}"
SA="${OPERATOR_NS}:netobserv-controller-manager"
CMD="${1:-apply}"

KUBECTL="$(command -v oc || command -v kubectl)"
[[ -n "$KUBECTL" ]] || { echo "oc/kubectl required" >&2; exit 1; }

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
step() { printf '%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }

wait_clusterrole() {
  local name="$1" timeout="${2:-120}" elapsed=0
  while (( elapsed < timeout )); do
    if "$KUBECTL" get clusterrole "$name" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
    (( elapsed += 5 ))
  done
  return 1
}

cmd_status() {
  step "NetObserv operator Secret RBAC"
  for cr in netobserv-secret-watcher netobserv-secret-creator; do
    if "$KUBECTL" get clusterrole "$cr" >/dev/null 2>&1; then
      ok "ClusterRole $cr present"
    else
      warn "ClusterRole $cr missing — install Network Observability operator first"
    fi
  done
  "$KUBECTL" get rolebinding secret-watcher -n "$LOKI_NS" >/dev/null 2>&1 \
    && ok "RoleBinding secret-watcher ($LOKI_NS)" \
    || warn "RoleBinding secret-watcher missing in $LOKI_NS"
  "$KUBECTL" get rolebinding secret-creator -n "$NETOBSERV_NS" >/dev/null 2>&1 \
    && ok "RoleBinding secret-creator ($NETOBSERV_NS)" \
    || warn "RoleBinding secret-creator missing in $NETOBSERV_NS"
  if "$KUBECTL" get flowcollector cluster >/dev/null 2>&1; then
    local ready degraded
    ready="$("$KUBECTL" get flowcollector cluster \
      -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo Unknown)"
    echo "FlowCollector Ready=$ready"
  fi
}

cmd_apply() {
  step "Ensure NetObserv operator can read Loki / netobserv Secrets"
  if ! wait_clusterrole netobserv-secret-watcher 180; then
    warn "ClusterRole netobserv-secret-watcher not found — operator may still be installing"
    return 1
  fi
  if ! wait_clusterrole netobserv-secret-creator 60; then
    warn "ClusterRole netobserv-secret-creator not found"
    return 1
  fi

  "$KUBECTL" apply -f - >/dev/null <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: secret-watcher
  namespace: ${LOKI_NS}
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: netobserv-secret-watcher
subjects:
  - kind: ServiceAccount
    name: netobserv-controller-manager
    namespace: ${OPERATOR_NS}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: secret-creator
  namespace: ${NETOBSERV_NS}
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: netobserv-secret-creator
subjects:
  - kind: ServiceAccount
    name: netobserv-controller-manager
    namespace: ${OPERATOR_NS}
EOF
  ok "RoleBindings secret-watcher ($LOKI_NS) + secret-creator ($NETOBSERV_NS)"
  ok "If FlowCollector was WebConsoleError, it should reconcile within ~1 min"
}

case "$CMD" in
  status) cmd_status ;;
  apply|""|all) cmd_apply ;;
  -h|--help|help)
    echo "Usage: $(basename "$0") [apply|status]"
    exit 0
    ;;
  *) echo "unknown command: $CMD (try apply|status)" >&2; exit 1 ;;
esac
