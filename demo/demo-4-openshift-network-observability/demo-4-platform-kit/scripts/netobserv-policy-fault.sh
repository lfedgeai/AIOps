#!/usr/bin/env bash
#
# netobserv-policy-fault.sh
# ---------------------------------------------------------------------------
# Microsegmentation / NetworkPolicy incident for the NetObserv todo demo.
#
# Pair with agent investigation in Control UI:
#   ./scripts/netobserv-e2e-openclaw-test.sh policy-all
#
# Restore policy (not Kraken/tc):
#   ./scripts/netobserv-policy-fault.sh restore
#
# Subcommands:
#   break     Apply broken DB NetworkPolicy (default MODE=wrong-label)
#   restore   Re-apply correct allow-db-from-todo-only policy
#   status    Show policy + quick todo→DB probe
#
# Modes (MODE env):
#   wrong-label  Ingress peer label app: todo → app: todo-wrong (default)
#   deny-all     Empty ingress on PostgreSQL — blocks todo and everyone
# ---------------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

APP_NS="${APP_NS:-todo-demo}"
CLIENT_NS="${CLIENT_NS:-todo-client}"
POLICY_NAME="${POLICY_NAME:-allow-db-from-todo-only}"
MODE="${MODE:-wrong-label}"   # wrong-label | deny-all
TARGET_PATH="${TARGET_PATH:-/api}"
STATE_DIR="${STATE_DIR:-/tmp/netobserv-policy-state}"

c_reset=$'\033[0m'; c_blue=$'\033[1;34m'; c_green=$'\033[1;32m'
c_yellow=$'\033[1;33m'; c_red=$'\033[1;31m'
step() { printf '%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
die()  { printf '%s[fail]%s %s\n' "$c_red" "$c_reset" "$*" >&2; exit 1; }

command -v oc >/dev/null 2>&1 || die "'oc' not found in PATH."
oc whoami >/dev/null 2>&1 || die "Not logged in. Run 'oc login ...' first."

apply_correct_policy() {
  oc apply -n "$APP_NS" -f - >/dev/null <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ${POLICY_NAME}
  namespace: ${APP_NS}
spec:
  podSelector:
    matchLabels:
      app: postgresql
  policyTypes: ["Ingress"]
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: todo
      ports:
        - protocol: TCP
          port: 5432
EOF
}

apply_broken_policy() {
  case "$MODE" in
    wrong-label)
      oc apply -n "$APP_NS" -f - >/dev/null <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ${POLICY_NAME}
  namespace: ${APP_NS}
spec:
  podSelector:
    matchLabels:
      app: postgresql
  policyTypes: ["Ingress"]
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: todo-wrong
      ports:
        - protocol: TCP
          port: 5432
EOF
      ;;
    deny-all)
      oc apply -n "$APP_NS" -f - >/dev/null <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ${POLICY_NAME}
  namespace: ${APP_NS}
spec:
  podSelector:
    matchLabels:
      app: postgresql
  policyTypes: ["Ingress"]
  ingress: []
EOF
      ;;
    *)
      die "MODE must be wrong-label or deny-all (got: $MODE)"
      ;;
  esac
}

cmd_break() {
  mkdir -p "$STATE_DIR"
  step "Backing up current NetworkPolicy (if present)"
  if oc get networkpolicy "$POLICY_NAME" -n "$APP_NS" >/dev/null 2>&1; then
    oc get networkpolicy "$POLICY_NAME" -n "$APP_NS" -o yaml >"$STATE_DIR/policy-backup.yaml"
    ok "Backup: $STATE_DIR/policy-backup.yaml"
  else
    warn "No existing $POLICY_NAME — will apply broken policy only"
  fi
  echo "$MODE" >"$STATE_DIR/mode"
  step "Applying broken DB NetworkPolicy (MODE=$MODE)"
  apply_broken_policy
  ok "PostgreSQL ingress policy is now misconfigured — expect todo→DB connectivity failure."
  warn "Report connectivity symptoms in Control UI — agent captures flows during investigation."
  warn "Restore with: ./scripts/netobserv-policy-fault.sh restore"
}

cmd_restore() {
  step "Restoring correct DB NetworkPolicy"
  if [[ -f "$STATE_DIR/policy-backup.yaml" ]]; then
    oc apply -f "$STATE_DIR/policy-backup.yaml" >/dev/null
    ok "Restored from $STATE_DIR/policy-backup.yaml"
  else
    apply_correct_policy
    ok "Re-applied canonical allow-db-from-todo-only policy"
  fi
  rm -f "$STATE_DIR/mode" "$STATE_DIR/policy-backup.yaml" 2>/dev/null || true
  ok "Policy restore complete. Verify todo API: oc exec -n ${CLIENT_NS} deploy/loadgen -- curl -s -o /dev/null -w 'time=%{time_total}s code=%{http_code}\n' -m 10 http://todo.${APP_NS}:8080${TARGET_PATH}"
}

cmd_status() {
  step "NetworkPolicy ${POLICY_NAME} in ${APP_NS}"
  oc get networkpolicy "$POLICY_NAME" -n "$APP_NS" -o yaml 2>/dev/null \
    || echo "  (policy not found)"
  if [[ -f "$STATE_DIR/mode" ]]; then
    echo "  recorded break mode: $(cat "$STATE_DIR/mode")"
  fi
  echo
  step "Todo deployment labels"
  oc get deploy todo -n "$APP_NS" -o jsonpath='{.spec.selector.matchLabels}{"\n"}' 2>/dev/null \
    || echo "  (todo deploy missing)"
  echo
  step "Sample todo API latency"
  local probe_deploy=""
  if oc get deployment/loadgen-heavy -n "$CLIENT_NS" >/dev/null 2>&1; then
    probe_deploy=loadgen-heavy
  elif oc get deployment/loadgen -n "$CLIENT_NS" >/dev/null 2>&1; then
    probe_deploy=loadgen
  fi
  if [[ -n "$probe_deploy" ]]; then
    oc exec -n "$CLIENT_NS" "deploy/${probe_deploy}" -- \
      curl -s -o /dev/null -w 'code=%{http_code} time=%{time_total}s\n' -m 10 \
      "http://todo.${APP_NS}:8080${TARGET_PATH}" 2>/dev/null || echo "  (probe failed)"
  else
    echo "  (no loadgen — run netobserv-krkn-fault.sh inject)"
  fi
}

case "${1:-}" in
  break)   cmd_break ;;
  restore) cmd_restore ;;
  status)  cmd_status ;;
  *)
    cat <<EOF
Usage: $(basename "$0") {break|restore|status}

Microsegmentation incident — wrong NetworkPolicy blocks todo→PostgreSQL (not latency).

Typical flow:
  ./scripts/netobserv-e2e-openclaw-test.sh policy-all
  # Control UI: report connectivity symptoms → agent investigates + captures + analyzes
  # /new → "Yes, restore the network policy"

Environment:
  APP_NS=$APP_NS  POLICY_NAME=$POLICY_NAME
  MODE=$MODE      # wrong-label (default) | deny-all
EOF
    exit 1
    ;;
esac
