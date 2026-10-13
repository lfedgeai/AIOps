#!/usr/bin/env bash
# Run after cluster shutdown/start or long idle — restores demo-ready state.
#
# Usage:
#   ./scripts/post-cluster-resume.sh              # apply fixes + verify
#   ./scripts/post-cluster-resume.sh quick        # RBAC + SPIRE repair only (no Grafana sync)
#   ./scripts/post-cluster-resume.sh status       # read-only checks
#
# When ZTWI is installed, restores SPIFFE mTLS on grafana-bridge after MLflow wire if needed.
#
# Env: same as child scripts (OPENCLAW_NS, LOKI_NS, NETOBSERV_NS, SLACK_CHANNEL_ID, …)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="${1:-all}"
# shellcheck source=resolve-slack-channel.sh
source "$ROOT/scripts/resolve-slack-channel.sh"
resolve_slack_channel_id
export SLACK_CHANNEL_ID

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_red=$'\033[1;31m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
fail() { printf '%s[fail]%s %s\n' "$c_red" "$c_reset" "$*" >&2; FAILURES=$((FAILURES + 1)); }
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }

KUBECTL="$(command -v oc || command -v kubectl)"
[[ -n "$KUBECTL" ]] || { fail "oc/kubectl required"; exit 1; }
FAILURES=0

chmod +x "$ROOT/scripts/fix-netobserv-operator-rbac.sh" \
         "$ROOT/scripts/install-ztwi-spire.sh" \
         "$ROOT/scripts/sync-grafana-demo-metrics.sh" \
         "$ROOT/scripts/repair-todo-postgresql.sh" \
         "$ROOT/scripts/wire-openclaw-mlflow.sh" \
         "$ROOT/scripts/wire-openclaw-spiffe.sh" \
         "$ROOT/scripts/tune-loki-ingestion.sh" \
         "$ROOT/scripts/netobserv-e2e-openclaw-test.sh" 2>/dev/null || true

cmd_status() {
  step "FlowCollector"
  if "$KUBECTL" get flowcollector cluster >/dev/null 2>&1; then
    "$KUBECTL" get flowcollector cluster \
      -o jsonpath='Ready={.status.conditions[?(@.type=="Ready")].status}{" reason="}{.status.conditions[?(@.type=="Ready")].reason}{"\n"}' 2>/dev/null || true
  else
    warn "FlowCollector CR missing"
  fi
  "$ROOT/scripts/fix-netobserv-operator-rbac.sh" status || true
  if "$KUBECTL" get crd clusterspiffeids.spire.spiffe.io >/dev/null 2>&1; then
    step "SPIFFE (optional)"
    "$ROOT/scripts/install-ztwi-spire.sh" status 2>&1 | tail -8 || true
  fi
  step "Event / SPIFFE pre-flight"
  "$ROOT/scripts/netobserv-e2e-openclaw-test.sh" event-aiops-check 2>&1 | tail -6 || fail "event-aiops-check failed"
  if "$KUBECTL" get crd clusterspiffeids.spire.spiffe.io >/dev/null 2>&1; then
    "$ROOT/scripts/netobserv-e2e-openclaw-test.sh" spiffe-check 2>&1 | tail -6 || fail "spiffe-check failed"
  fi
  [[ "$FAILURES" -eq 0 ]] && ok "Post-cluster resume status OK" || fail "${FAILURES} check(s) failed"
  exit "$FAILURES"
}

repair_todo_postgresql_if_needed() {
  local pod phase
  pod="$("$KUBECTL" get pod -n todo-demo -l app=postgresql -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  [[ -n "$pod" ]] || return 0
  phase="$("$KUBECTL" get pod -n todo-demo "$pod" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  if [[ "$phase" == "Running" ]] \
      && "$KUBECTL" get pod -n todo-demo "$pod" -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null | grep -q true; then
    return 0
  fi
  if "$KUBECTL" logs -n todo-demo "$pod" --tail=20 2>/dev/null | grep -q 'could not locate a valid checkpoint record'; then
    step "Repair todo-demo PostgreSQL (WAL checkpoint corruption)"
    "$ROOT/scripts/repair-todo-postgresql.sh" repair || warn "PostgreSQL repair failed"
  fi
}

cmd_apply() {
  local sync_grafana="${1:-1}"
  repair_todo_postgresql_if_needed
  step "NetObserv operator Secret RBAC"
  if "$ROOT/scripts/fix-netobserv-operator-rbac.sh" apply; then
    ok "Operator RBAC applied"
  else
    warn "Operator RBAC skipped (operator not ready?)"
  fi

  if "$KUBECTL" get lokistack loki -n "${LOKI_NS:-netobserv-loki}" >/dev/null 2>&1; then
    step "Loki ingestion limits (demo flow volume)"
    if "$ROOT/scripts/tune-loki-ingestion.sh" apply; then
      ok "Loki ingestion limits applied"
    else
      warn "tune-loki-ingestion failed"
    fi
    if [[ -x "$ROOT/scripts/repair-loki-stack.sh" ]]; then
      "$ROOT/scripts/repair-loki-stack.sh" status || true
      if ! "$ROOT/scripts/repair-loki-stack.sh" status 2>&1 | grep -q 'FlowCollector Loki integration: Ready'; then
        "$ROOT/scripts/repair-loki-stack.sh" apply || warn "repair-loki-stack failed"
      fi
    fi
  fi

  if "$KUBECTL" get crd clusterspiffeids.spire.spiffe.io >/dev/null 2>&1; then
    step "SPIRE / ZTWI repair (optional Phase 3d)"
    if "$ROOT/scripts/install-ztwi-spire.sh" repair; then
      ok "SPIRE repair complete"
    else
      warn "SPIRE repair had warnings"
    fi
  fi

  if [[ "$sync_grafana" == "1" ]] \
      && "$KUBECTL" get deploy openclaw-otel-prometheus -n "${OPENCLAW_NS:-openclaw}" >/dev/null 2>&1; then
    step "Grafana metrics federation"
    "$ROOT/scripts/sync-grafana-demo-metrics.sh" sync || warn "metrics sync failed"
  fi

  if "$KUBECTL" -n "${RHOAI_NS:-redhat-ods-applications}" get deploy/mlflow >/dev/null 2>&1; then
    step "Refresh MLflow Traces wiring (bridge + MCP + guard proxy)"
    "$ROOT/scripts/wire-openclaw-mlflow.sh" || warn "wire-openclaw-mlflow failed"
  fi

  # Cluster idle / MLflow refresh can leave the bridge on the bearer-only Deployment;
  # re-apply SPIFFE mTLS when Phase 3d ClusterSPIFFEID is still present.
  if "$KUBECTL" get clusterspiffeid netobserv-grafana-bridge >/dev/null 2>&1; then
    local bridge_mtls
    bridge_mtls="$("$KUBECTL" -n "${OPENCLAW_NS:-openclaw}" get deploy/netobserv-grafana-bridge \
      -o jsonpath='{.spec.template.spec.containers[?(@.name=="bridge")].env[?(@.name=="SPIFFE_MTLS")].value}' 2>/dev/null || echo "")"
    if [[ "$bridge_mtls" != "1" ]]; then
      step "Restore SPIFFE mTLS on grafana-bridge"
      if "$ROOT/scripts/wire-openclaw-spiffe.sh" all; then
        ok "SPIFFE mTLS restored"
      else
        warn "SPIFFE re-wire failed — try: SLACK_CHANNEL_ID=… ./scripts/post-cluster-spiffe-resume.sh"
      fi
    fi
  fi

  step "Verify event-AIOps path"
  if "$ROOT/scripts/netobserv-e2e-openclaw-test.sh" event-aiops-check; then
    ok "event-aiops-check passed"
  else
    fail "event-aiops-check failed"
  fi

  if "$KUBECTL" get crd clusterspiffeids.spire.spiffe.io >/dev/null 2>&1; then
    step "Verify SPIFFE path"
    if "$ROOT/scripts/netobserv-e2e-openclaw-test.sh" spiffe-check; then
      ok "spiffe-check passed"
    else
      fail "spiffe-check failed — try wire-openclaw-spiffe.sh rollback for Bearer fallback"
    fi
  fi

  if [[ "$FAILURES" -eq 0 ]]; then
    ok "Cluster resume complete — run: ./scripts/netobserv-e2e-openclaw-test.sh demo-a-fast"
    exit 0
  fi
  fail "${FAILURES} step(s) failed"
  exit 1
}

case "$CMD" in
  status) cmd_status ;;
  quick) cmd_apply 0 ;;
  all|apply|"") cmd_apply 1 ;;
  -h|--help|help)
    cat <<EOF
Usage: $(basename "$0") [all|quick|status]

  all     RBAC + SPIRE repair + Grafana metrics sync + pre-flight checks (default)
  quick   RBAC + SPIRE repair + pre-flight (skip Grafana sync)
  status  Read-only status + pre-flight checks

Run after cluster power cycle or long idle before demo-a-fast.
EOF
    exit 0
    ;;
  *) echo "unknown command: $CMD" >&2; exit 1 ;;
esac
