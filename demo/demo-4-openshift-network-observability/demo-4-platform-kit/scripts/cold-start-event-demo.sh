#!/usr/bin/env bash
# Cold-start event-driven Demo A — cluster resume + OpenClaw seed + event wiring (no synthetic webhooks).
#
# Does NOT run: post-cluster-resume.sh (spiffe-check + event-aiops-check POST false Slack threads),
#               spiffe-check, event-aiops-check, demo-a / demo-a-fast inject.
#
# Usage:
#   ./scripts/cold-start-event-demo.sh
#   ./scripts/cold-start-event-demo.sh status    # read-only preflight
#   ./scripts/cold-start-event-demo.sh seed-only # RBAC + seed + wire checks (no inject)
#
# Then inject fault:
#   ./scripts/prepare-event-demo.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="${1:-all}"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}"
# shellcheck source=resolve-slack-channel.sh
source "$ROOT/scripts/resolve-slack-channel.sh"
resolve_slack_channel_id
SKIP_SPIRE="${SKIP_SPIRE:-0}"
SKIP_MLFLOW="${SKIP_MLFLOW:-0}"
REWIRE_EVENTS="${REWIRE_EVENTS:-0}"
KUBECTL="$(command -v oc || command -v kubectl)"

c_green=$'\033[1;32m'; c_blue=$'\033[1;34m'; c_yellow=$'\033[1;33m'; c_reset=$'\033[0m'
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
die()  { printf '%s[fail]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; exit 1; }

[[ -n "$KUBECTL" ]] || die "oc/kubectl required"
export SLACK_CHANNEL_ID

chmod +x "$ROOT/scripts/"*.sh 2>/dev/null || true

fix_openclaw_crashloop() {
  if ! "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw >/dev/null 2>&1; then
    warn "openclaw deploy missing — install OpenClaw first"
    return 0
  fi
  local ready restarts
  ready="$("$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw \
    -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)"
  restarts="$("$KUBECTL" -n "$OPENCLAW_NS" get pods -l app.kubernetes.io/name=openclaw \
    -o jsonpath='{.items[0].status.containerStatuses[?(@.name=="openclaw")].restartCount}' 2>/dev/null || echo 0)"
  if [[ "${ready:-0}" == "0" ]] || [[ "${restarts:-0}" -gt 2 ]]; then
    step "OpenClaw not healthy — teardown deprecated netobserv-input-guard plugin"
    "$ROOT/scripts/teardown-custom-input-guard.sh" || warn "teardown-custom-input-guard failed"
  fi
}

cmd_preflight() {
  step "FlowCollector / operator RBAC"
  if [[ -x "$ROOT/scripts/fix-netobserv-operator-rbac.sh" ]]; then
    "$ROOT/scripts/fix-netobserv-operator-rbac.sh" apply || warn "operator RBAC apply skipped"
  fi

  step "OpenClaw + event path (read-only)"
  "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw >/dev/null 2>&1 \
    || die "openclaw deploy missing in $OPENCLAW_NS"
  "$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-capture-proxy >/dev/null 2>&1 \
    || die "netobserv-capture-proxy missing — run seed-only or ./scripts/seed-openclaw-netobserv-skills.sh"
  "$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-grafana-bridge >/dev/null 2>&1 \
    || die "netobserv-grafana-bridge missing — SLACK_CHANNEL_ID=$SLACK_CHANNEL_ID ./scripts/wire-openclaw-event-aiops.sh"

  SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" "$ROOT/scripts/prepare-event-demo.sh" status
}

cmd_seed_only() {
  require_slack_channel_id || die "SLACK_CHANNEL_ID required (env or site-secrets.local.yaml)"
  step "Operator RBAC"
  "$ROOT/scripts/fix-netobserv-operator-rbac.sh" apply || warn "operator RBAC skipped"

  if [[ "$SKIP_SPIRE" != "1" ]] && "$KUBECTL" get crd clusterspiffeids.spire.spiffe.io >/dev/null 2>&1; then
    step "SPIRE / ZTWI repair (optional)"
    "$ROOT/scripts/install-ztwi-spire.sh" repair || warn "SPIRE repair had warnings"
  fi

  if [[ "$SKIP_MLFLOW" != "1" ]] \
      && "$KUBECTL" -n "${RHOAI_NS:-redhat-ods-applications}" get deploy/mlflow >/dev/null 2>&1; then
    step "MLflow Traces wiring"
    "$ROOT/scripts/wire-openclaw-mlflow.sh" || warn "wire-openclaw-mlflow failed"
  fi

  fix_openclaw_crashloop

  step "Seed OpenClaw skills + MCP + capture-proxy"
  SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" "$ROOT/scripts/seed-openclaw-netobserv-skills.sh"

  if [[ "${SKIP_TRUSTYAI_GUARD:-0}" != "1" ]] \
      && "$KUBECTL" -n "${GUARDRAILS_NS:-netobserv-guardrails}" get guardrailsorchestrator guardrails-orchestrator >/dev/null 2>&1; then
    step "Reconcile kit-owned OpenClaw wiring"
    SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" SKIP_METRICS=1 \
      "$ROOT/scripts/reconcile-openclaw-demo-wiring.sh" apply || warn "reconcile failed"
  fi

  step "Clear stale hook / MCP sessions"
  "$ROOT/scripts/clear-openclaw-sessions.sh" || warn "clear-openclaw-sessions failed"

  if [[ "$REWIRE_EVENTS" == "1" ]]; then
    step "Re-wire Grafana alert + hooks"
    SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" "$ROOT/scripts/wire-openclaw-event-aiops.sh"
  else
    step "Event-AIOps wiring (status)"
    SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" "$ROOT/scripts/wire-openclaw-event-aiops.sh" status \
      || SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" "$ROOT/scripts/wire-openclaw-event-aiops.sh"
  fi

  step "Grafana metrics federation"
  "$ROOT/scripts/sync-grafana-demo-metrics.sh" sync || warn "metrics sync failed"

  ok "Seed complete — next: SLACK_CHANNEL_ID=$SLACK_CHANNEL_ID ./scripts/prepare-event-demo.sh"
}

cmd_all() {
  cmd_seed_only
  step "Inject fault (prepare-event-demo)"
  SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" "$ROOT/scripts/prepare-event-demo.sh"
}

case "$CMD" in
  all|"") cmd_all ;;
  seed-only|seed) cmd_seed_only ;;
  status|preflight) cmd_preflight ;;
  -h|--help|help)
    cat <<EOF
Usage: $(basename "$0") [all|seed-only|status]

  all        seed-only + prepare-event-demo (full cold start + inject)
  seed-only  RBAC, MLflow, seed, clear sessions, wire check, metrics sync
  status     read-only preflight (no changes)

Env: SLACK_CHANNEL_ID SKIP_SPIRE=1 SKIP_MLFLOW=1 REWIRE_EVENTS=1

Skips spiffe-check and event-aiops-check (no synthetic Slack threads).
EOF
    ;;
  *)
    die "Unknown subcommand: $CMD (try --help)"
    ;;
esac
