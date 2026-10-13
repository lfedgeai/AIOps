#!/usr/bin/env bash
# reconcile-openclaw-demo-wiring.sh — single reconciliation for NetObserv demo wiring.
#
# Addresses the design problem: lab kustomize, kit patches, and wire scripts each owned
# different slices of openclaw.json; any `apply -k` or partial patch reset the others.
#
# This script:
#   1. Merges kit-owned fields into lab config.yaml (baseUrl, hooks, slack, no input-guard)
#   2. Applies lab kustomize ONCE
#   3. Deploys satellite resources (bridge, hooks env) via existing wire scripts
#   4. Re-wires Grafana alerts (drifts after bridge restart)
#   5. Recycles OpenClaw ONCE if config changed
#
# Does NOT POST synthetic webhooks.
#
# Usage:
#   SLACK_CHANNEL_ID=C0123456789 ./scripts/reconcile-openclaw-demo-wiring.sh
#   ./scripts/reconcile-openclaw-demo-wiring.sh status
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
GUARDRAILS_NS="${GUARDRAILS_NS:-netobserv-guardrails}"
# shellcheck source=resolve-slack-channel.sh
source "$ROOT/scripts/resolve-slack-channel.sh"
resolve_slack_channel_id
SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}"
SKIP_METRICS="${SKIP_METRICS:-0}"
KUBECTL="$(command -v oc || command -v kubectl)"
CMD="${1:-apply}"
LAB_CFG="${LAB_CFG:-$HOME/labs/openshell-on-openshift-lab/manifests/openclaw/config.yaml}"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }

[[ -n "$KUBECTL" ]] || { echo "oc/kubectl required" >&2; exit 1; }
export SLACK_CHANNEL_ID
chmod +x "$ROOT/scripts/"*.sh 2>/dev/null || true

CONFIG_CHANGED=0
OPENCLAW_RECYCLED=0

recycle_openclaw_once() {
  [[ "$OPENCLAW_RECYCLED" == "1" ]] && return 0
  step "Recycle OpenClaw (single restart after reconcile)"
  "$KUBECTL" -n "$OPENCLAW_NS" rollout restart deployment/openclaw 2>/dev/null \
    || "$KUBECTL" -n "$OPENCLAW_NS" delete pod -l app=openclaw --ignore-not-found
  "$KUBECTL" -n "$OPENCLAW_NS" rollout status deployment/openclaw --timeout=300s 2>/dev/null || true
  OPENCLAW_RECYCLED=1
  ok "OpenClaw recycled"
}

cmd_apply() {
  step "Reconcile NetObserv demo wiring"

  "$ROOT/scripts/teardown-custom-input-guard.sh" 2>/dev/null || true
  if [[ -x "$ROOT/scripts/fix-netobserv-operator-rbac.sh" ]]; then
    "$ROOT/scripts/fix-netobserv-operator-rbac.sh" apply >/dev/null 2>&1 || true
  fi

  step "Merge kit-owned fields into lab openclaw.json"
  merge_out="$(SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" "$ROOT/scripts/merge-openclaw-lab-config.sh" apply 2>&1)" || {
    warn "merge-openclaw-lab-config failed"
    echo "$merge_out" >&2
  }
  if ! grep -q 'changed: none' <<<"$merge_out"; then
    CONFIG_CHANGED=1
  fi

  if [[ -f "$LAB_CFG" ]]; then
    step "Apply lab openclaw kustomize (single apply after merge)"
    "$KUBECTL" -n "$OPENCLAW_NS" apply -k "$(dirname "$LAB_CFG")"
    ok "openclaw-config from lab kustomize"
  else
    warn "Lab config missing: $LAB_CFG"
  fi

  if "$KUBECTL" -n "$OPENCLAW_NS" get secret openclaw-slack-tokens >/dev/null 2>&1; then
    step "Reconcile Slack plugin"
    RECYCLE_POD=0 SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" \
      "$ROOT/scripts/wire-openclaw-slack.sh" || warn "wire-openclaw-slack failed"
  fi

  if "$KUBECTL" -n "$OPENCLAW_NS" get secret openclaw-hooks-token >/dev/null 2>&1 \
      && [[ -n "$SLACK_CHANNEL_ID" ]]; then
    step "Reconcile event hooks + bridge (no smoke POST)"
    RECYCLE_POD=0 SKIP_KUSTOMIZE=1 SKIP_SMOKE=1 SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" \
      "$ROOT/scripts/wire-openclaw-hooks.sh" || warn "wire-openclaw-hooks failed"
  fi

  if "$KUBECTL" get clusterspiffeid netobserv-grafana-bridge >/dev/null 2>&1; then
    bridge_mtls="$("$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-grafana-bridge \
      -o jsonpath='{.spec.template.spec.containers[?(@.name=="bridge")].env[?(@.name=="SPIFFE_MTLS")].value}' 2>/dev/null || echo "")"
    if [[ "$bridge_mtls" != "1" ]]; then
      step "Restore SPIFFE mTLS on grafana-bridge (reconcile must not downgrade to Bearer)"
      if SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" "$ROOT/scripts/wire-openclaw-spiffe.sh" all; then
        ok "SPIFFE mTLS restored"
      else
        warn "SPIFFE re-wire failed — run post-cluster-spiffe-resume.sh"
      fi
    fi
  fi

  if "$KUBECTL" -n "$GUARDRAILS_NS" get guardrailsorchestrator guardrails-orchestrator >/dev/null 2>&1; then
    step "Verify TrustyAI guard-proxy path + annotate"
    RECYCLE_POD=0 "$ROOT/scripts/wire-openclaw-trustyai-guardrails.sh" all || \
      warn "wire-openclaw-trustyai-guardrails failed"
  fi

  if "$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-grafana-bridge >/dev/null 2>&1; then
    step "Reconcile Grafana RTT alert contact point"
    "$ROOT/scripts/wire-grafana-openclaw-alerts.sh" all || \
      warn "wire-grafana-openclaw-alerts failed"
  fi

  if [[ "$SKIP_METRICS" != "1" ]] \
      && "$KUBECTL" -n "$OPENCLAW_NS" get deploy openclaw-otel-prometheus >/dev/null 2>&1; then
    "$ROOT/scripts/sync-grafana-demo-metrics.sh" sync || warn "metrics sync failed"
  fi

  if [[ "$CONFIG_CHANGED" == "1" && "$OPENCLAW_RECYCLED" != "1" ]]; then
    recycle_openclaw_once
  elif [[ "$OPENCLAW_RECYCLED" != "1" ]]; then
    ok "OpenClaw config unchanged — skip recycle"
  fi

  ok "Reconcile complete — verify: ./scripts/demo-cluster-preflight.sh check"
}

cmd_status() {
  SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" "$ROOT/scripts/demo-cluster-preflight.sh" check
}

case "$CMD" in
  apply|reconcile|all|"") cmd_apply ;;
  status|check) cmd_status ;;
  -h|--help|help)
    cat <<EOF
Usage: $(basename "$0") [apply|status]

  apply    Merge kit config → lab file → kustomize → wire satellites (default)
  status   Run demo-cluster-preflight check

Kit-owned fields (persisted in lab config.yaml):
  - LLM baseUrl (guard-proxy when TrustyAI installed, else LiteMaaS)
  - hooks.* (when openclaw-hooks-token exists)
  - channels.slack.* (when Slack tokens + SLACK_CHANNEL_ID)
  - removal of deprecated netobserv-input-guard plugin

See: docs/DEMO-WIRING-ARCHITECTURE.md
EOF
    ;;
  *)
    echo "usage: $0 [apply|status]" >&2
    exit 1
    ;;
esac
