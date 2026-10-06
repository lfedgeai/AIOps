#!/usr/bin/env bash
# Prepare event-driven Demo A — restore, cleanup, inject Kraken (latency-only), sync metrics.
# Does NOT run spiffe-check / event-aiops-check (those POST synthetic webhooks → false Slack threads).
#
# Usage:
#   ./scripts/prepare-event-demo.sh
#   ./scripts/prepare-event-demo.sh status          # read-only: event path + fault + RTT sample
#   ./scripts/prepare-event-demo.sh inject-only   # skip restore/cleanup (fault already cleared)
#
# Env:
#   SLACK_CHANNEL_ID          env or site.slack_channel_id in site-secrets.local.yaml
#   LOSS=0                         default 0 — cleaner RTT alert than packet loss
#   LATENCY_MS=800                 Kraken pod egress delay
#   TEST_DURATION=1800             fault window (30 min)
#   REWIRE_ALERTS=0|1              default 1 — re-provision Grafana contact point (needed after bridge pod restart)
#   SKIP_RESTORE=0|1               default 0 — set 1 with inject-only
#   SETTLE_HINT_SEC=180             printed wait before Grafana alert fires (~2–3 min)
#   AGENT_HINT_SEC=480              printed wait for Slack thread after alert (~5–8 min more)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="${1:-all}"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
GRAFANA_NS="${GRAFANA_NS:-netobserv-demo}"
APP_NS="${APP_NS:-todo-demo}"
CLIENT_NS="${CLIENT_NS:-todo-client}"
SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}"
LOSS="${LOSS:-0}"
LATENCY_MS="${LATENCY_MS:-800}"
TEST_DURATION="${TEST_DURATION:-1800}"
REWIRE_ALERTS="${REWIRE_ALERTS:-1}"
SKIP_RESTORE="${SKIP_RESTORE:-0}"
SETTLE_HINT_SEC="${SETTLE_HINT_SEC:-180}"
AGENT_HINT_SEC="${AGENT_HINT_SEC:-480}"
KUBECTL="$(command -v oc || command -v kubectl)"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_cyan=$'\033[1;36m'; c_reset=$'\033[0m'
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
die()  { printf '%s[fail]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; exit 1; }
note() { printf '%s---%s %s\n' "$c_cyan" "$c_reset" "$*"; }

[[ -n "$KUBECTL" ]] || die "oc/kubectl required"

# shellcheck source=resolve-slack-channel.sh
source "$ROOT/scripts/resolve-slack-channel.sh"
require_slack_channel_id || die "SLACK_CHANNEL_ID required — set in site-secrets.local.yaml or env"

export SLACK_CHANNEL_ID LOSS LATENCY_MS TEST_DURATION

chmod +x "$ROOT/scripts/netobserv-e2e-openclaw-test.sh" \
         "$ROOT/scripts/netobserv-krkn-fault.sh" \
         "$ROOT/scripts/sync-grafana-demo-metrics.sh" \
         "$ROOT/scripts/wire-grafana-openclaw-alerts.sh" \
         "$ROOT/scripts/wire-openclaw-event-aiops.sh" 2>/dev/null || true

cleanup_debug_artifacts() {
  step "Remove stale debug / probe workloads (reduces agent noise)"
  local deleted=0 name
  for name in pg-debug pg-debug2 pg-debug3 pg-resetwal; do
    if "$KUBECTL" -n "$APP_NS" delete pod "$name" --ignore-not-found --wait=false >/dev/null 2>&1; then
      deleted=$((deleted + 1))
    fi
  done
  "$KUBECTL" -n "$CLIENT_NS" delete jobs -l 'job-name' --field-selector status.successful=1 \
    --ignore-not-found --wait=false 2>/dev/null || true
  while read -r name; do
    [[ -n "$name" ]] || continue
    case "$name" in netobserv-latency-probe-*)
      "$KUBECTL" -n "$CLIENT_NS" delete job "$name" --ignore-not-found --wait=false >/dev/null 2>&1 || true
      deleted=$((deleted + 1))
      ;;
    esac
  done < <("$KUBECTL" -n "$CLIENT_NS" get jobs -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true)
  ok "Cleanup requested (debug pods + old latency probe jobs)"
}

verify_event_path() {
  step "Event-AIOps path (read-only — no synthetic webhook POST)"
  "$KUBECTL" -n "$OPENCLAW_NS" get secret openclaw-slack-tokens >/dev/null 2>&1 \
    || die "openclaw-slack-tokens missing — wire Slack first (docs/SLACK-PRESENTER-GUIDE.md)"
  ok "openclaw-slack-tokens present"

  "$KUBECTL" -n "$OPENCLAW_NS" get secret openclaw-hooks-token >/dev/null 2>&1 \
    || die "openclaw-hooks-token missing — SLACK_CHANNEL_ID=$SLACK_CHANNEL_ID ./scripts/wire-openclaw-event-aiops.sh"
  ok "openclaw-hooks-token present"

  "$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-grafana-bridge >/dev/null 2>&1 \
    || die "netobserv-grafana-bridge missing — SLACK_CHANNEL_ID=$SLACK_CHANNEL_ID ./scripts/wire-openclaw-event-aiops.sh"
  local nready
  nready="$("$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-grafana-bridge \
    -o jsonpath='{.status.readyReplicas}/{.spec.replicas}' 2>/dev/null || echo 0/0)"
  [[ "$nready" != "0/0" && "$nready" != "0/"* ]] \
    || die "netobserv-grafana-bridge not ready ($nready)"
  ok "netobserv-grafana-bridge ready ($nready)"

  "$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-capture-proxy >/dev/null 2>&1 \
    || die "netobserv-capture-proxy missing — ./scripts/seed-openclaw-netobserv-skills.sh"
  nready="$("$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-capture-proxy \
    -o jsonpath='{.status.readyReplicas}/{.spec.replicas}' 2>/dev/null || echo 0/0)"
  [[ "$nready" != "0/0" && "$nready" != "0/"* ]] \
    || die "netobserv-capture-proxy not ready ($nready) — re-seed or rollout restart"
  ok "netobserv-capture-proxy ready ($nready)"

  if [[ "$REWIRE_ALERTS" == "1" ]]; then
    step "Re-provision Grafana RTT alert + contact point"
    "$ROOT/scripts/wire-grafana-openclaw-alerts.sh" all
  else
    "$ROOT/scripts/wire-grafana-openclaw-alerts.sh" status
  fi
  ok "Grafana alert → bridge contact point configured"
}

inject_fault() {
  step "Inject Kraken fault (LOSS=${LOSS}, LATENCY_MS=${LATENCY_MS}, TEST_DURATION=${TEST_DURATION}s)"
  MODE=pod TARGET=todo LOSS="$LOSS" TEST_DURATION="$TEST_DURATION" LATENCY_MS="$LATENCY_MS" \
    "$ROOT/scripts/netobserv-krkn-fault.sh" inject
  MODE=pod TARGET=todo LOSS="$LOSS" TEST_DURATION="$TEST_DURATION" LATENCY_MS="$LATENCY_MS" \
    "$ROOT/scripts/netobserv-krkn-fault.sh" slow "$LATENCY_MS"

  step "Sync NetObserv metrics into demo Prometheus (Grafana datasource)"
  "$ROOT/scripts/sync-grafana-demo-metrics.sh" sync || \
    warn "metrics sync failed — run: ./scripts/sync-grafana-demo-metrics.sh sync"
}

sample_rtt_sec() {
  local prom_pod
  prom_pod="$("$KUBECTL" -n "$OPENCLAW_NS" get pods -l app.kubernetes.io/name=openclaw-otel-prometheus \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  [[ -n "$prom_pod" ]] || { echo ""; return 1; }
  "$KUBECTL" -n "$OPENCLAW_NS" exec "$prom_pod" -- wget -qO- --post-data='query=sum(rate(netobserv_namespace_rtt_seconds_sum{SrcK8S_Namespace="todo-demo"}[2m])) / sum(rate(netobserv_namespace_rtt_seconds_count{SrcK8S_Namespace="todo-demo"}[2m]))' \
    http://127.0.0.1:9090/api/v1/query 2>/dev/null | \
    python3 -c 'import json,sys; d=json.load(sys.stdin); r=d.get("data",{}).get("result",[]); print(r[0]["value"][1] if r else "")' 2>/dev/null || true
}

sample_api_latency() {
  "$KUBECTL" exec -n "$CLIENT_NS" deploy/loadgen-heavy -- \
    curl -sS -o /dev/null -w '%{time_total}' -m 25 \
    "http://todo.${APP_NS}:8080/api" 2>/dev/null || echo ""
}

verify_fault_metrics() {
  step "Post-inject sanity (Kraken + latency samples)"
  "$ROOT/scripts/netobserv-krkn-fault.sh" status || true
  local rtt api_t
  rtt="$(sample_rtt_sec || true)"
  api_t="$(sample_api_latency || true)"
  if [[ -n "$rtt" ]]; then
    ok "Federated todo-demo avg RTT (instant): ${rtt}s"
  else
    warn "No RTT sample yet — wait ~${SETTLE_HINT_SEC}s for tc + rate() window"
  fi
  if [[ -n "$api_t" ]]; then
    ok "Loadgen → todo /api sample: ${api_t}s"
  else
    warn "Loadgen latency probe failed — check: oc get pods -n $CLIENT_NS"
  fi
}

print_next_steps() {
  local grafana_host
  grafana_host="$("$KUBECTL" get route grafana-network-aiops -n "$GRAFANA_NS" \
    -o jsonpath='{.spec.host}' 2>/dev/null || true)"
  cat <<EOF

${c_green}Event-driven Demo A prepared.${c_reset}

  Do NOT run:  ./scripts/netobserv-e2e-openclaw-test.sh spiffe-check
               ./scripts/netobserv-e2e-openclaw-test.sh event-aiops-check
  (those POST synthetic webhooks and start a false Slack investigate thread)

  Timeline (event-driven — allow ~8–10 min end-to-end):
    • ~${SETTLE_HINT_SEC}s — Grafana alert "NetObserv todo-demo avg RTT elevated" → Alerting
    • +2–3 min — bridge → OpenClaw hook (check: oc logs deploy/netobserv-grafana-bridge | grep POST)
    • +5–8 min — Slack #netobserv-demo thread (hook: Grafana-NetObserv) after capture + analyze

  Cold start from scratch: ./scripts/cold-start-event-demo.sh

  Grafana dashboard (Last 30m): ${grafana_host:+https://${grafana_host}/d/network-aiops-openclaw/network-aiops-e28094-gateway-diagnostics}
  Status:  ./scripts/prepare-event-demo.sh status
  Heal:    human confirms in Slack → then ./scripts/netobserv-e2e-openclaw-test.sh restore

  Guide: docs/EVENT-DRIVEN-AIOPS-GUIDE.md

EOF
}

cmd_all() {
  if [[ "$SKIP_RESTORE" != "1" ]]; then
    step "Restore cluster demo state (Kraken + loadgen + policy)"
    "$ROOT/scripts/netobserv-e2e-openclaw-test.sh" restore
  fi
  cleanup_debug_artifacts
  verify_event_path
  inject_fault
  verify_fault_metrics
  print_next_steps
}

cmd_inject_only() {
  SKIP_RESTORE=1
  verify_event_path
  inject_fault
  verify_fault_metrics
  print_next_steps
}

cmd_status() {
  verify_event_path
  "$ROOT/scripts/netobserv-krkn-fault.sh" status || true
  verify_fault_metrics
  print_next_steps
}

case "$CMD" in
  all|prepare|"") cmd_all ;;
  inject-only|inject) cmd_inject_only ;;
  status) cmd_status ;;
  -h|--help|help)
    cat <<EOF
Usage: $(basename "$0") [all|inject-only|status]

  all          restore → cleanup → verify event path → inject (LOSS=0) → sync metrics
  inject-only  skip restore; inject + sync only
  status       read-only checks + latency samples

Env: SLACK_CHANNEL_ID LOSS LATENCY_MS TEST_DURATION REWIRE_ALERTS SKIP_RESTORE

Does not run spiffe-check or event-aiops-check (avoids SpiffeCheck false Slack threads).
EOF
    ;;
  *)
    die "Unknown subcommand: $CMD (try --help)"
    ;;
esac
