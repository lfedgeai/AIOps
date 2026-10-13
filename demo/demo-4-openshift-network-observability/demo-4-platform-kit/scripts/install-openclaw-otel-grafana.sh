#!/usr/bin/env bash
# Gateway Diagnostics — OTel collector + diagnostics-otel + Grafana panels.
#
# Prerequisites: ./scripts/install-grafana-network-aiops.sh all
#
# Usage:
#   ./scripts/install-openclaw-otel-grafana.sh status
#   ./scripts/install-openclaw-otel-grafana.sh all       # wire OTel + apply Grafana CRs
#   ./scripts/install-openclaw-otel-grafana.sh wire      # OTel only (no Grafana CR sync)
#   ./scripts/install-openclaw-otel-grafana.sh grafana       # Grafana datasource/dashboard only
#   ./scripts/install-openclaw-otel-grafana.sh sync-metrics  # Thanos federation only
#
# Env: same as wire-openclaw-otel.sh + GRAFANA_NS=netobserv-demo
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRAFANA_NS="${GRAFANA_NS:-netobserv-demo}"
MANIFESTS="$ROOT/manifests/grafana-network-aiops"
KUBECTL="$(command -v oc || command -v kubectl)"
CMD="${1:-all}"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -n "$KUBECTL" ]] || die "oc/kubectl required"
chmod +x "$ROOT/scripts/wire-openclaw-otel.sh" "$ROOT/scripts/install-grafana-network-aiops.sh" \
         "$ROOT/scripts/sync-grafana-demo-metrics.sh"

sync_grafana_phase_d() {
  step "Apply Grafana Gateway Diagnostics CRs (${GRAFANA_NS})"
  if ! "$KUBECTL" get grafana network-aiops -n "$GRAFANA_NS" >/dev/null 2>&1; then
    warn "Grafana instance missing — run ./scripts/install-grafana-network-aiops.sh all first"
  fi
  "$KUBECTL" apply -n "$GRAFANA_NS" \
    -f "$MANIFESTS/08-datasource-openclaw-otel.yaml" \
    -f "$MANIFESTS/06-dashboard-network-aiops.yaml"
  sleep 8
  "$ROOT/scripts/sync-grafana-demo-metrics.sh" sync || \
    warn "metrics sync failed — run: ./scripts/sync-grafana-demo-metrics.sh sync"
  ok "Network AIOps demo datasource + dashboard applied (single Prometheus, no fix-auth)"
}

print_status() {
  "$ROOT/scripts/wire-openclaw-otel.sh" status
  step "Grafana Gateway Diagnostics (${GRAFANA_NS})"
  "$KUBECTL" get grafanadatasource prometheus-openclaw-otel -n "$GRAFANA_NS" 2>/dev/null || \
    warn "GrafanaDatasource prometheus-openclaw-otel not applied"
  local host
  host="$("$KUBECTL" get route grafana-network-aiops -n "$GRAFANA_NS" -o jsonpath='{.spec.host}' 2>/dev/null || true)"
  [[ -n "$host" ]] && ok "Grafana: https://${host}/ → Network AIOps — Gateway Diagnostics (flows/packets + agent economics + LLM cost)"
}

case "$CMD" in
  status) print_status ;;
  wire)
    "$ROOT/scripts/wire-openclaw-otel.sh" all
    ;;
  grafana)
    sync_grafana_phase_d
    print_status
    ;;
  sync-metrics)
    "$ROOT/scripts/sync-grafana-demo-metrics.sh" sync
    ;;
  all)
    "$ROOT/scripts/wire-openclaw-otel.sh" all
    sync_grafana_phase_d
    print_status
    cat <<EOF

Phase D complete.

Before the room:
  ./scripts/netobserv-e2e-openclaw-test.sh demo-a-fast   # includes Grafana metrics sync
  1. Send one short message in OpenClaw Control UI (/new → warm-up)
  2. Wait ~30s for OTel flush
  3. Open Grafana → **Network flows & packets** + **Agent economics & gateway** (pricing legend + estimated LLM cost)

Presenter guide: docs/OPENCLAW-OTEL-PRESENTER-GUIDE.md
EOF
    ;;
  *)
    echo "usage: $0 [all|wire|grafana|sync-metrics|status]" >&2
    exit 1
    ;;
esac
