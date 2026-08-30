#!/usr/bin/env bash
# Wire full event-driven AIOps path: OpenClaw hooks + Grafana alert → Slack investigate.
#
# Usage:
#   ./scripts/wire-openclaw-event-aiops.sh
#   ./scripts/wire-openclaw-event-aiops.sh status
#
# Env: SLACK_CHANNEL_ID (required — env or site-secrets), RECYCLE_POD (default 1 on hooks step only)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="${1:-all}"
KUBECTL="$(command -v oc || command -v kubectl)"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
# shellcheck source=resolve-slack-channel.sh
source "$ROOT/scripts/resolve-slack-channel.sh"
resolve_slack_channel_id

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_reset=$'\033[0m'
ok() { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }

chmod +x "$ROOT/scripts/wire-openclaw-hooks.sh" \
         "$ROOT/scripts/wire-grafana-openclaw-alerts.sh" \
         "$ROOT/scripts/wire-openclaw-slack.sh" \
         "$ROOT/scripts/sync-grafana-demo-metrics.sh" 2>/dev/null || true

case "$CMD" in
  all|wire)
    require_slack_channel_id || {
      echo "error: SLACK_CHANNEL_ID required — set env or site.slack_channel_id in site-secrets.local.yaml" >&2
      exit 1
    }
    if ! "$KUBECTL" -n "${OPENCLAW_NS:-openclaw}" get secret openclaw-slack-tokens >/dev/null 2>&1; then
      echo "error: openclaw-slack-tokens missing — complete Slack setup first (docs/SLACK-PRESENTER-GUIDE.md)" >&2
      exit 1
    fi
    "$ROOT/scripts/sync-grafana-demo-metrics.sh" sync || \
      warn "metrics sync failed — alert may stay NoData until ./scripts/sync-grafana-demo-metrics.sh sync"
    RECYCLE_POD="${RECYCLE_POD:-1}" "$ROOT/scripts/wire-openclaw-hooks.sh"
    "$ROOT/scripts/wire-grafana-openclaw-alerts.sh" all
    ok "Event-driven AIOps wired. Run demo-a-fast; after ~2m of elevated RTT, Slack should auto-investigate."
    ok "Verify: ./scripts/netobserv-e2e-openclaw-test.sh event-aiops-check"
    ;;
  status)
    "$ROOT/scripts/netobserv-e2e-openclaw-test.sh" event-aiops-check
    ;;
  *)
    echo "usage: $0 [all|wire|status]" >&2
    exit 1
    ;;
esac
