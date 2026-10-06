#!/usr/bin/env bash
# Restore SPIFFE mTLS + event-AIOps after cluster power cycle.
#
# Use this instead of post-cluster-resume.sh when preparing a security demo —
# post-cluster-resume runs synthetic webhook checks that can spawn Slack threads.
#
# Usage:
#   ./scripts/post-cluster-spiffe-resume.sh
#   ./scripts/post-cluster-spiffe-resume.sh verify  # spiffe-check only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="${1:-all}"
SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}"

# shellcheck source=resolve-slack-channel.sh
source "$ROOT/scripts/resolve-slack-channel.sh"
resolve_slack_channel_id

c_green=$'\033[1;32m'; c_blue=$'\033[1;34m'; c_yellow=$'\033[1;33m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

require_slack_channel_id || die "SLACK_CHANNEL_ID required (env or site.slack_channel_id in site-secrets.local.yaml)"

chmod +x "$ROOT/scripts/install-ztwi-spire.sh" \
         "$ROOT/scripts/wire-openclaw-event-aiops.sh" \
         "$ROOT/scripts/wire-openclaw-spiffe.sh" \
         "$ROOT/scripts/netobserv-e2e-openclaw-test.sh" 2>/dev/null || true

cmd_verify() {
  step "Verify SPIFFE path"
  SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" "$ROOT/scripts/netobserv-e2e-openclaw-test.sh" spiffe-check
}

cmd_all() {
  step "SPIRE repair (agents + CSI + OpenClaw SPIFFE workloads)"
  "$ROOT/scripts/install-ztwi-spire.sh" repair

  step "Event-AIOps (hooks + Grafana alert contact point)"
  RECYCLE_POD=0 SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" \
    "$ROOT/scripts/wire-openclaw-event-aiops.sh" all

  step "SPIFFE mTLS (bridge → openclaw-hooks-mtls)"
  SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" "$ROOT/scripts/wire-openclaw-spiffe.sh" all

  cmd_verify
  ok "SPIFFE ready — run demo-cluster-preflight check, then demo-a-fast at T−5 min"
  warn "Control UI: /new → warm-up message (clear-openclaw-sessions.sh only if wedged)"
}

case "$CMD" in
  all|"") cmd_all ;;
  verify|check) cmd_verify ;;
  *)
    echo "usage: SLACK_CHANNEL_ID=… $0 [all|verify]" >&2
    exit 1
    ;;
esac
