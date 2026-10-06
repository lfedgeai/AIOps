#!/usr/bin/env bash
# resolve-slack-channel.sh — load SLACK_CHANNEL_ID from env or greenfield site-secrets.
#
# Source from kit scripts (do not execute directly):
#   # shellcheck source=resolve-slack-channel.sh
#   source "$ROOT/scripts/resolve-slack-channel.sh"
#   resolve_slack_channel_id
#   require_slack_channel_id   # optional — exit 1 when still unset

_resolve_slack_kit_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$script_dir/.." && pwd
}

resolve_slack_channel_id() {
  [[ -n "${SLACK_CHANNEL_ID:-}" ]] && return 0
  local secrets="${SITE_SECRETS_FILE:-}"
  local root
  root="$(_resolve_slack_kit_root)"
  if [[ -z "$secrets" || ! -f "$secrets" ]]; then
    for candidate in \
      "$HOME/AIOps/demo/demo-4-openshift-network-observability/demo-4-greenfield-install/config/site-secrets.local.yaml" \
      "$HOME/AIOps/demo/demo-4-greenfield-install/config/site-secrets.local.yaml" \
      "$root/../demo-4-greenfield-install/config/site-secrets.local.yaml"; do
      if [[ -f "$candidate" ]]; then
        secrets="$candidate"
        break
      fi
    done
  fi
  if [[ -f "$secrets" ]]; then
    SLACK_CHANNEL_ID="$(python3 - "$secrets" <<'PY' 2>/dev/null || true
import sys
try:
    import yaml
except ImportError:
    sys.exit(0)
p = sys.argv[1]
d = yaml.safe_load(open(p)) or {}
print((d.get("site") or {}).get("slack_channel_id", ""))
PY
)"
    export SLACK_CHANNEL_ID
  fi
}

require_slack_channel_id() {
  resolve_slack_channel_id
  if [[ -z "${SLACK_CHANNEL_ID:-}" ]]; then
    echo "SLACK_CHANNEL_ID required — set env or site.slack_channel_id in site-secrets.local.yaml" >&2
    return 1
  fi
}
