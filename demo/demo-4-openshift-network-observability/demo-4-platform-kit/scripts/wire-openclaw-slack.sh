#!/usr/bin/env bash
# Wire OpenClaw Slack (Socket Mode) for enterprise demo channels.
#
# Prerequisites (manual, one-time):
#   - Slack app installed to workspace; secret openclaw-slack-tokens with
#     SLACK_BOT_TOKEN + SLACK_APP_TOKEN
#   - oc -n openclaw set env deployment/openclaw --from=secret/openclaw-slack-tokens
#   - channels.slack block in lab config.yaml (channel allowlist, requireMention)
#
# This script:
#   1) Pins plugins.allow + plugins.entries.slack in lab openclaw.json
#   2) Hardens seed-openclaw init to npm install @openclaw/slack@2026.6.11 (v5)
#   3) Recycles the OpenClaw pod so init reinstalls plugins on EmptyDir
#
# Env:
#   SLACK_CHANNEL_ID=C0123456789   optional — merge into channels.slack.channels (env or site-secrets)
#   RECYCLE_POD=1|0                default 1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
LAB_CFG="${LAB_CFG:-$HOME/labs/openshell-on-openshift-lab/manifests/openclaw/config.yaml}"
KUBECTL="$(command -v oc || command -v kubectl)"
RECYCLE_POD="${RECYCLE_POD:-1}"
SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}"
# shellcheck source=resolve-slack-channel.sh
source "$ROOT/scripts/resolve-slack-channel.sh"
resolve_slack_channel_id

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
step() { printf '\n%s==>%s %s\n' "$c_green" "$c_reset" "$*"; }

[[ -n "$KUBECTL" ]] || { echo "oc/kubectl required" >&2; exit 1; }

step "Pin slack in lab openclaw.json (plugins.allow + entries.slack)"
if [[ ! -f "$LAB_CFG" ]]; then
  warn "Lab config not found: $LAB_CFG — skip JSON pin (ensure plugins.allow includes slack)"
else
  python3 - "$LAB_CFG" "$SLACK_CHANNEL_ID" <<'PY'
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
channel_id = (sys.argv[2] or "").strip()
d = json.loads(p.read_text())
changed = False

plugins = d.setdefault("plugins", {})
allow = list(plugins.get("allow") or [])
if "slack" not in allow:
    allow.append("slack")
    plugins["allow"] = allow
    changed = True

entries = plugins.setdefault("entries", {})
if entries.get("slack") != {"enabled": True}:
    entries["slack"] = {"enabled": True}
    changed = True

slack_ch = d.setdefault("channels", {}).setdefault("slack", {})
if slack_ch.get("enabled") is not True:
    slack_ch["enabled"] = True
    changed = True
if channel_id:
    ch_map = slack_ch.setdefault("channels", {})
    cur = dict(ch_map.get(channel_id) or {})
    if "allow" in cur:
        cur.pop("allow", None)
        changed = True
    if cur.get("enabled") is not True:
        cur["enabled"] = True
        changed = True
    if cur.get("requireMention") is not True:
        cur["requireMention"] = True
        changed = True
    if changed:
        ch_map[channel_id] = cur

if changed:
    p.write_text(json.dumps(d, indent=2) + "\n")
    print("patched", p)
else:
    print("unchanged", p)
PY
  ok "Updated $LAB_CFG"
  "$KUBECTL" -n "$OPENCLAW_NS" apply -k "$(dirname "$LAB_CFG")"
  ok "Applied openclaw kustomize"
fi

step "Harden seed-openclaw init (@openclaw/slack npm install on every pod start)"
chmod +x "$ROOT/scripts/patch-openclaw-seed-idempotent.sh"
OPENCLAW_NS="$OPENCLAW_NS" "$ROOT/scripts/patch-openclaw-seed-idempotent.sh"
ok "seed-openclaw init patched (NETOBSERV_SEED_HARDENED_v5)"

if [[ "$RECYCLE_POD" == "1" ]]; then
  step "Recycle OpenClaw pod (init reinstalls plugins on EmptyDir)"
  "$KUBECTL" -n "$OPENCLAW_NS" delete pod -l app.kubernetes.io/name=openclaw --wait=false
  "$KUBECTL" -n "$OPENCLAW_NS" rollout status deployment/openclaw --timeout=600s
  ok "OpenClaw pod recycled"
fi

step "Verify slack plugin + channel probe"
POD="$("$KUBECTL" -n "$OPENCLAW_NS" get pods -l app.kubernetes.io/name=openclaw \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
if [[ -z "$POD" ]]; then
  warn "No running OpenClaw pod yet"
  exit 0
fi

"$KUBECTL" -n "$OPENCLAW_NS" exec "$POD" -c openclaw -- sh -lc '
  set -e
  f=$(ls /opt/openclaw/config/npm/projects/*/node_modules/@openclaw/slack/dist/index.js 2>/dev/null | head -1)
  if [ -n "$f" ]; then echo "slack plugin: $f"; else echo "MISSING slack plugin" >&2; exit 1; fi
  HOME=/opt/openclaw OPENCLAW_CONFIG_PATH=/opt/openclaw/config/openclaw.json
  node /app/openclaw.mjs plugins list 2>&1 | grep -i slack || true
  echo "---"
  node /app/openclaw.mjs channels status --probe 2>&1
' || warn "Slack verification failed — check init logs: oc logs -n openclaw $POD -c seed-openclaw"

ok "Done. Test in Slack: @OpenClaw in your allowlisted channel (requireMention: true)."
