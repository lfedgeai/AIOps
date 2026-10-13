#!/usr/bin/env bash
# Clear wedged OpenClaw Control UI sessions (fixes "session initialization conflicted",
# stalled dashboard sessions, blocked exec when sandbox ImagePullBackOff, and Streamable HTTP MCP "Session not found").
# Keeps workspace/skills on EmptyDir. Restarts netobserv-mcp by default.
#
# Typical flow: ./scripts/clear-openclaw-sessions.sh → hard-refresh → /new → one message.
set -euo pipefail

OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
RESTART_MCP="${RESTART_MCP:-1}"
RESTART_GATEWAY="${RESTART_GATEWAY:-1}"

c_blue=$'\033[1;34m'; c_green=$'\033[1;32m'; c_reset=$'\033[0m'
step() { printf '%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }

command -v oc >/dev/null 2>&1 || { echo "oc required" >&2; exit 1; }

POD="$(oc -n "$OPENCLAW_NS" get pods -l app.kubernetes.io/name=openclaw \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
[[ -n "$POD" ]] || { echo "No Running openclaw pod in $OPENCLAW_NS" >&2; exit 1; }

step "Session count before cleanup"
oc -n "$OPENCLAW_NS" exec "$POD" -c openclaw -- sh -lc '
  HOME=/opt/openclaw OPENCLAW_CONFIG_PATH=/opt/openclaw/config/openclaw.json
  node /app/openclaw.mjs sessions list --json 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin); print(\"sessions:\", d.get(\"count\", \"?\"))
" 2>/dev/null || true
'

step "Wipe session store (transcripts + index)"
oc -n "$OPENCLAW_NS" exec "$POD" -c openclaw -- sh -lc '
  rm -rf /opt/openclaw/.openclaw/agents/main/sessions/*
  mkdir -p /opt/openclaw/.openclaw/agents/main/sessions
'

if [[ "$RESTART_MCP" == "1" ]]; then
  step "Restart netobserv-mcp (clears Streamable HTTP session errors)"
  oc -n "$OPENCLAW_NS" rollout restart deploy/netobserv-mcp
  oc -n "$OPENCLAW_NS" rollout status deploy/netobserv-mcp --timeout=120s
fi

if [[ "$RESTART_GATEWAY" == "1" ]]; then
  step "Reload OpenClaw gateway (keeps EmptyDir workspace)"
  oc -n "$OPENCLAW_NS" exec "$POD" -c openclaw -- kill 1 || true
  oc -n "$OPENCLAW_NS" rollout status deploy/openclaw --timeout=180s
fi

step "Recreate OpenShell sandboxes"
POD="$(oc -n "$OPENCLAW_NS" get pods -l app.kubernetes.io/name=openclaw \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
oc -n "$OPENCLAW_NS" exec "$POD" -c openclaw -- sh -lc '
  HOME=/opt/openclaw OPENCLAW_CONFIG_PATH=/opt/openclaw/config/openclaw.json
  node /app/openclaw.mjs sandbox recreate --all --force 2>/dev/null || true
' || true

ok "Sessions cleared. In Control UI: close extra tabs → hard-refresh → /new → one message."
