#!/usr/bin/env bash
# Enable OpenClaw Gateway hooks for event-driven AIOps (Grafana → bridge → /hooks/agent → Slack).
#
# Creates openclaw-hooks-token secret, pins minimal hooks config in lab openclaw.json,
# wires OPENCLAW_HOOKS_TOKEN on deploy/openclaw, deploys netobserv-grafana-bridge.
#
# Prerequisites: Slack wired (wire-openclaw-slack.sh) for deliver target.
#
# Env:
#   SLACK_CHANNEL_ID=C0123456789     Slack deliver target (required for bridge)
#   OPENCLAW_HOOKS_TOKEN=...         optional — generate if unset
#   RECYCLE_POD=1|0                  default 1
#   SKIP_KUSTOMIZE=1|0               skip apply -k (reconcile already applied)
#   SKIP_SMOKE=1|0                   skip synthetic POST to bridge
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
LAB_CFG="${LAB_CFG:-$HOME/labs/openshell-on-openshift-lab/manifests/openclaw/config.yaml}"
KUBECTL="$(command -v oc || command -v kubectl)"
RECYCLE_POD="${RECYCLE_POD:-1}"
SKIP_KUSTOMIZE="${SKIP_KUSTOMIZE:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}"
# shellcheck source=resolve-slack-channel.sh
source "$ROOT/scripts/resolve-slack-channel.sh"
resolve_slack_channel_id

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
step() { printf '\n%s==>%s %s\n' "$c_green" "$c_reset" "$*"; }

[[ -n "$KUBECTL" ]] || { echo "oc/kubectl required" >&2; exit 1; }
require_slack_channel_id || {
  warn "SLACK_CHANNEL_ID unset — set env or site.slack_channel_id in site-secrets.local.yaml"
  exit 1
}

step "Ensure openclaw-hooks-token secret"
HOOKS_TOKEN="${OPENCLAW_HOOKS_TOKEN:-}"
if [[ -z "$HOOKS_TOKEN" ]]; then
  if "$KUBECTL" -n "$OPENCLAW_NS" get secret openclaw-hooks-token >/dev/null 2>&1; then
    HOOKS_TOKEN="$("$KUBECTL" -n "$OPENCLAW_NS" get secret openclaw-hooks-token \
      -o jsonpath='{.data.OPENCLAW_HOOKS_TOKEN}' | base64 -d)"
    ok "Reusing existing openclaw-hooks-token"
  else
    HOOKS_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
    "$KUBECTL" -n "$OPENCLAW_NS" create secret generic openclaw-hooks-token \
      --from-literal=OPENCLAW_HOOKS_TOKEN="$HOOKS_TOKEN" \
      --dry-run=client -o yaml | "$KUBECTL" apply -f -
    ok "Created openclaw-hooks-token"
  fi
fi

step "Pin minimal hooks config in lab openclaw.json (plain token — init-safe)"
if [[ ! -f "$LAB_CFG" ]]; then
  warn "Lab config not found: $LAB_CFG"
  exit 1
fi

python3 - "$LAB_CFG" "$HOOKS_TOKEN" <<'PY'
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
token = sys.argv[2]
d = json.loads(p.read_text())

wanted_hooks = {
    "enabled": True,
    "path": "/hooks",
    # Plain string required — SecretRef and ${ENV} break seed-openclaw init validation.
    "token": token,
}

if d.get("hooks") != wanted_hooks:
    d["hooks"] = wanted_hooks
    p.write_text(json.dumps(d, indent=2) + "\n")
    print("patched", p)
else:
    print("unchanged", p)
PY

ok "Updated $LAB_CFG"
if [[ "$SKIP_KUSTOMIZE" != "1" ]]; then
  "$KUBECTL" -n "$OPENCLAW_NS" apply -k "$(dirname "$LAB_CFG")"
  ok "Applied openclaw kustomize"
else
  ok "Skipped kustomize apply (SKIP_KUSTOMIZE=1)"
fi

step "Wire OPENCLAW_HOOKS_TOKEN on deploy/openclaw"
"$KUBECTL" -n "$OPENCLAW_NS" set env deployment/openclaw \
  --from=secret/openclaw-hooks-token 2>/dev/null || true
ok "OPENCLAW_HOOKS_TOKEN from secret openclaw-hooks-token"

step "Deploy Grafana → OpenClaw alert bridge"
"$KUBECTL" -n "$OPENCLAW_NS" create configmap netobserv-grafana-bridge-scripts \
  --from-file=netobserv-grafana-bridge.py="$ROOT/openclaw-skills/netobserv-heal/scripts/netobserv-grafana-bridge.py" \
  --dry-run=client -o yaml | "$KUBECTL" apply -f -
if "$KUBECTL" get clusterspiffeid netobserv-grafana-bridge >/dev/null 2>&1; then
  warn "SPIFFE mTLS active — skip bearer bridge manifest (use wire-openclaw-spiffe.sh to refresh)"
  if [[ -n "$SLACK_CHANNEL_ID" ]]; then
    "$KUBECTL" -n "$OPENCLAW_NS" set env deployment/netobserv-grafana-bridge \
      --containers=bridge SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" 2>/dev/null || true
  fi
else
  python3 - "$ROOT/openclaw-skills/manifests/netobserv-grafana-bridge.yaml" "$SLACK_CHANNEL_ID" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
slack = sys.argv[2]
text = p.read_text()
for needle in ('SLACK_CHANNEL_ID_PLACEHOLDER', 'C0BRJARNPEU'):
    text = text.replace(f'value: "{needle}"', f'value: "{slack}"')
Path("/tmp/netobserv-grafana-bridge.yaml").write_text(text)
PY
  "$KUBECTL" -n "$OPENCLAW_NS" apply -f /tmp/netobserv-grafana-bridge.yaml
  "$KUBECTL" -n "$OPENCLAW_NS" rollout restart deployment/netobserv-grafana-bridge 2>/dev/null || true
  "$KUBECTL" -n "$OPENCLAW_NS" rollout status deployment/netobserv-grafana-bridge --timeout=180s || \
    warn "bridge rollout slow — oc logs -n openclaw deploy/netobserv-grafana-bridge"
fi
ok "netobserv-grafana-bridge ready"

if [[ "$RECYCLE_POD" == "1" ]]; then
  step "Recycle OpenClaw pod (load hooks config)"
  "$KUBECTL" -n "$OPENCLAW_NS" delete pod -l app.kubernetes.io/name=openclaw --wait=false
  "$KUBECTL" -n "$OPENCLAW_NS" rollout status deployment/openclaw --timeout=900s || \
    warn "OpenClaw rollout slow — check: oc -n openclaw get pods; oc logs -c seed-openclaw"
  ok "OpenClaw pod recycled"
fi

step "Smoke-test bridge → OpenClaw /hooks/agent"
if [[ "$SKIP_SMOKE" == "1" ]]; then
  ok "Skipped bridge smoke test (SKIP_SMOKE=1)"
else
"$KUBECTL" -n "$OPENCLAW_NS" run grafana-bridge-smoke --rm -i --restart=Never \
  --image=registry.access.redhat.com/ubi9/ubi-minimal:latest \
  --command -- sh -lc "
command -v curl >/dev/null 2>&1 || microdnf install -y curl >/dev/null 2>&1
code=\$(curl -sS -o /tmp/out -w '%{http_code}' \\
  -X POST 'http://netobserv-grafana-bridge.openclaw.svc.cluster.local:8080/grafana' \\
  -H 'Content-Type: application/json' \\
  -d '{\"status\":\"firing\",\"title\":\"smoke-test\",\"message\":\"Bridge smoke test — ignore during wiring.\",\"commonLabels\":{\"alertname\":\"SmokeTest\"}}')
echo HTTP=\$code
head -c 400 /tmp/out 2>/dev/null || true
test \"\$code\" = '200' || test \"\$code\" = '202'
" && ok "Bridge smoke test accepted" || warn "Bridge smoke test failed — wait for OpenClaw ready and retry"
fi

ok "Hooks ready. Next: ./scripts/wire-grafana-openclaw-alerts.sh"
