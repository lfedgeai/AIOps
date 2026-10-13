#!/usr/bin/env bash
# Ensure OpenClaw Control UI accepts the OpenShift Route browser origin.
# Fixes: "Browser origin not allowed" / origin not allowed (gateway.controlUi.allowedOrigins)
#
# Also sets gateway.controlUi.root to /opt/openclaw/control-ui (writable copy for logo patch).
# Logo fix itself lives in scripts/patch-openclaw-seed-idempotent.sh (NETOBSERV_SEED_HARDENED_v3).
#
# Patches the openshell-on-openshift-lab OpenClaw config + OPENCLAW_PUBLIC_URL,
# re-applies kustomize, restarts the Deployment, then reminds you to re-seed skills.
set -euo pipefail

OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
LAB_ROOT="${LAB_ROOT:-$HOME/labs/openshell-on-openshift-lab}"
CFG="${CFG:-${LAB_ROOT}/manifests/openclaw/config.yaml}"
DEP="${DEP:-${LAB_ROOT}/manifests/openclaw/deployment.yaml}"
RESTART="${RESTART:-1}"
SEED_AFTER="${SEED_AFTER:-1}"
DEMO_ROOT="${DEMO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

c_blue=$'\033[1;34m'; c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_reset=$'\033[0m'
step() { printf '%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }

command -v kubectl >/dev/null 2>&1 || command -v oc >/dev/null 2>&1 || {
  echo "kubectl/oc required" >&2
  exit 1
}
KUBECTL="$(command -v oc || command -v kubectl)"
[[ -f "$CFG" ]] || { echo "config not found: $CFG (set LAB_ROOT)" >&2; exit 1; }

ORIGIN="${ORIGIN:-}"
if [[ -z "$ORIGIN" ]]; then
  HOST="$("$KUBECTL" -n "$OPENCLAW_NS" get route openclaw -o jsonpath='{.spec.host}' 2>/dev/null || true)"
  [[ -n "$HOST" ]] || { echo "could not resolve openclaw Route; set ORIGIN=https://..." >&2; exit 1; }
  ORIGIN="https://${HOST}"
fi
step "Control UI origin: $ORIGIN"

python3 - "$CFG" "$ORIGIN" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
origin = sys.argv[2]
d = json.loads(p.read_text())
cu = d.setdefault("gateway", {}).setdefault("controlUi", {})
base = [
    origin,
    "http://openclaw:18789",
    "http://localhost:18789",
    "http://127.0.0.1:18789",
]
origins = []
for o in base + list(cu.get("allowedOrigins") or []):
    if str(o).startswith("${"):
        continue
    if o not in origins:
        origins.append(o)
cu["allowedOrigins"] = origins
cu.setdefault("dangerouslyDisableDeviceAuth", True)
cu.setdefault("root", "/opt/openclaw/control-ui")
p.write_text(json.dumps(d, indent=2) + "\n")
print("allowedOrigins=", json.dumps(origins))
PY

if [[ -f "$DEP" ]]; then
  python3 - "$DEP" "$ORIGIN" <<'PY'
from pathlib import Path
import re, sys
p = Path(sys.argv[1])
origin = sys.argv[2]
text = p.read_text()
# Replace OPENCLAW_PUBLIC_URL value when present
pat = re.compile(
    r"(name:\s*OPENCLAW_PUBLIC_URL\s*\n\s*value:\s*)(\S+)",
    re.M,
)
m = pat.search(text)
if m:
    text2 = pat.sub(rf"\g<1>{origin}", text, count=1)
    if text2 != text:
        p.write_text(text2)
        print(f"OPENCLAW_PUBLIC_URL -> {origin}")
    else:
        print("OPENCLAW_PUBLIC_URL already set")
else:
    print("OPENCLAW_PUBLIC_URL not found in deployment.yaml (skip)")
PY
fi

step "Apply OpenClaw kustomize"
"$KUBECTL" -n "$OPENCLAW_NS" apply -k "${LAB_ROOT}/manifests/openclaw"

if [[ "$RESTART" == "1" ]]; then
  step "Rollout restart openclaw"
  "$KUBECTL" -n "$OPENCLAW_NS" rollout restart deploy/openclaw
  "$KUBECTL" -n "$OPENCLAW_NS" rollout status deploy/openclaw --timeout=180s
fi

POD=""
for _ in $(seq 1 30); do
  POD="$("$KUBECTL" -n "$OPENCLAW_NS" get pod -l app.kubernetes.io/name=openclaw \
    --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  [[ -n "$POD" ]] && break
  sleep 2
done
if [[ -n "$POD" ]]; then
  "$KUBECTL" -n "$OPENCLAW_NS" exec "$POD" -- sh -lc \
    'HOME=/opt/openclaw OPENCLAW_CONFIG_PATH=/opt/openclaw/config/openclaw.json node /app/openclaw.mjs config get gateway.controlUi.allowedOrigins' \
    || warn "could not verify allowedOrigins in pod (config still patched on disk)"
else
  warn "no running openclaw pod after rollout — skip in-pod config verify"
fi

if [[ "$SEED_AFTER" == "1" && -x "$DEMO_ROOT/scripts/seed-openclaw-netobserv-skills.sh" ]]; then
  step "Re-seed NetObserv skills (emptyDir wiped on restart)"
  "$DEMO_ROOT/scripts/seed-openclaw-netobserv-skills.sh"
else
  warn "Skipped skill seed. After restart run: $DEMO_ROOT/scripts/seed-openclaw-netobserv-skills.sh"
fi

ok "Open $ORIGIN and hard-refresh the Control UI (gateway token from secret openclaw-gateway-token)."
