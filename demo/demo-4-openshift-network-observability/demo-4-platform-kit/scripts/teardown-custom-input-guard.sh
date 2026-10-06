#!/usr/bin/env bash
# Remove deprecated custom netobserv-input-guard (OpenClaw plugin + sidecar).
# Safe to run when migrating to TrustyAI Guardrails.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
KUBECTL="$(command -v oc || command -v kubectl)"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }

[[ -n "$KUBECTL" ]] || exit 0

"$KUBECTL" -n "$OPENCLAW_NS" delete deploy/netobserv-input-guard svc/netobserv-input-guard \
  configmap/netobserv-input-guard-scripts configmap/netobserv-input-guard-plugin \
  --ignore-not-found 2>/dev/null && ok "Deleted netobserv-input-guard resources" || true

if "$KUBECTL" -n "$OPENCLAW_NS" get configmap openclaw-config >/dev/null 2>&1; then
  raw="$("$KUBECTL" -n "$OPENCLAW_NS" get configmap openclaw-config -o jsonpath='{.data.openclaw\.json}' 2>/dev/null || echo '{}')"
  if echo "$raw" | grep -q netobserv-input-guard; then
    RAW="$raw" python3 - <<'PY' | "$KUBECTL" -n "$OPENCLAW_NS" patch configmap openclaw-config --type merge -p "$(cat)"
import json, os
d = json.loads(os.environ.get("RAW", "{}"))
plugins = d.setdefault("plugins", {})
allow = plugins.get("allow", [])
if isinstance(allow, list):
    plugins["allow"] = [x for x in allow if x != "netobserv-input-guard"]
entries = plugins.get("entries", {})
if isinstance(entries, dict):
    entries.pop("netobserv-input-guard", None)
load = plugins.setdefault("load", {})
paths = load.get("paths", [])
if isinstance(paths, list):
    load["paths"] = [p for p in paths if "netobserv-input-guard" not in p]
    if not load["paths"]:
        load.pop("paths", None)
print(json.dumps({"data": {"openclaw.json": json.dumps(d, indent=2)}}))
PY
    ok "Stripped netobserv-input-guard from openclaw.json (entries + load.paths)"
  fi
fi

# Best-effort: remove init container / volume patches (json patch revert is manual if needed)
warn "If OpenClaw still has stage-input-guard-plugin initContainer, recycle from clean deployment template or re-seed."
