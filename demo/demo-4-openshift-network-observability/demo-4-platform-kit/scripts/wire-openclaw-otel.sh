#!/usr/bin/env bash
# Phase D — Deploy OTel collector + enable diagnostics-otel on OpenClaw gateway.
#
# Usage:
#   ./scripts/wire-openclaw-otel.sh              # deploy + patch config + restart
#   ./scripts/wire-openclaw-otel.sh status
#
# Env:
#   OPENCLAW_NS=openclaw
#   OTEL_ENDPOINT=http://openclaw-otel-collector.openclaw.svc:4318
#   LAB_CFG=~/labs/openshell-on-openshift-lab/manifests/openclaw/config.yaml
#   OTEL_SAMPLE_RATE=1.0
#   OTEL_FLUSH_MS=15000
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
OTEL_ENDPOINT="${OTEL_ENDPOINT:-http://openclaw-otel-collector.${OPENCLAW_NS}.svc:4318}"
LAB_CFG="${LAB_CFG:-$HOME/labs/openshell-on-openshift-lab/manifests/openclaw/config.yaml}"
OTEL_SAMPLE_RATE="${OTEL_SAMPLE_RATE:-1.0}"
OTEL_FLUSH_MS="${OTEL_FLUSH_MS:-15000}"
KUBECTL="$(command -v oc || command -v kubectl)"
CMD="${1:-all}"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -n "$KUBECTL" ]] || die "oc/kubectl required"

patch_openclaw_json() {
  python3 - "$1" "$OTEL_ENDPOINT" "$OTEL_SAMPLE_RATE" "$OTEL_FLUSH_MS" <<'PY'
import json, sys
from pathlib import Path

path, endpoint, sample_rate, flush_ms = sys.argv[1:5]
p = Path(path)
d = json.loads(p.read_text())

plugins = d.setdefault("plugins", {})
allow = plugins.setdefault("allow", [])
if "diagnostics-otel" not in allow:
    allow.append("diagnostics-otel")
entries = plugins.setdefault("entries", {})
entries["diagnostics-otel"] = {"enabled": True}

diag = d.setdefault("diagnostics", {})
diag["enabled"] = True
otel = diag.setdefault("otel", {})
otel.update({
    "enabled": True,
    "endpoint": endpoint,
    "protocol": "http/protobuf",
    "serviceName": "openclaw-gateway",
    "traces": True,
    "metrics": True,
    "logs": False,
    "sampleRate": float(sample_rate),
    "flushIntervalMs": int(flush_ms),
})

p.write_text(json.dumps(d, indent=2) + "\n")
print("diagnostics-otel enabled →", endpoint)
PY
}

patch_live_configmap() {
  step "Patch openclaw-config ConfigMap (diagnostics-otel)"
  local raw patched
  raw="$("$KUBECTL" -n "$OPENCLAW_NS" get configmap openclaw-config -o jsonpath='{.data.openclaw\.json}' 2>/dev/null || true)"
  [[ -n "$raw" ]] || { warn "openclaw-config missing — apply lab kustomize first"; return 1; }
  patched="$(python3 - "$OTEL_ENDPOINT" "$OTEL_SAMPLE_RATE" "$OTEL_FLUSH_MS" <<'PY'
import json, sys
endpoint, sample_rate, flush_ms = sys.argv[1:4]
d = json.loads(sys.stdin.read())
plugins = d.setdefault("plugins", {})
allow = plugins.setdefault("allow", [])
if "diagnostics-otel" not in allow:
    allow.append("diagnostics-otel")
entries = plugins.setdefault("entries", {})
entries["diagnostics-otel"] = {"enabled": True}
diag = d.setdefault("diagnostics", {})
diag["enabled"] = True
otel = diag.setdefault("otel", {})
otel.update({
    "enabled": True,
    "endpoint": endpoint,
    "protocol": "http/protobuf",
    "serviceName": "openclaw-gateway",
    "traces": True,
    "metrics": True,
    "logs": False,
    "sampleRate": float(sample_rate),
    "flushIntervalMs": int(flush_ms),
})
print(json.dumps(d, indent=2))
PY
<<<"$raw")"
  "$KUBECTL" -n "$OPENCLAW_NS" patch configmap openclaw-config --type merge \
    -p "$(python3 -c 'import json,sys; print(json.dumps({"data":{"openclaw.json":sys.stdin.read()}}))' <<<"$patched")"
  ok "openclaw-config patched"
}

deploy_collector() {
  step "Deploy OTel collector + mini Prometheus (${OPENCLAW_NS})"
  "$KUBECTL" get ns "$OPENCLAW_NS" >/dev/null 2>&1 || die "namespace ${OPENCLAW_NS} missing"
  "$KUBECTL" apply -k "$ROOT/manifests/openclaw-otel"
  "$KUBECTL" -n "$OPENCLAW_NS" rollout status deploy/openclaw-otel-collector --timeout=180s
  "$KUBECTL" -n "$OPENCLAW_NS" rollout status deploy/openclaw-otel-prometheus --timeout=180s
  ok "OTel stack ready"
}

wire_openclaw() {
  if [[ -f "$LAB_CFG" ]]; then
    step "Enable diagnostics-otel in lab config: $LAB_CFG"
    patch_openclaw_json "$LAB_CFG"
    "$KUBECTL" -n "$OPENCLAW_NS" apply -k "$(dirname "$LAB_CFG")"
  else
    patch_live_configmap || true
  fi

  step "Restart OpenClaw gateway (pick up diagnostics-otel)"
  "$KUBECTL" -n "$OPENCLAW_NS" rollout restart deploy/openclaw
  "$KUBECTL" -n "$OPENCLAW_NS" rollout status deploy/openclaw --timeout=240s
  ok "OpenClaw gateway restarted with OTel export"
}

verify_metrics() {
  step "Verify Prometheus sees OpenClaw OTel metrics (after one Control UI turn)"
  local prom_pod
  prom_pod="$("$KUBECTL" -n "$OPENCLAW_NS" get pods -l app.kubernetes.io/name=openclaw-otel-prometheus \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  [[ -n "$prom_pod" ]] || { warn "prometheus pod missing"; return 1; }
  local sample
  sample="$("$KUBECTL" -n "$OPENCLAW_NS" exec "$prom_pod" -- wget -qO- \
    'http://127.0.0.1:9090/api/v1/label/__name__/values' 2>/dev/null | \
    python3 -c 'import json,sys; d=json.load(sys.stdin); print("\n".join(x for x in d.get("data",[]) if "openclaw" in x or "gen_ai" in x))' 2>/dev/null || true)"
  if [[ -n "$sample" ]]; then
    ok "Sample OpenClaw/gen_ai metrics in Prometheus:"
    printf '%s\n' "$sample" | head -12
  else
    warn "No openclaw_* metrics yet — send one message in Control UI, wait ~30s, re-run status"
  fi
}

print_status() {
  step "OTel collector + Prometheus (${OPENCLAW_NS})"
  "$KUBECTL" get deploy,svc -n "$OPENCLAW_NS" 2>/dev/null | grep -E 'openclaw-otel|NAME' || true
  "$KUBECTL" get pods -n "$OPENCLAW_NS" -l 'app.kubernetes.io/name in (openclaw-otel-collector,openclaw-otel-prometheus)' 2>/dev/null || true

  local cfg
  cfg="$("$KUBECTL" -n "$OPENCLAW_NS" get configmap openclaw-config -o jsonpath='{.data.openclaw\.json}' 2>/dev/null || true)"
  if [[ -n "$cfg" ]] && python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print(d.get("diagnostics",{}).get("otel",{}).get("enabled"))' <<<"$cfg" 2>/dev/null | grep -q True; then
    ok "diagnostics.otel.enabled=true on gateway"
  else
    warn "diagnostics.otel not enabled — run: $0 all"
  fi
  verify_metrics || true
}

case "$CMD" in
  status) print_status ;;
  all)
    deploy_collector
    wire_openclaw
    print_status
    cat <<EOF

Phase D OTel wired. Next:
  ./scripts/install-openclaw-otel-grafana.sh grafana   # dashboard + metrics federation
  Send one Control UI message, then open **Network AIOps — Gateway Diagnostics** → **Gateway Diagnostics** row

Docs: docs/OPENCLAW-OTEL-PRESENTER-GUIDE.md
EOF
    ;;
  *)
    echo "usage: $0 [all|status]" >&2
    exit 1
    ;;
esac
