#!/usr/bin/env bash
# Pre-demo sanity — optional cluster checks before Scenario A.
# Grafana metrics sync is folded into demo-a-fast; use this for extra probes or when skipping the e2e driver.
#
# Prefer the unified drift checker for demo pipes:
#   ./scripts/demo-cluster-preflight.sh [check|heal|prepare]
#
# Usage:
#   ./scripts/pre-demo-sanity.sh              # cluster + Grafana probes
#   ./scripts/pre-demo-sanity.sh quick        # skip metrics sync (faster)
#   ./scripts/pre-demo-sanity.sh grafana      # Grafana only
#
# Env: GRAFANA_NS=netobserv-demo  OPENCLAW_NS=openclaw
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="${1:-all}"
GRAFANA_NS="${GRAFANA_NS:-netobserv-demo}"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
KUBECTL="$(command -v oc || command -v kubectl)"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_red=$'\033[1;31m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
fail() { printf '%s[FAIL]%s %s\n' "$c_red" "$c_reset" "$*" >&2; FAILURES=$((FAILURES + 1)); }
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }

[[ -n "$KUBECTL" ]] || { fail "oc/kubectl required"; exit 1; }
FAILURES=0

chmod +x "$ROOT/scripts/install-grafana-network-aiops.sh" \
         "$ROOT/scripts/install-openclaw-otel-grafana.sh" \
         "$ROOT/scripts/sync-grafana-demo-metrics.sh" \
         "$ROOT/scripts/netobserv-e2e-openclaw-test.sh" 2>/dev/null || true

grafana_sanity() {
  local host run_sync="${1:-1}"
  step "Grafana (${GRAFANA_NS})"
  if [[ "$run_sync" == "1" ]]; then
    if "$KUBECTL" get deploy openclaw-otel-prometheus -n "$OPENCLAW_NS" >/dev/null 2>&1; then
      "$ROOT/scripts/sync-grafana-demo-metrics.sh" sync || fail "metrics sync failed"
    else
      warn "OTel Prometheus missing — legacy fix-auth fallback"
      "$ROOT/scripts/install-grafana-network-aiops.sh" fix-auth || fail "fix-auth failed"
    fi
  fi
  host="$("$KUBECTL" get route grafana-network-aiops -n "$GRAFANA_NS" -o jsonpath='{.spec.host}' 2>/dev/null || true)"
  [[ -n "$host" ]] || { fail "Grafana route missing"; return; }
  ok "URL: https://${host}/d/network-aiops-openclaw/network-aiops-e28094-gateway-diagnostics"
  command -v python3 >/dev/null || { warn "python3 missing — skip API probes"; return; }
  python3 - "$host" <<'PY' || { fail "Grafana API probe failed"; return; }
import base64, json, ssl, sys, urllib.error, urllib.request
host = sys.argv[1]
auth = base64.b64encode(b"admin:netobserv-demo").decode()
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
headers = {"Authorization": f"Basic {auth}"}

def health(uid):
    req = urllib.request.Request(f"https://{host}/api/datasources/uid/{uid}/health", method="POST", headers=headers)
    return json.load(urllib.request.urlopen(req, context=ctx))

def series_values(frame):
    values = (frame.get("data") or {}).get("values") or []
    if len(values) > 1:
        return values[1]
    return values[0] if values else []

h = health("prometheus-openclaw-otel")
if h.get("status") != "OK":
    print("health prometheus-openclaw-otel:", h, file=sys.stderr)
    sys.exit(1)
print("health prometheus-openclaw-otel: OK")

body = json.dumps({"queries":[
  {"refId":"N","datasource":{"type":"prometheus","uid":"prometheus-openclaw-otel"},
   "expr":"sum(rate(netobserv_namespace_flows_total{SrcK8S_Namespace=\"todo-demo\"}[2m]))",
   "range":True,"instant":False,"intervalMs":15000,"maxDataPoints":10},
  {"refId":"O","datasource":{"type":"prometheus","uid":"prometheus-openclaw-otel"},
   "expr":"sum(rate(openclaw_openclaw_message_processed_total[5m]))",
   "range":True,"instant":False,"intervalMs":15000,"maxDataPoints":10},
],"from":"now-6h","to":"now"}).encode()
req = urllib.request.Request(f"https://{host}/api/ds/query", data=body, method="POST",
    headers={**headers, "Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, context=ctx))
for k, label in (("N", "NetObserv todo-demo flows"), ("O", "Gateway Diagnostics messages")):
    res = r.get("results", {}).get(k) or {}
    if res.get("error"):
        print(f"query {k}: {res['error']}", file=sys.stderr); sys.exit(1)
    frames = res.get("frames") or []
    if not frames:
        print(f"query {k}: no frames", file=sys.stderr); sys.exit(1)
    vals = series_values(frames[0])
    nz = sum(1 for x in vals if x is not None and float(x) > 0)
    last = vals[-1] if vals else 0
    print(f"query {label}: {nz} nonzero points (last={last})")
    if k == "N" and nz == 0:
        print("NetObserv panel may look empty — run demo-a-fast or widen time range to Last 6h", file=sys.stderr)
    if k == "O" and nz == 0:
        print("Gateway OTel messages empty until Control UI warm-up (expected)", file=sys.stderr)
PY
  ok "Grafana datasource + panel queries OK"
}

cluster_sanity() {
  step "Cluster prerequisites"
  "$KUBECTL" whoami >/dev/null || fail "not logged in to cluster"
  if [[ -x "$ROOT/scripts/fix-netobserv-operator-rbac.sh" ]]; then
    "$ROOT/scripts/fix-netobserv-operator-rbac.sh" apply >/dev/null 2>&1 \
      && ok "NetObserv operator Secret RBAC" \
      || warn "NetObserv operator RBAC not applied (operator still installing?)"
  fi
  if "$KUBECTL" get flowcollector cluster >/dev/null 2>&1 \
      || "$KUBECTL" get flowcollector -n netobserv >/dev/null 2>&1; then
    ok "FlowCollector present"
  else
    fail "FlowCollector missing"
  fi
  if "$KUBECTL" get deploy -n openclaw openclaw >/dev/null 2>&1; then
    ok "OpenClaw deployment present"
  else
    fail "OpenClaw deployment missing"
  fi
  step "OpenClaw / MCP health"
  "$ROOT/scripts/netobserv-e2e-openclaw-test.sh" status 2>&1 | tail -8 || warn "e2e status had warnings"
}

otel_sanity() {
  step "Gateway Diagnostics (OTel) stack"
  if "$KUBECTL" get deploy -n "$OPENCLAW_NS" openclaw-otel-collector >/dev/null 2>&1; then
    "$ROOT/scripts/install-openclaw-otel-grafana.sh" status 2>&1 | tail -6 || warn "otel status warnings"
  else
    warn "OTel not installed — skip (optional: ./scripts/install-openclaw-otel-grafana.sh all)"
  fi
}

case "$CMD" in
  grafana) grafana_sanity 1 ;;
  quick)
    grafana_sanity 0
    cluster_sanity
    ;;
  preflight|cluster)
    exec "$ROOT/scripts/demo-cluster-preflight.sh" "${@:2}"
    ;;
  all|*)
    if [[ -x "$ROOT/scripts/demo-cluster-preflight.sh" ]]; then
      note "Running demo-cluster-preflight (core pipes) — use pre-demo-sanity.sh grafana for panel probes"
      SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}" SKIP_METRICS=1 \
        "$ROOT/scripts/demo-cluster-preflight.sh" check || FAILURES=$((FAILURES + 1))
    fi
    cluster_sanity
    grafana_sanity 1
    otel_sanity
    ;;
esac

step "Summary"
if [[ "$FAILURES" -eq 0 ]]; then
  ok "Pre-demo sanity passed"
  cat <<EOF

Next:
  ./scripts/netobserv-e2e-openclaw-test.sh demo-a-fast   # includes Grafana metrics sync

Grafana: set time range **Last 6 hours** if NetObserv rows look sparse.
EOF
  exit 0
fi
fail "${FAILURES} check(s) failed — fix above, then re-run: $0"
exit 1
