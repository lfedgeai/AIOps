#!/usr/bin/env bash
# Grafana Operator + Network AIOps — Gateway Diagnostics dashboard (flows/packets + agent economics).
#
# Usage:
#   ./scripts/install-grafana-network-aiops.sh status
#   ./scripts/install-grafana-network-aiops.sh install   # same as all
#   ./scripts/install-grafana-network-aiops.sh all          # install + wait + datasource auth
#   ./scripts/install-grafana-network-aiops.sh recover    # restart wedged Grafana pod (503 / readiness)
#   ./scripts/install-grafana-network-aiops.sh cleanup-default  # remove stale OG/sub from default ns (early install mistake)
#
# Notes:
#   - Applies operator first, then Grafana CRs (CRDs must exist).
#   - Always uses -n netobserv-demo (never default namespace).
#   - Grafana SQLite DB on PVC network-aiops-pvc (2Gi) — alerts/contact points survive restart.
#   - First install or PVC migration: run wire-grafana-openclaw-alerts.sh after apply.
#   - Lab sizing: 512Mi request / 2Gi limit (auto-refresh + unified alerting); still close refresh during event-AIOps.
#   - Post-install API sync sets Prometheus Authorization header (operator valuesFrom alone is unreliable).
#
# Env:
#   GRAFANA_NS=netobserv-demo
#   GRAFANA_CSV=grafana-operator.v5.24.0
#   WAIT_GRAFANA_SEC=600
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFESTS="$ROOT/manifests/grafana-network-aiops"
KUBECTL="$(command -v oc || command -v kubectl)"
CMD="${1:-all}"
GRAFANA_NS="${GRAFANA_NS:-netobserv-demo}"
GRAFANA_CSV="${GRAFANA_CSV:-grafana-operator.v5.24.0}"
WAIT_GRAFANA_SEC="${WAIT_GRAFANA_SEC:-600}"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -n "$KUBECTL" ]] || die "oc/kubectl required"

cleanup_stale_default() {
  step "Remove stale Grafana resources from default namespace (if any)"
  local removed=0
  for kind_name in \
    "route/grafana-network-aiops" \
    "grafana/network-aiops" \
    "grafanadatasource/prometheus-netobserv" \
    "grafanadashboard/network-aiops-openclaw" \
    "subscription/grafana-operator" \
    "operatorgroup/grafana-operator-group" \
    "pvc/network-aiops-pvc" \
    "serviceaccount/grafana-prometheus-reader"; do
    if "$KUBECTL" get -n default "$kind_name" >/dev/null 2>&1; then
      "$KUBECTL" delete -n default "$kind_name" --ignore-not-found
      removed=1
    fi
  done
  if "$KUBECTL" get csv grafana-operator.v5.24.0 -n default >/dev/null 2>&1; then
    "$KUBECTL" delete csv grafana-operator.v5.24.0 -n default --ignore-not-found
    removed=1
  fi
  # SA token + dockercfg secrets left when RBAC was applied without -n
  for sec in grafana-prometheus-token grafana-prometheus-reader-dockercfg; do
    if "$KUBECTL" get secret "$sec" -n default >/dev/null 2>&1; then
      "$KUBECTL" delete secret "$sec" -n default --ignore-not-found
      removed=1
    fi
  done
  # Orphan secrets use hashed suffix — delete by prefix
  while IFS= read -r sec; do
    [[ -n "$sec" ]] || continue
    "$KUBECTL" delete secret "$sec" -n default --ignore-not-found
    removed=1
  done < <("$KUBECTL" get secret -n default -o name 2>/dev/null | sed -n 's|secret/||p' | grep '^grafana-prometheus-reader-dockercfg' || true)
  if [[ "$removed" == "1" ]]; then
    ok "Stale default-namespace Grafana resources removed"
    sleep 3
  else
    ok "No stale Grafana resources in default namespace"
  fi
}

wait_grafana_csv() {
  step "Wait Grafana Operator CSV (timeout ${WAIT_GRAFANA_SEC}s)"
  local end=$((SECONDS + WAIT_GRAFANA_SEC))
  while (( SECONDS < end )); do
    local phase
    phase="$("$KUBECTL" get csv "$GRAFANA_CSV" -n "$GRAFANA_NS" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    if [[ "$phase" == "Succeeded" ]]; then
      ok "CSV $GRAFANA_CSV Succeeded"
      return 0
    fi
    sleep 10
  done
  warn "CSV $GRAFANA_CSV not Succeeded — oc get csv -n $GRAFANA_NS"
  return 1
}

wait_grafana_cr() {
  step "Wait Grafana instance deployment"
  local end=$((SECONDS + WAIT_GRAFANA_SEC))
  while (( SECONDS < end )); do
    local phase
    phase="$("$KUBECTL" get grafana network-aiops -n "$GRAFANA_NS" -o jsonpath='{.status.stage}' 2>/dev/null || true)"
    if [[ "$phase" == "complete" ]]; then
      ok "Grafana CR stage=complete"
      return 0
    fi
    sleep 10
  done
  warn "Grafana CR not complete — oc get grafana network-aiops -n $GRAFANA_NS -o yaml"
  "$KUBECTL" get deploy -n "$GRAFANA_NS" -l app=grafana 2>/dev/null || true
  return 1
}

wait_prometheus_token() {
  step "Wait Prometheus reader token secret"
  local end=$((SECONDS + 120))
  while (( SECONDS < end )); do
    if "$KUBECTL" get secret grafana-prometheus-token -n "$GRAFANA_NS" -o jsonpath='{.data.token}' 2>/dev/null | grep -q .; then
      ok "grafana-prometheus-token populated"
      return 0
    fi
    sleep 5
  done
  warn "Token secret empty — oc describe secret grafana-prometheus-token -n $GRAFANA_NS"
  return 1
}

ensure_prometheus_auth_secret() {
  step "Build grafana-prometheus-auth (Bearer token for Thanos)"
  local raw
  raw="$("$KUBECTL" get secret grafana-prometheus-token -n "$GRAFANA_NS" -o jsonpath='{.data.token}' 2>/dev/null | base64 -d 2>/dev/null || true)"
  [[ -n "$raw" ]] || { warn "missing SA token — skip auth secret"; return 1; }
  "$KUBECTL" create secret generic grafana-prometheus-auth -n "$GRAFANA_NS" \
    --from-literal=authorization="Bearer ${raw}" \
    --dry-run=client -o yaml | "$KUBECTL" apply -f -
  ok "grafana-prometheus-auth ready"
}

sync_prometheus_datasource_auth() {
  step "Sync Prometheus Bearer token into Grafana datasource (API)"
  local host raw bearer attempt
  host="$("$KUBECTL" get route grafana-network-aiops -n "$GRAFANA_NS" -o jsonpath='{.spec.host}' 2>/dev/null || true)"
  [[ -n "$host" ]] || { warn "no Grafana route — skip datasource auth sync"; return 1; }
  raw="$("$KUBECTL" get secret grafana-prometheus-token -n "$GRAFANA_NS" -o jsonpath='{.data.token}' 2>/dev/null | base64 -d 2>/dev/null || true)"
  [[ -n "$raw" ]] || { warn "no prometheus SA token — skip datasource auth sync"; return 1; }
  bearer="Bearer ${raw}"
  command -v python3 >/dev/null || { warn "python3 required for datasource auth sync"; return 1; }

  for attempt in 1 2 3; do
    if python3 - "$host" "$bearer" <<'PY'
import base64, json, ssl, sys, urllib.error, urllib.request
host, bearer = sys.argv[1], sys.argv[2]
auth = base64.b64encode(b"admin:netobserv-demo").decode()
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
payload = {
    "name": "Prometheus",
    "type": "prometheus",
    "access": "proxy",
    "uid": "prometheus-netobserv",
    "url": "https://thanos-querier.openshift-monitoring.svc:9091",
    "isDefault": True,
    "jsonData": {
        "tlsSkipVerify": True,
        "timeInterval": "30s",
        "httpHeaderName1": "Authorization",
    },
    "secureJsonData": {"httpHeaderValue1": bearer},
}
body = json.dumps(payload).encode()
headers = {"Content-Type": "application/json", "Authorization": f"Basic {auth}"}
get = urllib.request.Request(
    f"https://{host}/api/datasources/uid/prometheus-netobserv",
    headers=headers,
)
exists = True
try:
    urllib.request.urlopen(get, context=ctx)
except urllib.error.HTTPError as exc:
    if exc.code == 404:
        exists = False
    else:
        sys.stderr.write(exc.read().decode())
        sys.exit(exc.code)
method = "PUT" if exists else "POST"
url = (
    f"https://{host}/api/datasources/uid/prometheus-netobserv"
    if exists
    else f"https://{host}/api/datasources"
)
req = urllib.request.Request(url, data=body, method=method, headers=headers)
try:
    urllib.request.urlopen(req, context=ctx)
except urllib.error.HTTPError as exc:
    sys.stderr.write(exc.read().decode())
    sys.exit(exc.code)
health = urllib.request.Request(
    f"https://{host}/api/datasources/uid/prometheus-netobserv/health",
    method="POST",
    headers=headers,
)
try:
    resp = urllib.request.urlopen(health, context=ctx)
except urllib.error.HTTPError as exc:
    sys.stderr.write(exc.read().decode())
    sys.exit(exc.code)
print(resp.read().decode())
PY
    then
      ok "Prometheus datasource health OK"
      return 0
    fi
    warn "datasource auth sync attempt ${attempt}/3 failed — retrying in 5s"
    sleep 5
  done
  warn "datasource auth sync failed after 3 attempts"
  return 1
}

force_resync_grafana_crds() {
  step "Force Grafana operator resync (dashboard CR only — keep datasource to avoid auth wipe)"
  "$KUBECTL" delete -n "$GRAFANA_NS" grafanadashboard network-aiops-openclaw --ignore-not-found
  sleep 3
  "$KUBECTL" apply -n "$GRAFANA_NS" \
    -f "$MANIFESTS/06-dashboard-network-aiops.yaml" \
    -f "$MANIFESTS/08-datasource-openclaw-otel.yaml"
  sleep 8
  sync_prometheus_datasource_auth || warn "datasource auth sync after dashboard resync failed — re-run: $0 fix-auth"
}

grafana_datasource_missing() {
  local host
  host="$("$KUBECTL" get route grafana-network-aiops -n "$GRAFANA_NS" -o jsonpath='{.spec.host}' 2>/dev/null || true)"
  [[ -n "$host" ]] || return 0
  local code
  code="$(curl -sk -o /dev/null -w '%{http_code}' -u admin:netobserv-demo \
    "https://${host}/api/datasources/uid/prometheus-netobserv" 2>/dev/null || echo 000)"
  [[ "$code" == "404" || "$code" == "000" ]]
}

print_status() {
  step "Grafana Operator ($GRAFANA_NS)"
  "$KUBECTL" get sub,csv -n "$GRAFANA_NS" 2>/dev/null | grep -E 'grafana|NAME' || warn "Grafana subscription not installed"
  step "Grafana instance + dashboard"
  "$KUBECTL" get grafana,grafanadatasource,grafanadashboard -n "$GRAFANA_NS" 2>/dev/null || true
  "$KUBECTL" get route grafana-network-aiops -n "$GRAFANA_NS" 2>/dev/null || true
  "$KUBECTL" get pvc network-aiops-pvc -n "$GRAFANA_NS" 2>/dev/null || warn "PVC network-aiops-pvc missing — Grafana DB ephemeral"
  "$KUBECTL" get deploy/network-aiops-deployment -n "$GRAFANA_NS" 2>/dev/null || true
  "$KUBECTL" get pods -n "$GRAFANA_NS" -l app.kubernetes.io/instance=network-aiops 2>/dev/null || \
    "$KUBECTL" get pods -n "$GRAFANA_NS" --field-selector=status.phase=Running 2>/dev/null | grep -E 'network-aiops|NAME' || true

  local host
  host="$("$KUBECTL" get route grafana-network-aiops -n "$GRAFANA_NS" -o jsonpath='{.spec.host}' 2>/dev/null || true)"
  if [[ -n "$host" ]]; then
    ok "Grafana URL: https://${host}/"
    ok "Login: admin / netobserv-demo (lab default — change for production)"
    ok "Dashboard: Network AIOps — Gateway Diagnostics"
  fi
}

install_grafana() {
  cleanup_stale_default

  step "Ensure namespace $GRAFANA_NS"
  "$KUBECTL" get ns "$GRAFANA_NS" >/dev/null 2>&1 || \
    "$KUBECTL" create ns "$GRAFANA_NS"

  step "Apply Grafana Operator subscription + Prometheus RBAC"
  "$KUBECTL" apply -n "$GRAFANA_NS" \
    -f "$MANIFESTS/01-operatorgroup.yaml" \
    -f "$MANIFESTS/02-subscription.yaml" \
    -f "$MANIFESTS/03-prometheus-access-rbac.yaml"

  wait_grafana_csv || true

  step "Wait Grafana Operator CRDs"
  local crd_end=$((SECONDS + 300))
  while (( SECONDS < crd_end )); do
    if "$KUBECTL" get crd grafanas.grafana.integreatly.org >/dev/null 2>&1; then
      ok "Grafana CRDs available"
      break
    fi
    sleep 5
  done
  "$KUBECTL" wait --for=condition=Established crd/grafanas.grafana.integreatly.org --timeout=120s 2>/dev/null || \
    warn "Grafana CRD not Established — continuing"

  wait_prometheus_token || true
  ensure_prometheus_auth_secret || true

  step "Apply Grafana instance, datasource, dashboard, route"
  "$KUBECTL" apply -n "$GRAFANA_NS" \
    -f "$MANIFESTS/04-grafana.yaml" \
    -f "$MANIFESTS/05-datasource-prometheus.yaml" \
    -f "$MANIFESTS/06-dashboard-network-aiops.yaml" \
    -f "$MANIFESTS/07-route.yaml" \
    -f "$MANIFESTS/08-datasource-openclaw-otel.yaml"

  sleep 15
  wait_grafana_cr || true
  sync_prometheus_datasource_auth || warn "datasource auth sync failed — run: $0 fix-auth"

  ok "Grafana Phase C install applied"
  print_status
  cat <<EOF

Presenter notes:
  - Inject Kraken fault first, then open dashboard — RTT panel on todo-demo should spike
  - Before the room: ./scripts/netobserv-e2e-openclaw-test.sh demo-a-fast  (or demo-a on cold start)
  - Grafana wedged (503): ./scripts/install-grafana-network-aiops.sh recover
  - Legacy (Grafana only): ./scripts/install-grafana-network-aiops.sh fix-auth
  - Set time range Last 6 hours if panels look empty after an earlier demo run
  - Correlate with Console Network Traffic + MLflow Runs (Acts 5/8)
  - RHCL OAuth front door for Grafana is optional (Phase C uses lab admin login)

Docs: docs/GRAFANA-PRESENTER-GUIDE.md
Phase D (OpenClaw OTel panels): ./scripts/install-openclaw-otel-grafana.sh all
EOF
}

cmd_recover() {
  step "Restart Grafana deployment (auto-refresh / SQLite wedge)"
  "$KUBECTL" -n "$GRAFANA_NS" rollout restart deploy/network-aiops-deployment
  "$KUBECTL" -n "$GRAFANA_NS" rollout status deploy/network-aiops-deployment --timeout=300s
  local host
  host="$("$KUBECTL" get route grafana-network-aiops -n "$GRAFANA_NS" -o jsonpath='{.spec.host}' 2>/dev/null || true)"
  if [[ -n "$host" ]] && command -v curl >/dev/null 2>&1; then
    curl -sk -o /dev/null -w "Grafana /api/health HTTP=%{http_code}\n" "https://${host}/api/health" || true
  fi
  if [[ -x "$ROOT/scripts/sync-grafana-demo-metrics.sh" ]]; then
    "$ROOT/scripts/sync-grafana-demo-metrics.sh" sync || warn "metrics sync failed"
  fi
  if [[ -x "$ROOT/scripts/wire-grafana-openclaw-alerts.sh" ]]; then
    "$ROOT/scripts/wire-grafana-openclaw-alerts.sh" status 2>/dev/null || \
      warn "alert check skipped — run wire-grafana-openclaw-alerts.sh if event-AIOps missing"
  fi
  ok "Grafana recover complete — disable dashboard auto-refresh during demos"
}

case "$CMD" in
  status) print_status ;;
  recover) cmd_recover ;;
  fix-auth)
    ensure_prometheus_auth_secret || true
    if grafana_datasource_missing; then
      warn "Prometheus datasource missing in Grafana — re-applying operator CRs"
      force_resync_grafana_crds
    fi
    sync_prometheus_datasource_auth
    ;;
  cleanup-default) cleanup_stale_default ;;
  install) install_grafana ;;
  all) install_grafana ;;
  *)
    echo "usage: $0 [status|install|all|fix-auth|recover|cleanup-default]" >&2
    exit 1
    ;;
esac
