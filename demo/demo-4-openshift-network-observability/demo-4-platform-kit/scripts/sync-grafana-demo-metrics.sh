#!/usr/bin/env bash
# Federate NetObserv metrics from Thanos into openclaw-otel-prometheus so Grafana
# uses a single datasource (prometheus-openclaw-otel) — no manual fix-auth for panels.
#
# Usage:
#   ./scripts/sync-grafana-demo-metrics.sh sync    # refresh token + restart if needed (default)
#   ./scripts/sync-grafana-demo-metrics.sh status  # verify federation in mini Prometheus
#
# Env: GRAFANA_NS=netobserv-demo  OPENCLAW_NS=openclaw
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="${1:-sync}"
GRAFANA_NS="${GRAFANA_NS:-netobserv-demo}"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
KUBECTL="$(command -v oc || command -v kubectl)"
OTEL_MANIFESTS="$ROOT/manifests/openclaw-otel"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_blue=$'\033[1;34m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -n "$KUBECTL" ]] || die "oc/kubectl required"

ensure_reader_token() {
  if ! "$KUBECTL" get secret grafana-prometheus-token -n "$GRAFANA_NS" >/dev/null 2>&1; then
    step "Ensure grafana-prometheus-reader token (${GRAFANA_NS})"
    "$KUBECTL" apply -n "$GRAFANA_NS" -f "$ROOT/manifests/grafana-network-aiops/03-prometheus-access-rbac.yaml"
    local i
    for i in $(seq 1 12); do
      "$KUBECTL" get secret grafana-prometheus-token -n "$GRAFANA_NS" >/dev/null 2>&1 && break
      sleep 2
    done
    "$KUBECTL" get secret grafana-prometheus-token -n "$GRAFANA_NS" >/dev/null 2>&1 || \
      die "grafana-prometheus-token not created — check SA token controller"
  fi
}

ensure_otel_prometheus() {
  if ! "$KUBECTL" get deploy openclaw-otel-prometheus -n "$OPENCLAW_NS" >/dev/null 2>&1; then
    die "openclaw-otel-prometheus missing — run: ./scripts/install-openclaw-otel-grafana.sh all"
  fi
  step "Apply demo Prometheus config (NetObserv federation + OTel scrape)"
  "$KUBECTL" apply -n "$OPENCLAW_NS" \
    -f "$OTEL_MANIFESTS/03-prometheus-config.yaml" \
    -f "$OTEL_MANIFESTS/04-prometheus.yaml"
  reload_prometheus_config
}

reload_prometheus_config() {
  local prom_pod
  prom_pod="$("$KUBECTL" -n "$OPENCLAW_NS" get pods -l app.kubernetes.io/name=openclaw-otel-prometheus \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  [[ -n "$prom_pod" ]] || return 0
  if "$KUBECTL" -n "$OPENCLAW_NS" exec "$prom_pod" -- wget -qO- --post-data='' \
    'http://127.0.0.1:9090/-/reload' >/dev/null 2>&1; then
    ok "Prometheus config reloaded"
  else
    warn "Prometheus reload failed — restarting pod"
    restart_otel_prometheus
  fi
}

restart_otel_prometheus() {
  "$KUBECTL" rollout restart deploy/openclaw-otel-prometheus -n "$OPENCLAW_NS"
  "$KUBECTL" rollout status deploy/openclaw-otel-prometheus -n "$OPENCLAW_NS" --timeout=180s
}

config_changed() {
  local prom_pod live_hash cm_hash
  prom_pod="$("$KUBECTL" -n "$OPENCLAW_NS" get pods -l app.kubernetes.io/name=openclaw-otel-prometheus \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  [[ -n "$prom_pod" ]] || return 1
  live_hash="$("$KUBECTL" -n "$OPENCLAW_NS" exec "$prom_pod" -- md5sum /etc/prometheus/prometheus.yml 2>/dev/null \
    | awk '{print $1}' || echo none)"
  cm_hash="$("$KUBECTL" -n "$OPENCLAW_NS" get cm openclaw-otel-prometheus-config \
    -o jsonpath='{.data.prometheus\.yml}' 2>/dev/null | md5sum | awk '{print $1}')"
  [[ "$live_hash" != "$cm_hash" ]]
}

sync_thanos_token() {
  ensure_reader_token
  local token old_hash new_hash
  token="$("$KUBECTL" get secret grafana-prometheus-token -n "$GRAFANA_NS" \
    -o jsonpath='{.data.token}' | base64 -d)"
  [[ -n "$token" ]] || die "empty Thanos reader token"

  old_hash="$("$KUBECTL" get secret thanos-federate-token -n "$OPENCLAW_NS" \
    -o jsonpath='{.data.token}' 2>/dev/null | md5sum | awk '{print $1}' || echo none)"

  "$KUBECTL" create secret generic thanos-federate-token -n "$OPENCLAW_NS" \
    --from-literal=token="$token" --dry-run=client -o yaml | "$KUBECTL" apply -f -

  new_hash="$("$KUBECTL" get secret thanos-federate-token -n "$OPENCLAW_NS" \
    -o jsonpath='{.data.token}' | md5sum | awk '{print $1}')"

  if [[ "$old_hash" != "$new_hash" ]]; then
    step "Thanos token changed — restart demo Prometheus"
    restart_otel_prometheus
  elif config_changed; then
    step "Prometheus config drift — restart demo Prometheus"
    restart_otel_prometheus
  else
    ok "Thanos federate token unchanged"
    reload_prometheus_config
  fi
}

repair_dashboard_datasource() {
  local ds manifest_ver cr_ver
  manifest_ver="$(python3 -c 'import json,re,sys; t=open(sys.argv[1]).read(); m=re.search(r"\"version\":\s*(\d+)", t); print(m.group(1) if m else "0")' \
    "$ROOT/manifests/grafana-network-aiops/06-dashboard-network-aiops.yaml" 2>/dev/null || echo 0)"
  cr_ver="$("$KUBECTL" get grafanadashboard network-aiops-openclaw -n "$GRAFANA_NS" \
    -o jsonpath='{.spec.json}' 2>/dev/null | \
    python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("version",0))' 2>/dev/null || echo 0)"
  ds="$("$KUBECTL" get grafanadashboard network-aiops-openclaw -n "$GRAFANA_NS" \
    -o jsonpath='{.spec.json}' 2>/dev/null | \
    python3 -c 'import json,sys; d=json.load(sys.stdin); p=next((x for x in d.get("panels",[]) if x.get("type")=="timeseries" and "todo-demo" in x.get("title","").lower()), {}); print((p.get("datasource") or {}).get("uid",""))' 2>/dev/null || true)"
  if [[ "$ds" == "prometheus-openclaw-otel" && "${cr_ver:-0}" == "${manifest_ver:-0}" ]]; then
    ok "Dashboard current (version ${cr_ver}, datasource prometheus-openclaw-otel)"
    return 0
  fi
  if [[ "$ds" != "prometheus-openclaw-otel" ]]; then
    warn "Dashboard datasource drift (${ds:-missing}) — re-applying Network AIOps dashboard CR"
  else
    warn "Dashboard version drift (cluster=${cr_ver:-?} manifest=${manifest_ver:-?}) — re-applying Network AIOps dashboard CR"
  fi
  "$KUBECTL" delete -n "$GRAFANA_NS" grafanadashboard network-aiops-openclaw --ignore-not-found
  sleep 2
  "$KUBECTL" apply -n "$GRAFANA_NS" \
    -f "$ROOT/manifests/grafana-network-aiops/06-dashboard-network-aiops.yaml" \
    -f "$ROOT/manifests/grafana-network-aiops/08-datasource-openclaw-otel.yaml"
  sleep 8
  ds="$("$KUBECTL" get grafanadashboard network-aiops-openclaw -n "$GRAFANA_NS" \
    -o jsonpath='{.spec.json}' 2>/dev/null | \
    python3 -c 'import json,sys; d=json.load(sys.stdin); p=next((x for x in d.get("panels",[]) if x.get("type")=="timeseries" and "todo-demo" in x.get("title","").lower()), {}); print((p.get("datasource") or {}).get("uid",""))' 2>/dev/null || true)"
  cr_ver="$("$KUBECTL" get grafanadashboard network-aiops-openclaw -n "$GRAFANA_NS" \
    -o jsonpath='{.spec.json}' 2>/dev/null | \
    python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("version",0))' 2>/dev/null || echo 0)"
  if [[ "$ds" == "prometheus-openclaw-otel" && "${cr_ver:-0}" == "${manifest_ver:-0}" ]]; then
    ok "Dashboard repaired → version ${cr_ver}, prometheus-openclaw-otel"
    return 0
  fi
  warn "Dashboard still on ${ds:-unknown} version ${cr_ver:-?} — run: ./scripts/install-grafana-network-aiops.sh fix-auth"
  return 1
}

verify_federation() {
  local prom_pod sample target_health
  prom_pod="$("$KUBECTL" -n "$OPENCLAW_NS" get pods -l app.kubernetes.io/name=openclaw-otel-prometheus \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  [[ -n "$prom_pod" ]] || { warn "prometheus pod missing"; return 1; }

  target_health="$("$KUBECTL" -n "$OPENCLAW_NS" exec "$prom_pod" -- wget -qO- \
    'http://127.0.0.1:9090/api/v1/targets' 2>/dev/null | \
    python3 -c 'import json,sys; d=json.load(sys.stdin); t=[x for x in d.get("data",{}).get("activeTargets",[]) if x.get("labels",{}).get("job")=="federate-netobserv"]; print(t[0].get("health","missing") if t else "missing")' 2>/dev/null || echo missing)"
  if [[ "$target_health" != "up" ]]; then
    warn "federate-netobserv target is ${target_health} — check prometheus.yml uses prometheus-k8s.openshift-monitoring.svc:9091 (not thanos-querier /federate 404)"
    return 1
  fi

  sample="$("$KUBECTL" -n "$OPENCLAW_NS" exec "$prom_pod" -- wget -qO- \
    'http://127.0.0.1:9090/api/v1/query?query=count(netobserv_namespace_flows_total)' 2>/dev/null | \
    python3 -c 'import json,sys; d=json.load(sys.stdin); r=d.get("data",{}).get("result",[]); print(r[0]["value"][1] if r else "0")' 2>/dev/null || echo 0)"

  if [[ "${sample:-0}" != "0" ]] && [[ -n "${sample:-}" ]]; then
    ok "NetObserv metrics federated into demo Prometheus (series count=${sample})"
    return 0
  fi

  warn "No netobserv_* series in demo Prometheus yet — FlowCollector may still be warming"
  return 1
}

cmd_sync() {
  step "Sync Grafana demo metrics (single datasource, no fix-auth)"
  ensure_otel_prometheus
  sync_thanos_token
  verify_federation || true
  repair_dashboard_datasource || true
  ok "Grafana NetObserv panels use prometheus-openclaw-otel (no Thanos Bearer in Grafana)"
}

cmd_status() {
  ensure_otel_prometheus
  verify_federation
  local prom_pod otel
  prom_pod="$("$KUBECTL" -n "$OPENCLAW_NS" get pods -l app.kubernetes.io/name=openclaw-otel-prometheus \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  otel="$("$KUBECTL" -n "$OPENCLAW_NS" exec "$prom_pod" -- wget -qO- \
    'http://127.0.0.1:9090/api/v1/label/__name__/values' 2>/dev/null | \
    python3 -c 'import json,sys; d=json.load(sys.stdin); n=[x for x in d.get("data",[]) if x.startswith("netobserv_") or x.startswith("openclaw_")]; print("\n".join(n[:20]))' 2>/dev/null || true)"
  [[ -n "$otel" ]] && printf '%s\n' "$otel"
}

case "$CMD" in
  sync) cmd_sync ;;
  status) cmd_status ;;
  *)
    echo "usage: $0 [sync|status]" >&2
    exit 1
    ;;
esac
