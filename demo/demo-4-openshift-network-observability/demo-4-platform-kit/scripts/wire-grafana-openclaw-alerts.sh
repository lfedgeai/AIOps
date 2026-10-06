#!/usr/bin/env bash
# Provision Grafana unified alert → netobserv-grafana-bridge → OpenClaw /hooks/agent
#
# Alert: avg NetObserv RTT for todo-demo > threshold for ALERT_FOR.
# Contact point: webhook to netobserv-grafana-bridge → OpenClaw /hooks/agent
#
# Prerequisites:
#   - Grafana installed (install-grafana-network-aiops.sh all)
#   - OTel Prometheus + metrics sync (install-openclaw-otel-grafana.sh / sync-grafana-demo-metrics.sh)
#   - wire-openclaw-hooks.sh (hooks + Slack deliver mapping)
#
# Usage:
#   ./scripts/wire-grafana-openclaw-alerts.sh
#   ./scripts/wire-grafana-openclaw-alerts.sh status
#
# Env:
#   GRAFANA_NS=netobserv-demo
#   RTT_THRESHOLD_SEC=0.35        fires when avg RTT > 350ms (Kraken 800ms demo clears easily)
#   ALERT_FOR=2m
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRAFANA_NS="${GRAFANA_NS:-netobserv-demo}"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
KUBECTL="$(command -v oc || command -v kubectl)"
CMD="${1:-all}"
RTT_THRESHOLD_SEC="${RTT_THRESHOLD_SEC:-0.35}"
ALERT_FOR="${ALERT_FOR:-2m}"

c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
step() { printf '\n%s==>%s %s\n' "$c_green" "$c_reset" "$*"; }

[[ -n "$KUBECTL" ]] || { echo "oc/kubectl required" >&2; exit 1; }

wire_alerts() {
  local host
  host="$("$KUBECTL" get route grafana-network-aiops -n "$GRAFANA_NS" -o jsonpath='{.spec.host}' 2>/dev/null || true)"
  [[ -n "$host" ]] || { warn "Grafana route missing — run install-grafana-network-aiops.sh all"; return 1; }

  if ! "$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-grafana-bridge >/dev/null 2>&1; then
    warn "netobserv-grafana-bridge missing — run wire-openclaw-hooks.sh first"
    return 1
  fi

  step "Provision Grafana alert + bridge webhook contact point"
  python3 - "$host" "$RTT_THRESHOLD_SEC" "$ALERT_FOR" <<'PY'
import base64, json, ssl, sys, urllib.error, urllib.request

host, threshold, alert_for = sys.argv[1], float(sys.argv[2]), sys.argv[3]
auth = base64.b64encode(b"admin:netobserv-demo").decode()
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
headers = {"Content-Type": "application/json", "Authorization": f"Basic {auth}"}


def req(method, path, body=None, ok_codes=(200, 201, 202, 204), allow_fail=False):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(f"https://{host}{path}", data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, context=ctx) as resp:
            raw = resp.read().decode()
            if not raw.strip():
                return resp.status, {}
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as exc:
        err = exc.read().decode()
        if allow_fail:
            return exc.code, err
        if exc.code not in ok_codes:
            sys.stderr.write(f"{method} {path} -> {exc.code}: {err}\n")
            raise
        return exc.code, err


# Folder
folder_uid = "netobserv-aiops"
code, _ = req("GET", f"/api/folders/{folder_uid}", allow_fail=True)
if code == 404:
    _, folder = req("POST", "/api/folders", {"uid": folder_uid, "title": "NetObserv Alerts"})
    folder_uid = folder.get("uid", folder_uid)

# Contact point (receiver name must match notification_settings.receiver)
cp_uid = "openclaw-netobserv"
cp_name = "OpenClaw NetObserv AIOps"
cp_payload = {
    "uid": cp_uid,
    "name": cp_name,
    "type": "webhook",
    "settings": {
        "url": "http://netobserv-grafana-bridge.openclaw.svc.cluster.local:8080/grafana",
        "httpMethod": "POST",
    },
    "disableResolveMessage": True,
}
code, _ = req("PUT", f"/api/v1/provisioning/contact-points/{cp_uid}", cp_payload, allow_fail=True)
if code == 404:
    req("POST", "/api/v1/provisioning/contact-points", cp_payload)

expr = (
    'sum(rate(netobserv_namespace_rtt_seconds_sum{SrcK8S_Namespace="todo-demo"}[2m])) '
    '/ sum(rate(netobserv_namespace_rtt_seconds_count{SrcK8S_Namespace="todo-demo"}[2m]))'
)

rule_uid = "netobserv-todo-rtt-high"
rule_payload = {
    "uid": rule_uid,
    "title": "NetObserv todo-demo avg RTT elevated",
    "ruleGroup": "netobserv-aiops",
    "folderUID": folder_uid,
    "noDataState": "NoData",
    "execErrState": "Error",
    "for": alert_for,
    "condition": "C",
    "annotations": {
        "summary": "Elevated avg flow RTT for todo-demo — possible todo→PostgreSQL path degradation.",
        "description": "NetObserv namespace RTT from federated Prometheus (openclaw-otel).",
    },
    "labels": {"severity": "warning", "team": "netobserv-demo", "alertname": "NetObservTodoRttHigh"},
    "notification_settings": {
        "receiver": cp_name,
        "group_wait": "10s",
        "group_interval": "1m",
        "repeat_interval": "5m",
    },
    "data": [
        {
            "refId": "A",
            "relativeTimeRange": {"from": 600, "to": 0},
            "datasourceUid": "prometheus-openclaw-otel",
            "model": {
                "expr": expr,
                "refId": "A",
                "intervalMs": 1000,
                "maxDataPoints": 43200,
            },
        },
        {
            "refId": "B",
            "relativeTimeRange": {"from": 0, "to": 0},
            "datasourceUid": "__expr__",
            "model": {
                "type": "reduce",
                "expression": "A",
                "reducer": "last",
                "refId": "B",
            },
        },
        {
            "refId": "C",
            "relativeTimeRange": {"from": 0, "to": 0},
            "datasourceUid": "__expr__",
            "model": {
                "type": "threshold",
                "expression": "B",
                "conditions": [
                    {
                        "evaluator": {"params": [threshold], "type": "gt"},
                        "operator": {"type": "and"},
                        "query": {"params": ["C"]},
                        "reducer": {"params": [], "type": "last"},
                        "type": "query",
                    }
                ],
                "refId": "C",
            },
        },
    ],
}

code, _ = req("PUT", f"/api/v1/provisioning/alert-rules/{rule_uid}", rule_payload, allow_fail=True)
if code == 404:
    req("POST", "/api/v1/provisioning/alert-rules", rule_payload)

print(json.dumps({
    "folder": folder_uid,
    "contact_point": cp_uid,
    "rule": rule_uid,
    "threshold_sec": threshold,
    "for": alert_for,
    "expr": expr,
}, indent=2))
PY
  ok "Grafana alert wired (RTT > ${RTT_THRESHOLD_SEC}s for ${ALERT_FOR})"
}

status_alerts() {
  local host
  host="$("$KUBECTL" get route grafana-network-aiops -n "$GRAFANA_NS" -o jsonpath='{.spec.host}' 2>/dev/null || true)"
  [[ -n "$host" ]] || { warn "Grafana route missing"; return 1; }
  python3 - "$host" <<'PY'
import base64, json, ssl, sys, urllib.error, urllib.request
host = sys.argv[1]
auth = base64.b64encode(b"admin:netobserv-demo").decode()
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
headers = {"Authorization": f"Basic {auth}"}

def get(path):
    req = urllib.request.Request(f"https://{host}{path}", headers=headers)
    with urllib.request.urlopen(req, context=ctx) as resp:
        return json.loads(resp.read().decode())

try:
    contact_points = get("/api/v1/provisioning/contact-points")
    cp = next((c for c in contact_points if c.get("uid") == "openclaw-netobserv"), None)
    if not cp:
        print("contact point openclaw-netobserv MISSING")
        sys.exit(1)
    print("contact point OK", cp.get("name"))

    rule = get("/api/v1/provisioning/alert-rules/netobserv-todo-rtt-high")
    print("alert rule OK", rule.get("title"))
except urllib.error.HTTPError as exc:
    print("Grafana provisioning check failed", exc.code, exc.read().decode())
    sys.exit(1)
PY
  ok "Grafana event-AIOps alert resources present"
}

case "$CMD" in
  all|wire) wire_alerts ;;
  status) status_alerts ;;
  *)
    echo "usage: $0 [all|wire|status]" >&2
    exit 1
    ;;
esac

ok "Event-driven path: Grafana alert → bridge → OpenClaw /hooks/agent → Slack"
