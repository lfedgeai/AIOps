#!/usr/bin/env bash
#
# netobserv-fault-injection.sh
# ---------------------------------------------------------------------------
# Induces a controllable JDBC connection-pool exhaustion on the todo+Postgres
# demo, so Network Observability (and the AI analysis step) has a real,
# repeatable incident to observe.
#
# Mechanism:
#   * Insert Toxiproxy between todo and PostgreSQL (todo -> toxiproxy -> postgresql)
#   * Repoint todo's datasource at toxiproxy and shorten the acquisition timeout
#   * Run a high-concurrency load ramp against the app
#   * Dial in DB latency on demand -> the 16-connection pool saturates,
#     requests queue past the acquisition timeout -> HTTP 500s + Agroal timeouts
#
# Subcommands:
#   inject    Set up Toxiproxy + repoint app + start load (HEALTHY baseline)
#   slow [ms] Add DB latency (default 800ms) -> triggers the exhaustion
#   heal      Remove the latency (recover without tearing down)
#   status    Show proxy toxics, datasource config, pod health
#   restore   Full rollback: remove Toxiproxy/load, restore original config
# ---------------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

APP_NS="${APP_NS:-todo-demo}"
CLIENT_NS="${CLIENT_NS:-todo-client}"
REGISTRY="${REGISTRY:-registry.access.redhat.com}"
CLIENT_IMAGE="${CLIENT_IMAGE:-${REGISTRY}/ubi9/ubi:latest}"
TOXIPROXY_IMAGE="${TOXIPROXY_IMAGE:-ghcr.io/shopify/toxiproxy:2.9.0}"  # mirror if disconnected
CONCURRENCY="${CONCURRENCY:-80}"
LATENCY_MS="${LATENCY_MS:-800}"
ACQUISITION_TIMEOUT="${ACQUISITION_TIMEOUT:-2}"   # seconds Agroal waits for a connection
TARGET_PATH="${TARGET_PATH:-/api}"                # todo endpoint that hits the DB

c_reset=$'\033[0m'; c_blue=$'\033[1;34m'; c_green=$'\033[1;32m'
c_yellow=$'\033[1;33m'; c_red=$'\033[1;31m'
step() { printf '%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
die()  { printf '%s[fail]%s %s\n' "$c_red" "$c_reset" "$*" >&2; exit 1; }

command -v oc >/dev/null 2>&1 || die "'oc' not found in PATH."
oc whoami >/dev/null 2>&1 || die "Not logged in. Run 'oc login ...' first."

# --- helpers ---------------------------------------------------------------
apply_todo_config() {  # apply_todo_config <db-host> [extra-props]
  local dbhost="$1" extra="${2:-}"
  oc apply -n "$APP_NS" -f - >/dev/null <<EOF
kind: ConfigMap
apiVersion: v1
metadata:
  name: todo-config
data:
  application.properties: |-
    quarkus.datasource.db-kind=postgresql
    quarkus.datasource.username=todo
    quarkus.datasource.password=todo
    quarkus.datasource.jdbc.url=jdbc:postgresql://${dbhost}:5432/todo
    quarkus.datasource.jdbc.max-size=16
    quarkus.hibernate-orm.database.generation=drop-and-create
${extra}
EOF
}

apply_db_policy() {  # apply_db_policy <todo-only|todo-and-toxiproxy>
  local mode="$1" from
  if [[ "$mode" == "todo-and-toxiproxy" ]]; then
    from='        - podSelector:
            matchLabels:
              app: todo
        - podSelector:
            matchLabels:
              app: toxiproxy'
  else
    from='        - podSelector:
            matchLabels:
              app: todo'
  fi
  oc apply -n "$APP_NS" -f - >/dev/null <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-db-from-todo-only
  namespace: ${APP_NS}
spec:
  podSelector:
    matchLabels:
      app: postgresql
  policyTypes: ["Ingress"]
  ingress:
    - from:
${from}
      ports:
        - protocol: TCP
          port: 5432
EOF
}

toxi_api() {  # run a curl against the toxiproxy API from the load pod
  oc exec -n "$CLIENT_NS" deploy/loadgen-heavy -- \
    curl -s -m 5 "$@" 2>/dev/null || return 1
}

# --- subcommands -----------------------------------------------------------
cmd_inject() {
  step "Deploying Toxiproxy (todo -> toxiproxy -> postgresql) in $APP_NS"
  oc apply -n "$APP_NS" -f - >/dev/null <<EOF
kind: ConfigMap
apiVersion: v1
metadata:
  name: toxiproxy-config
data:
  toxiproxy.json: |-
    [
      { "name": "postgres", "listen": "0.0.0.0:5432", "upstream": "postgresql:5432", "enabled": true }
    ]
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: toxiproxy
  namespace: ${APP_NS}
  labels:
    app: toxiproxy
spec:
  replicas: 1
  selector:
    matchLabels:
      app: toxiproxy
  template:
    metadata:
      labels:
        app: toxiproxy
    spec:
      securityContext:
        runAsNonRoot: true
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: toxiproxy
          image: ${TOXIPROXY_IMAGE}
          args: ["-host=0.0.0.0", "-config=/config/toxiproxy.json"]
          ports:
            - { containerPort: 5432 }
            - { containerPort: 8474 }
          volumeMounts:
            - name: config
              mountPath: /config
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
      volumes:
        - name: config
          configMap:
            name: toxiproxy-config
---
apiVersion: v1
kind: Service
metadata:
  name: toxiproxy
  namespace: ${APP_NS}
  labels:
    app: toxiproxy
spec:
  selector:
    app: toxiproxy
  ports:
    - { name: pg,  port: 5432, targetPort: 5432 }
    - { name: api, port: 8474, targetPort: 8474 }
EOF
  oc rollout status deployment/toxiproxy -n "$APP_NS" --timeout=120s || warn "toxiproxy not ready."

  step "Allowing toxiproxy -> postgresql in the DB microsegmentation policy"
  apply_db_policy todo-and-toxiproxy
  ok "DB policy updated (todo + toxiproxy permitted)."

  step "Repointing todo at Toxiproxy and setting acquisition-timeout=${ACQUISITION_TIMEOUT}s"
  apply_todo_config toxiproxy "    quarkus.datasource.jdbc.acquisition-timeout=${ACQUISITION_TIMEOUT}"
  oc rollout restart deployment/todo -n "$APP_NS" >/dev/null
  oc rollout status deployment/todo -n "$APP_NS" --timeout=180s || warn "todo not ready yet."

  step "Starting high-concurrency load (${CONCURRENCY} workers -> todo${TARGET_PATH})"
  oc apply -n "$CLIENT_NS" -f - >/dev/null <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: loadgen-heavy
  namespace: ${CLIENT_NS}
  labels:
    app: loadgen-heavy
spec:
  replicas: 1
  selector:
    matchLabels:
      app: loadgen-heavy
  template:
    metadata:
      labels:
        app: loadgen-heavy
    spec:
      securityContext:
        runAsNonRoot: true
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: loadgen
          image: ${CLIENT_IMAGE}
          env:
            - { name: CONCURRENCY, value: "${CONCURRENCY}" }
            - { name: TARGET,      value: "http://todo.${APP_NS}:8080${TARGET_PATH}" }
          command: ["/bin/bash", "-c"]
          args:
            - |
              echo "heavy loadgen: \$CONCURRENCY workers -> \$TARGET"
              run_worker() {
                while true; do
                  code=\$(curl -s -o /dev/null -w "%{http_code}" -m 6 "\$TARGET")
                  if [ "\$code" = "200" ]; then echo "200"; else echo "ERR \$code"; fi
                done
              }
              for i in \$(seq 1 "\$CONCURRENCY"); do run_worker & done
              wait
          resources:
            requests: { cpu: 50m, memory: 64Mi }
            limits:   { cpu: 500m, memory: 256Mi }
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
EOF
  oc rollout status deployment/loadgen-heavy -n "$CLIENT_NS" --timeout=120s || warn "loadgen-heavy not ready."

  cat <<MSG

${c_green}Baseline is live and HEALTHY${c_reset} (Toxiproxy passthrough, no latency).
  Induce the incident:   ./$(basename "$0") slow ${LATENCY_MS}
  Watch the app:         oc logs -n ${CLIENT_NS} deploy/loadgen-heavy -f
                         (200s now; ERR 500 / ERR 000 once you inject latency)
MSG
}

cmd_slow() {
  local ms="${1:-$LATENCY_MS}"
  oc get deployment/loadgen-heavy -n "$CLIENT_NS" >/dev/null 2>&1 || die "Run 'inject' first."
  step "Injecting ${ms}ms of downstream DB latency via Toxiproxy"
  # delete-then-create for idempotency
  toxi_api -X DELETE "http://toxiproxy.${APP_NS}:8474/proxies/postgres/toxics/lat_down" >/dev/null 2>&1 || true
  toxi_api -X POST "http://toxiproxy.${APP_NS}:8474/proxies/postgres/toxics" \
    -H 'Content-Type: application/json' \
    -d "{\"name\":\"lat_down\",\"type\":\"latency\",\"stream\":\"downstream\",\"attributes\":{\"latency\":${ms},\"jitter\":100}}" >/dev/null \
    || die "Failed to add latency toxic (is toxiproxy ready?)."
  ok "DB latency injected. The pool will now saturate under load — 500s/timeouts incoming."
  echo "   Capture it:  ./netobserv-capture-and-bundle.sh"
  echo "   Recover:     ./$(basename "$0") heal"
}

cmd_heal() {
  step "Removing DB latency toxic"
  toxi_api -X DELETE "http://toxiproxy.${APP_NS}:8474/proxies/postgres/toxics/lat_down" >/dev/null 2>&1 || true
  ok "Latency removed — the app should recover within seconds."
}

cmd_status() {
  step "Toxiproxy toxics"
  toxi_api "http://toxiproxy.${APP_NS}:8474/proxies/postgres" 2>/dev/null || echo "  (toxiproxy/load not reachable — inject first)"
  echo
  step "todo datasource config"
  oc get cm todo-config -n "$APP_NS" -o jsonpath='{.data.application\.properties}' 2>/dev/null | grep -E 'jdbc.url|max-size|acquisition' || true
  echo
  step "Pod health"
  oc get pods -n "$APP_NS" -l 'app in (todo,postgresql,toxiproxy)' 2>/dev/null || true
  oc get pods -n "$CLIENT_NS" -l app=loadgen-heavy 2>/dev/null || true
}

cmd_restore() {
  step "Removing load generator and Toxiproxy"
  oc delete deployment/loadgen-heavy -n "$CLIENT_NS" --ignore-not-found >/dev/null
  oc delete deployment/toxiproxy service/toxiproxy configmap/toxiproxy-config -n "$APP_NS" --ignore-not-found >/dev/null

  step "Restoring original datasource config and DB policy"
  apply_todo_config postgresql
  apply_db_policy todo-only
  oc rollout restart deployment/todo -n "$APP_NS" >/dev/null
  oc rollout status deployment/todo -n "$APP_NS" --timeout=180s || warn "todo not ready yet."
  ok "Restored: todo -> postgresql direct, DB locked to the app tier, load stopped."
}

case "${1:-}" in
  inject)  cmd_inject ;;
  slow)    shift; cmd_slow "${1:-}" ;;
  heal)    cmd_heal ;;
  status)  cmd_status ;;
  restore) cmd_restore ;;
  *) echo "Usage: $(basename "$0") {inject|slow [ms]|heal|status|restore}"; exit 1 ;;
esac
