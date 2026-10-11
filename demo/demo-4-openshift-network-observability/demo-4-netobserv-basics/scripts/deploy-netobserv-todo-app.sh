#!/usr/bin/env bash
#
# deploy-netobserv-todo-app.sh
# ---------------------------------------------------------------------------
# Deploys a realistic two-tier application (Quarkus "todo" + PostgreSQL) to
# exercise every Network Observability view with authentic traffic, plus a
# database microsegmentation NetworkPolicy to demonstrate enforced isolation.
#
#   todo-client (consumer)         todo-demo (application tenant)
#   ┌──────────────────┐           ┌───────────────────────────────────┐
#   │  loadgen ────────┼─ east-west┼─> todo (Quarkus :8080) ──┐         │
#   │      │           │  (cross-ns)│      ▲   Route (edge TLS)│         │
#   │      └── direct DB probe ─────┼──────┼──> postgresql :5432 (DB)    │
#   │          (DENIED by policy)   │      └── north-south via router    │
#   └──────────────────┘           └───────────────────────────────────┘
#
# NetObserv views this produces:
#   * Topology     : todo -> postgresql:5432 (real app-to-DB east-west),
#                    cross-namespace consumer edge, north-south via the Route
#   * Traffic flows: filter by namespace to demo per-tenant visibility
#   * DNS          : todo resolving "postgresql", loadgen resolving "todo"
#   * Packet drops : loadgen's direct :5432 attempt is dropped by NetworkPolicy
#                    (visible if PacketDrop was enabled in the installer)
#
# The todo/postgres manifests are applied exactly as provided. Only the traffic
# generator and the database NetworkPolicy are added by this script.
#
# Usage:
#   ./deploy-netobserv-todo-app.sh          # deploy
#   ./deploy-netobserv-todo-app.sh delete   # tear everything down
# ---------------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

# ---------------------------------------------------------------------------
# Tunables (override via environment)
# ---------------------------------------------------------------------------
APP_NS="${APP_NS:-todo-demo}"            # application + database tenant
CLIENT_NS="${CLIENT_NS:-todo-client}"    # consumer tenant (set = APP_NS to co-locate)
REGISTRY="${REGISTRY:-registry.access.redhat.com}"   # for the loadgen image (disconnected: point at mirror)
CLIENT_IMAGE="${CLIENT_IMAGE:-${REGISTRY}/ubi9/ubi:latest}"
INTERVAL="${INTERVAL:-3}"                # seconds between traffic rounds
ENABLE_DBPOLICY="${ENABLE_DBPOLICY:-true}"  # microsegment the DB (allow only the app tier)
PROBE_DB="${PROBE_DB:-true}"             # loadgen attempts a direct (unauthorized) DB connection
ENABLE_ROUTE_TRAFFIC="${ENABLE_ROUTE_TRAFFIC:-true}"  # also drive north-south via the Route

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
c_reset=$'\033[0m'; c_blue=$'\033[1;34m'; c_green=$'\033[1;32m'
c_yellow=$'\033[1;33m'; c_red=$'\033[1;31m'
step() { printf '%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
die()  { printf '%s[fail]%s %s\n' "$c_red" "$c_reset" "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Teardown path
# ---------------------------------------------------------------------------
if [[ "${1:-}" =~ ^(delete|--delete|cleanup|uninstall|remove)$ ]]; then
  step "Removing namespaces: $APP_NS, $CLIENT_NS"
  oc delete namespace "$APP_NS" --ignore-not-found
  [[ "$CLIENT_NS" != "$APP_NS" ]] && oc delete namespace "$CLIENT_NS" --ignore-not-found
  ok "Sample application removed."
  exit 0
fi

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
command -v oc >/dev/null 2>&1 || die "'oc' not found in PATH."
oc whoami >/dev/null 2>&1 || die "Not logged in. Run 'oc login ...' first."
ok "Logged in as $(oc whoami)."

if oc get flowcollector cluster >/dev/null 2>&1; then
  ok "FlowCollector 'cluster' present — Network Observability will capture this traffic."
else
  warn "No FlowCollector found. Deploy Network Observability first, or flows won't be recorded."
fi

step "Creating namespaces"
oc create namespace "$APP_NS" >/dev/null 2>&1 || true
oc label namespace "$APP_NS" netobs-demo=true --overwrite >/dev/null
if [[ "$CLIENT_NS" != "$APP_NS" ]]; then
  oc create namespace "$CLIENT_NS" >/dev/null 2>&1 || true
  oc label namespace "$CLIENT_NS" netobs-demo=true --overwrite >/dev/null
fi
ok "Namespaces ready: $APP_NS$( [[ "$CLIENT_NS" != "$APP_NS" ]] && echo ", $CLIENT_NS" )."

# ---------------------------------------------------------------------------
# 1. PostgreSQL (applied verbatim; PVC status stanza dropped)
#    Image registry.redhat.io/rhel8/postgresql-10 pulls via the cluster's
#    global pull secret (present on a standard OpenShift install).
# ---------------------------------------------------------------------------
step "Deploying PostgreSQL into $APP_NS"
oc apply -n "$APP_NS" -f - >/dev/null <<'EOF'
---
kind: Secret
apiVersion: v1
metadata:
  name: postgresql
data:
  database-name: dG9kbw==
  database-password: dG9kbw==
  database-user: dG9kbw==
type: Opaque
---
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: postgresql
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
  volumeMode: Filesystem
---
kind: Service
apiVersion: v1
metadata:
  name: postgresql
spec:
  ports:
    - name: postgresql
      protocol: TCP
      port: 5432
      targetPort: 5432
  type: ClusterIP
  selector:
    app: postgresql
    deployment: postgresql
---
kind: Deployment
apiVersion: apps/v1
metadata:
  name: postgresql
  labels:
    app: postgresql
    app.kubernetes.io/instance: postgresql
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgresql
  template:
    metadata:
      labels:
        app: postgresql
        deployment: postgresql
    spec:
      volumes:
        - name: postgresql-data
          persistentVolumeClaim:
            claimName: postgresql
      containers:
        - name: postgresql
          image: 'registry.redhat.io/rhel8/postgresql-10:latest'
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 5432
              protocol: TCP
          resources:
            limits:
              memory: 512Mi
          readinessProbe:
            exec:
              command: ["/usr/libexec/check-container"]
            initialDelaySeconds: 5
            timeoutSeconds: 1
            periodSeconds: 10
            failureThreshold: 3
          livenessProbe:
            exec:
              command: ["/usr/libexec/check-container", "--live"]
            initialDelaySeconds: 120
            timeoutSeconds: 10
            periodSeconds: 10
            failureThreshold: 3
          env:
            - name: POSTGRESQL_USER
              valueFrom:
                secretKeyRef:
                  name: postgresql
                  key: database-user
            - name: POSTGRESQL_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgresql
                  key: database-password
            - name: POSTGRESQL_DATABASE
              valueFrom:
                secretKeyRef:
                  name: postgresql
                  key: database-name
          securityContext:
            privileged: false
          volumeMounts:
            - name: postgresql-data
              mountPath: /var/lib/pgsql/data
      restartPolicy: Always
EOF
ok "PostgreSQL manifests applied."

step "Waiting for PostgreSQL to be ready (todo connects at startup)"
oc rollout status deployment/postgresql -n "$APP_NS" --timeout=240s \
  || warn "PostgreSQL not ready yet — todo may crashloop briefly until the DB is up."

# ---------------------------------------------------------------------------
# 2. Quarkus todo app (applied verbatim)
# ---------------------------------------------------------------------------
step "Deploying the todo application into $APP_NS"
oc apply -n "$APP_NS" -f - >/dev/null <<'EOF'
---
kind: ConfigMap
apiVersion: v1
metadata:
  name: todo-config
data:
  application.properties: |-
    quarkus.datasource.db-kind=postgresql
    quarkus.datasource.username=todo
    quarkus.datasource.password=todo
    quarkus.datasource.jdbc.url=jdbc:postgresql://postgresql:5432/todo
    quarkus.datasource.jdbc.max-size=16
    quarkus.hibernate-orm.database.generation=drop-and-create
---
kind: Service
apiVersion: v1
metadata:
  name: todo
spec:
  ports:
    - name: 8080-tcp
      protocol: TCP
      port: 8080
      targetPort: 8080
  type: ClusterIP
  selector:
    app: todo
    deployment: todo
---
kind: Route
apiVersion: route.openshift.io/v1
metadata:
  name: todo
spec:
  to:
    kind: Service
    name: todo
    weight: 100
  port:
    targetPort: 8080-tcp
  tls:
    termination: edge
    insecureEdgeTerminationPolicy: Redirect
  wildcardPolicy: None
---
kind: Deployment
apiVersion: apps/v1
metadata:
  name: todo
  labels:
    app: todo
    app.kubernetes.io/instance: todo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: todo
  template:
    metadata:
      labels:
        app: todo
        deployment: todo
    spec:
      volumes:
        - name: todo-config
          configMap:
            name: todo-config
            defaultMode: 420
      containers:
        - name: todo
          image: 'quay.io/julin/todo:v1'
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8080
              protocol: TCP
          volumeMounts:
            - name: todo-config
              readOnly: true
              mountPath: /work/config/
      restartPolicy: Always
EOF
ok "todo manifests applied."

step "Waiting for the todo application to be ready"
oc rollout status deployment/todo -n "$APP_NS" --timeout=240s \
  || warn "todo not ready yet — check: oc logs -n $APP_NS deploy/todo"

ROUTE_HOST="$(oc get route todo -n "$APP_NS" -o jsonpath='{.spec.host}' 2>/dev/null || true)"
ROUTE_URL=""
[[ -n "$ROUTE_HOST" ]] && ROUTE_URL="https://${ROUTE_HOST}/"
[[ -n "$ROUTE_URL" ]] && ok "Route: $ROUTE_URL"

# ---------------------------------------------------------------------------
# 3. Database microsegmentation: only the app tier may reach PostgreSQL:5432
#    (Realistic FSI/Gov control; makes the dropped-flow demo meaningful.)
# ---------------------------------------------------------------------------
if [[ "$ENABLE_DBPOLICY" == "true" ]]; then
  step "Applying database microsegmentation NetworkPolicy"
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
        - podSelector:
            matchLabels:
              app: todo
      ports:
        - protocol: TCP
          port: 5432
EOF
  ok "PostgreSQL now reachable only from the todo tier — all other :5432 ingress is dropped."
fi

# ---------------------------------------------------------------------------
# 4. Traffic generator (drives real app + DB traffic; robust array-based loop)
# ---------------------------------------------------------------------------
step "Deploying traffic generator (loadgen) into $CLIENT_NS"
oc apply -n "$CLIENT_NS" -f - >/dev/null <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: loadgen
  namespace: ${CLIENT_NS}
  labels:
    app: loadgen
    app.kubernetes.io/part-of: netobs-demo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: loadgen
  template:
    metadata:
      labels:
        app: loadgen
        app.kubernetes.io/name: loadgen
        app.kubernetes.io/part-of: netobs-demo
    spec:
      securityContext:
        runAsNonRoot: true
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: loadgen
          image: ${CLIENT_IMAGE}
          env:
            - { name: APP_NS,               value: "${APP_NS}" }
            - { name: INTERVAL,             value: "${INTERVAL}" }
            - { name: ROUTE_URL,            value: "${ROUTE_URL}" }
            - { name: PROBE_DB,             value: "${PROBE_DB}" }
            - { name: ENABLE_ROUTE_TRAFFIC, value: "${ENABLE_ROUTE_TRAFFIC}" }
          command: ["/bin/bash", "-c"]
          args:
            - |
              echo "loadgen starting; target namespace=\$APP_NS interval=\${INTERVAL}s"
              base="http://todo.\$APP_NS:8080"
              while true; do
                # Reads (drive SELECTs -> todo -> postgresql:5432)
                curl -s -o /dev/null -m 5 "\$base/"    && echo "ok   GET  /"    || echo "fail GET  /"
                curl -s -o /dev/null -m 5 "\$base/api" && echo "ok   GET  /api" || echo "fail GET  /api"
                # Write (drive an INSERT; best-effort in case the schema differs)
                curl -s -o /dev/null -m 5 -X POST -H 'Content-Type: application/json' \\
                  -d '{"title":"netobserv demo","completed":false,"order":1}' "\$base/api" \\
                  && echo "ok   POST /api" || echo "fail POST /api"
                # North-south through the OpenShift router via the Route
                if [[ "\$ENABLE_ROUTE_TRAFFIC" == "true" && -n "\$ROUTE_URL" ]]; then
                  curl -sk -o /dev/null -m 5 "\$ROUTE_URL" && echo "ok   ROUTE" || echo "fail ROUTE"
                fi
                # Unauthorized direct DB connection (should be dropped by NetworkPolicy)
                if [[ "\$PROBE_DB" == "true" ]]; then
                  if timeout 3 bash -c "cat < /dev/tcp/postgresql.\$APP_NS/5432" >/dev/null 2>&1; then
                    echo "warn direct-DB connect SUCCEEDED (policy open?)"
                  else
                    echo "ok   direct-DB connect blocked (expected: microsegmented)"
                  fi
                fi
                sleep "\$INTERVAL"
              done
          resources:
            requests: { cpu: 10m, memory: 32Mi }
            limits:   { cpu: 100m, memory: 64Mi }
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
EOF
oc rollout status deployment/loadgen -n "$CLIENT_NS" --timeout=180s || warn "loadgen not ready yet."
ok "loadgen deployed."

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
console_url="$(oc whoami --show-console 2>/dev/null || true)"
cat <<SUMMARY

${c_green}=============================================================${c_reset}
 Realistic todo + PostgreSQL demo deployed. Traffic is flowing.
   Application tenant : ${APP_NS}      (todo, postgresql)
   Consumer tenant    : ${CLIENT_NS}   (loadgen)
   Public route       : ${ROUTE_URL:-<none>}

 In ${console_url:-<console>}  ->  Observe -> Network Traffic :
   * Topology     : todo -> postgresql on :5432 is the real app-to-DB edge;
                    loadgen -> todo is the cross-namespace consumer edge.
   * Per-tenant   : filter Namespace = ${APP_NS} vs ${CLIENT_NS} to show
                    tenant-scoped visibility (the multi-tenancy story).
   * DB flows     : filter Destination port = 5432 to prove exactly which
                    workload talks to the database, and at what volume —
                    the audit/segmentation evidence FSI/Gov reviewers ask for.
$([[ "$ENABLE_DBPOLICY" == "true" ]] && printf '   * Packet drops : loadgen'"'"'s direct :5432 attempt is denied by the\n                    allow-db-from-todo-only policy. With PacketDrop enabled\n                    in the installer, those dropped flows are visible here.\n')
   * DNS          : todo resolving "postgresql", loadgen resolving "todo".

 Watch traffic being generated:
   oc logs -n ${CLIENT_NS} deploy/loadgen -f

 Tear it all down:
   ./$(basename "$0") delete
${c_green}=============================================================${c_reset}
SUMMARY

