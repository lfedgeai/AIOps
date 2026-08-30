#!/usr/bin/env bash
# Install Red Hat Zero Trust Workload Identity Manager + SPIRE operands.
#
# Usage:
#   ./scripts/install-ztwi-spire.sh status
#   ./scripts/install-ztwi-spire.sh operator     # OperatorHub subscription only
#   ./scripts/install-ztwi-spire.sh operands     # SPIRE CRs (operator must be Ready)
#   ./scripts/install-ztwi-spire.sh repair       # restart agents after server CA rotation / bundle drift
#
# Env:
#   ZTWIM_NS=zero-trust-workload-identity-manager
#   TRUST_DOMAIN=...              default: cluster ingress apps domain
#   CLUSTER_NAME=cluster-name    default: short name derived from ingress domain
#   SPIRE_STORAGE_CLASS=gp3-csi   default: cluster default StorageClass
#   SPIRE_PVC_SIZE=5Gi
#   WAIT_OPERATOR_SEC=900
#   WAIT_OPERANDS_SEC=900
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFESTS="$ROOT/manifests/ztwi"
KUBECTL="$(command -v oc || command -v kubectl)"
CMD="${1:-all}"
ZTWIM_NS="${ZTWIM_NS:-zero-trust-workload-identity-manager}"
WAIT_OPERATOR_SEC="${WAIT_OPERATOR_SEC:-900}"
WAIT_OPERANDS_SEC="${WAIT_OPERANDS_SEC:-900}"
SPIRE_PVC_SIZE="${SPIRE_PVC_SIZE:-5Gi}"

c_green=$'\033[1;32m'
c_yellow=$'\033[1;33m'
c_blue=$'\033[1;34m'
c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
step() { printf '\n%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -n "$KUBECTL" ]] || die "oc/kubectl required"

detect_cluster_values() {
  INGRESS_DOMAIN="$("$KUBECTL" get ingresses.config/cluster -o jsonpath='{.spec.domain}')"
  [[ -n "$INGRESS_DOMAIN" ]] || die "Could not read cluster ingress domain"
  TRUST_DOMAIN="${TRUST_DOMAIN:-$INGRESS_DOMAIN}"
  JWT_ISSUER="${JWT_ISSUER:-https://oidc-discovery.${INGRESS_DOMAIN}}"
  if [[ -z "${CLUSTER_NAME:-}" ]]; then
    CLUSTER_NAME="$(printf '%s' "$INGRESS_DOMAIN" | sed -E 's/^apps\.cluster-([^.]+)\..*/\1/; t; s/^apps\.([^.]+)\..*/\1/; t; s/\..*//')"
    [[ -n "$CLUSTER_NAME" ]] || CLUSTER_NAME="openshift"
  fi
  SPIRE_STORAGE_CLASS="${SPIRE_STORAGE_CLASS:-$("$KUBECTL" get storageclass -o jsonpath='{.items[?(@.metadata.annotations.storageclass\.kubernetes\.io/is-default-class=="true")].metadata.name}' 2>/dev/null)}"
  [[ -n "$SPIRE_STORAGE_CLASS" ]] || SPIRE_STORAGE_CLASS="gp3-csi"
}

cmd_status() {
  detect_cluster_values
  step "Cluster values"
  echo "  trustDomain=$TRUST_DOMAIN"
  echo "  clusterName=$CLUSTER_NAME"
  echo "  jwtIssuer=$JWT_ISSUER"
  echo "  storageClass=$SPIRE_STORAGE_CLASS"
  echo
  step "Operator"
  "$KUBECTL" get ns "$ZTWIM_NS" 2>/dev/null || warn "namespace $ZTWIM_NS missing"
  "$KUBECTL" get subscription,csv -n "$ZTWIM_NS" 2>/dev/null || true
  "$KUBECTL" get deployment -l name=zero-trust-workload-identity-manager -n "$ZTWIM_NS" 2>/dev/null || true
  echo
  step "Operands"
  "$KUBECTL" get ZeroTrustWorkloadIdentityManager,spireserver,spireagent,spiffecsidriver,spireoidcdiscoveryprovider 2>/dev/null || \
    warn "ZTWI / SPIRE CRDs not available yet"
  local ready
  ready="$("$KUBECTL" get ZeroTrustWorkloadIdentityManager cluster \
    -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo Unknown)"
  echo "ZeroTrustWorkloadIdentityManager Ready=$ready"
  "$KUBECTL" get statefulset/spire-server -n "$ZTWIM_NS" 2>/dev/null || true
  "$KUBECTL" get daemonset/spire-agent,spire-spiffe-csi-driver -n "$ZTWIM_NS" 2>/dev/null || true
  "$KUBECTL" get pods -n "$ZTWIM_NS" 2>/dev/null | head -20 || true
}

wait_csv() {
  step "Wait for ZTWI CSV Succeeded (timeout ${WAIT_OPERATOR_SEC}s)"
  local end=$((SECONDS + WAIT_OPERATOR_SEC))
  while (( SECONDS < end )); do
    if "$KUBECTL" get csv -n "$ZTWIM_NS" -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\n"}{end}' 2>/dev/null | grep -q Succeeded; then
      ok "CSV installed"
      "$KUBECTL" get csv -n "$ZTWIM_NS"
      return 0
    fi
    sleep 10
  done
  warn "CSV not Succeeded yet — check: oc get csv,installplan -n $ZTWIM_NS"
  return 1
}

wait_controller() {
  step "Wait for ZTWI controller deployment"
  "$KUBECTL" -n "$ZTWIM_NS" rollout status deployment/zero-trust-workload-identity-manager-controller-manager \
    --timeout="${WAIT_OPERATOR_SEC}s" 2>/dev/null && ok "Controller ready" || \
    warn "Controller rollout slow — oc get pods -n $ZTWIM_NS"
}

cmd_operator() {
  detect_cluster_values
  step "Install ZTWI operator subscription"
  "$KUBECTL" apply -f "$MANIFESTS/01-operator-subscription.yaml"
  wait_csv || true
  wait_controller || true
  ok "Operator install initiated — run './scripts/install-ztwi-spire.sh operands' when CSV is Succeeded"
}

apply_operands() {
  detect_cluster_values
  step "Apply SPIRE operand CRs (trustDomain=$TRUST_DOMAIN)"
  "$KUBECTL" apply -f - <<EOF
apiVersion: operator.openshift.io/v1alpha1
kind: ZeroTrustWorkloadIdentityManager
metadata:
  name: cluster
  labels:
    app.kubernetes.io/name: zero-trust-workload-identity-manager
    app.kubernetes.io/managed-by: zero-trust-workload-identity-manager
spec:
  trustDomain: "${TRUST_DOMAIN}"
  clusterName: "${CLUSTER_NAME}"
  bundleConfigMap: spire-bundle
EOF

  "$KUBECTL" apply -f - <<EOF
apiVersion: operator.openshift.io/v1alpha1
kind: SpireServer
metadata:
  name: cluster
spec:
  logLevel: info
  logFormat: text
  jwtIssuer: "${JWT_ISSUER}"
  caValidity: 24h
  defaultX509Validity: 1h
  defaultJWTValidity: 5m
  jwtKeyType: rsa-2048
  caSubject:
    country: US
    organization: Red Hat
    commonName: SPIRE Server CA
  persistence:
    size: ${SPIRE_PVC_SIZE}
    accessMode: ReadWriteOnce
    storageClass: ${SPIRE_STORAGE_CLASS}
  datastore:
    databaseType: sqlite3
    connectionString: /run/spire/data/datastore.sqlite3
    tlsSecretName: ""
    maxOpenConns: 100
    maxIdleConns: 10
    connMaxLifetime: 0
    disableMigration: "false"
EOF

  step "Wait for SPIRE Server"
  local end=$((SECONDS + WAIT_OPERANDS_SEC))
  while (( SECONDS < end )); do
    if "$KUBECTL" get statefulset/spire-server -n "$ZTWIM_NS" >/dev/null 2>&1; then
      break
    fi
    sleep 5
  done
  "$KUBECTL" -n "$ZTWIM_NS" rollout status statefulset/spire-server --timeout="${WAIT_OPERANDS_SEC}s" || \
    warn "spire-server rollout slow"

  "$KUBECTL" apply -f - <<EOF
apiVersion: operator.openshift.io/v1alpha1
kind: SpireAgent
metadata:
  name: cluster
spec:
  socketPath: /run/spire/agent-sockets
  logLevel: info
  logFormat: text
  nodeAttestor:
    k8sPSATEnabled: "true"
  workloadAttestors:
    k8sEnabled: "true"
    workloadAttestorsVerification:
      type: auto
      hostCertBasePath: /etc/kubernetes
      hostCertFileName: kubelet-ca.crt
    disableContainerSelectors: "false"
    useNewContainerLocator: "true"
EOF

  step "Wait for SPIRE Agent"
  end=$((SECONDS + WAIT_OPERANDS_SEC))
  while (( SECONDS < end )); do
    if "$KUBECTL" get daemonset/spire-agent -n "$ZTWIM_NS" >/dev/null 2>&1; then
      break
    fi
    sleep 5
  done
  "$KUBECTL" -n "$ZTWIM_NS" rollout status daemonset/spire-agent --timeout="${WAIT_OPERANDS_SEC}s" || \
    warn "spire-agent rollout slow"

  "$KUBECTL" apply -f - <<EOF
apiVersion: operator.openshift.io/v1alpha1
kind: SpiffeCSIDriver
metadata:
  name: cluster
spec:
  agentSocketPath: /run/spire/agent-sockets
  pluginName: csi.spiffe.io
EOF

  step "Wait for SPIFFE CSI driver"
  end=$((SECONDS + WAIT_OPERANDS_SEC))
  while (( SECONDS < end )); do
    if "$KUBECTL" get daemonset/spire-spiffe-csi-driver -n "$ZTWIM_NS" >/dev/null 2>&1; then
      break
    fi
    sleep 5
  done
  "$KUBECTL" -n "$ZTWIM_NS" rollout status daemonset/spire-spiffe-csi-driver --timeout="${WAIT_OPERANDS_SEC}s" || \
    warn "spire-spiffe-csi-driver rollout slow"

  "$KUBECTL" apply -f - <<EOF
apiVersion: operator.openshift.io/v1alpha1
kind: SpireOIDCDiscoveryProvider
metadata:
  name: cluster
spec:
  logLevel: info
  logFormat: text
  csiDriverName: csi.spiffe.io
  jwtIssuer: "${JWT_ISSUER}"
  replicaCount: 1
  managedRoute: "true"
EOF

  step "Wait for ZeroTrustWorkloadIdentityManager Ready"
  end=$((SECONDS + WAIT_OPERANDS_SEC))
  while (( SECONDS < end )); do
    local ready
    ready="$("$KUBECTL" get ZeroTrustWorkloadIdentityManager cluster \
      -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "")"
    if [[ "$ready" == "True" ]]; then
      ok "ZeroTrustWorkloadIdentityManager Ready"
      return 0
    fi
    sleep 10
  done
  warn "ZeroTrustWorkloadIdentityManager not Ready yet — oc get ZeroTrustWorkloadIdentityManager cluster -o yaml"
  return 1
}

cmd_repair() {
  step "Restart SPIRE agents + CSI + OIDC (common fix after spire-server CA rotation)"
  if ! "$KUBECTL" get daemonset/spire-agent -n "$ZTWIM_NS" >/dev/null 2>&1; then
    die "spire-agent missing — run ./scripts/install-ztwi-spire.sh operands first"
  fi
  "$KUBECTL" -n "$ZTWIM_NS" rollout restart daemonset/spire-agent
  "$KUBECTL" -n "$ZTWIM_NS" rollout restart daemonset/spire-spiffe-csi-driver 2>/dev/null || true
  "$KUBECTL" -n "$ZTWIM_NS" rollout restart deployment/spire-spiffe-oidc-discovery-provider 2>/dev/null || true
  "$KUBECTL" -n "$ZTWIM_NS" rollout status daemonset/spire-agent --timeout="${WAIT_OPERANDS_SEC}s"
  "$KUBECTL" -n "$ZTWIM_NS" rollout status daemonset/spire-spiffe-csi-driver --timeout="${WAIT_OPERANDS_SEC}s" 2>/dev/null || true
  end=$((SECONDS + 120))
  while (( SECONDS < end )); do
    local ready
    ready="$("$KUBECTL" get ZeroTrustWorkloadIdentityManager cluster \
      -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "")"
    if [[ "$ready" == "True" ]]; then
      ok "ZeroTrustWorkloadIdentityManager Ready"
      break
    fi
    sleep 5
  done
  if "$KUBECTL" get deploy/openclaw-hooks-mtls -n openclaw >/dev/null 2>&1; then
    step "Restart OpenClaw SPIFFE workloads"
    "$KUBECTL" -n openclaw rollout restart deploy/openclaw-hooks-mtls deploy/netobserv-grafana-bridge
    "$KUBECTL" -n openclaw rollout status deploy/openclaw-hooks-mtls --timeout=180s
    "$KUBECTL" -n openclaw rollout status deploy/netobserv-grafana-bridge --timeout=180s
  fi
  ok "SPIRE repair complete — run: ./scripts/netobserv-e2e-openclaw-test.sh spiffe-check"
}

cmd_operands() {
  if ! "$KUBECTL" get crd zerotrustworkloadidentitymanagers.operator.openshift.io >/dev/null 2>&1; then
    die "ZTWI CRD missing — run ./scripts/install-ztwi-spire.sh operator first"
  fi
  apply_operands
  ok "SPIRE operands applied"
  ok "Next: ./scripts/wire-openclaw-spiffe.sh all  # requires SLACK_CHANNEL_ID or site-secrets"
}

case "$CMD" in
  status)
    cmd_status
    ;;
  operator)
    cmd_operator
    ;;
  operands)
    cmd_operands
    ;;
  repair)
    cmd_repair
    ;;
  all)
    cmd_operator
    if "$KUBECTL" get crd zerotrustworkloadidentitymanagers.operator.openshift.io >/dev/null 2>&1; then
      apply_operands || true
    else
      step "Wait for ZTWI CRDs"
      end=$((SECONDS + 120))
      while (( SECONDS < end )); do
        if "$KUBECTL" get crd zerotrustworkloadidentitymanagers.operator.openshift.io >/dev/null 2>&1; then
          break
        fi
        sleep 5
      done
      apply_operands || true
    fi
    cmd_status
    ;;
  *)
    echo "usage: $0 [status|operator|operands|repair|all]" >&2
    exit 1
    ;;
esac
