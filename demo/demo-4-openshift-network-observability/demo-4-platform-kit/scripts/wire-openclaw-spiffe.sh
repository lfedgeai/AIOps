#!/usr/bin/env bash
# Phase 3d — SPIFFE/SPIRE mTLS for event-driven AIOps (ZTWI).
#
# Upgrades Grafana bridge → OpenClaw hooks path from Bearer token to SPIFFE mTLS:
#   Grafana ──HTTP──▶ netobserv-grafana-bridge ──mTLS──▶ openclaw-hooks-mtls ──Bearer──▶ OpenClaw
#
# Prerequisites:
#   1. Zero Trust Workload Identity Manager installed and Ready (OperatorHub)
#   2. SPIRE operands deployed (SpireServer, SpireAgent, SpiffeCSIDriver — name: cluster)
#   3. Event-AIOps wired: wire-openclaw-event-aiops.sh (hooks token + bridge + Grafana alert)
#
# Usage:
#   SLACK_CHANNEL_ID=C0123456789 ./scripts/wire-openclaw-spiffe.sh all
#   ./scripts/wire-openclaw-spiffe.sh status
#   ./scripts/wire-openclaw-spiffe.sh rollback
#
# Env:
#   SLACK_CHANNEL_ID              required for bridge deploy (same as hooks wiring)
#   SPIFFE_HELPER_IMAGE           default: digest from RH errata (tags like :1.1.0 often missing)
#   CLUSTER_TRUST_DOMAIN          auto-detected from SpireServer when unset
#   OPENCLAW_NS                   default openclaw
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="${1:-all}"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
KUBECTL="$(command -v oc || command -v kubectl)"
SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}"
SPIFFE_HELPER_IMAGE="${SPIFFE_HELPER_IMAGE:-}"
CLUSTER_TRUST_DOMAIN="${CLUSTER_TRUST_DOMAIN:-}"
MANIFESTS="$ROOT/manifests/openclaw-spiffe"
TMPDIR="${TMPDIR:-/tmp}"

# shellcheck source=resolve-slack-channel.sh
source "$ROOT/scripts/resolve-slack-channel.sh"
resolve_slack_channel_id

c_green=$'\033[1;32m'
c_yellow=$'\033[1;33m'
c_red=$'\033[1;31m'
c_reset=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
fail() { printf '%s[fail]%s %s\n' "$c_red" "$c_reset" "$*" >&2; exit 1; }
step() { printf '\n%s==>%s %s\n' "$c_green" "$c_reset" "$*"; }

[[ -n "$KUBECTL" ]] || fail "oc/kubectl required"

detect_spiffe_helper_image() {
  if [[ -n "$SPIFFE_HELPER_IMAGE" ]]; then
    ok "SPIFFE Helper image: ${SPIFFE_HELPER_IMAGE}"
    return 0
  fi
  local arch digest
  arch="$("$KUBECTL" get nodes -o jsonpath='{.items[0].status.nodeInfo.architecture}' 2>/dev/null || echo amd64)"
  case "$arch" in
    amd64|x86_64) digest=c652bad4e445343ec969bb24ab19360230d3b544f753a7408a303d20e35ee627 ;;
    arm64|aarch64) digest=933bd1b34f161f9fb64693e0862ed379682e9e11b73f6cfb1f563242eb80266e ;;
    ppc64le) digest=70608fa6cc9ae890076a9fae2db2257d0fe4730e45de7560a945512f6b218a57 ;;
    s390x) digest=94d1835d1f7b237c80c457974f6331d7c06b709c02cc1c804818aaa569dc8efd ;;
    *) fail "Unsupported node arch for SPIFFE Helper digest: $arch — set SPIFFE_HELPER_IMAGE" ;;
  esac
  SPIFFE_HELPER_IMAGE="registry.redhat.io/zero-trust-workload-identity-manager/spiffe-helper-rhel9@sha256:${digest}"
  ok "SPIFFE Helper image (${arch}): ${SPIFFE_HELPER_IMAGE}"
}

ensure_redhat_pull_secret() {
  step "Ensure registry.redhat.io pull secret in ${OPENCLAW_NS}"
  if ! "$KUBECTL" -n "$OPENCLAW_NS" get secret redhat-io-pull >/dev/null 2>&1; then
    "$KUBECTL" get secret pull-secret -n openshift-config -o jsonpath='{.data.\.dockerconfigjson}' | base64 -d > "$TMPDIR/redhat-io-pull.json"
    "$KUBECTL" -n "$OPENCLAW_NS" create secret generic redhat-io-pull \
      --from-file=.dockerconfigjson="$TMPDIR/redhat-io-pull.json" \
      --type=kubernetes.io/dockerconfigjson \
      --dry-run=client -o yaml | "$KUBECTL" apply -f -
  fi
  for sa in netobserv-grafana-bridge openclaw-hooks-mtls; do
    "$KUBECTL" -n "$OPENCLAW_NS" get sa "$sa" >/dev/null 2>&1 || continue
    "$KUBECTL" -n "$OPENCLAW_NS" secrets link "$sa" redhat-io-pull --for=pull 2>/dev/null || true
  done
  ok "Pull secret linked for SPIFFE Helper"
}

detect_trust_domain() {
  if [[ -n "$CLUSTER_TRUST_DOMAIN" ]]; then
    return 0
  fi
  CLUSTER_TRUST_DOMAIN="$("$KUBECTL" get ZeroTrustWorkloadIdentityManager cluster \
    -o jsonpath='{.spec.trustDomain}' 2>/dev/null || true)"
  if [[ -z "$CLUSTER_TRUST_DOMAIN" ]]; then
    CLUSTER_TRUST_DOMAIN="$("$KUBECTL" get spireserver cluster \
      -o jsonpath='{.spec.trustDomain}' 2>/dev/null || true)"
  fi
  if [[ -z "$CLUSTER_TRUST_DOMAIN" ]]; then
    local ingress
    ingress="$("$KUBECTL" get ingresses.config/cluster -o jsonpath='{.spec.domain}' 2>/dev/null || true)"
    CLUSTER_TRUST_DOMAIN="${ingress:-}"
  fi
  [[ -n "$CLUSTER_TRUST_DOMAIN" ]] || fail "Could not detect trust domain — set CLUSTER_TRUST_DOMAIN"
}

check_ztwi_ready() {
  step "Verify Zero Trust Workload Identity Manager"
  if ! "$KUBECTL" get crd clusterspiffeids.spire.spiffe.io >/dev/null 2>&1; then
    fail "ClusterSPIFFEID CRD missing — install Zero Trust Workload Identity Manager from OperatorHub"
  fi
  local ready
  ready="$("$KUBECTL" get ZeroTrustWorkloadIdentityManager cluster \
    -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo Unknown)"
  if [[ "$ready" != "True" ]]; then
    fail "ZeroTrustWorkloadIdentityManager cluster not Ready (status=$ready). Deploy SpireServer/Agent/SpiffeCSIDriver first — see docs/SPIFFE-WORKLOAD-IDENTITY-GUIDE.md"
  fi
  ok "ZTWI Ready"
  "$KUBECTL" get spireserver/cluster spireagent/cluster spiffecsidriver/cluster 2>/dev/null || \
    warn "One or more SPIRE operand CRs missing — confirm SpireServer, SpireAgent, SpiffeCSIDriver named cluster exist"
}

check_event_aiops_prereq() {
  step "Verify event-AIOps prerequisites"
  "$KUBECTL" -n "$OPENCLAW_NS" get secret openclaw-hooks-token >/dev/null 2>&1 || \
    fail "openclaw-hooks-token missing — run ./scripts/wire-openclaw-event-aiops.sh first"
  "$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-grafana-bridge >/dev/null 2>&1 || \
    fail "netobserv-grafana-bridge missing — run ./scripts/wire-openclaw-hooks.sh first"
  ok "hooks token + bridge present"
}

render_manifests() {
  local out="$TMPDIR/openclaw-spiffe-rendered.yaml"
  require_slack_channel_id || fail "SLACK_CHANNEL_ID required (env or site-secrets.local.yaml)"
  detect_trust_domain
  python3 - "$MANIFESTS" "$out" "$SPIFFE_HELPER_IMAGE" "$CLUSTER_TRUST_DOMAIN" "$SLACK_CHANNEL_ID" <<'PY'
import sys
from pathlib import Path

manifests, out, helper_image, trust_domain, slack = sys.argv[1:6]
chunks = []
for rel in (
    "00-scc-privileged-bindings.yaml",
    "01-serviceaccounts.yaml",
    "02-clusterspiffeid-bridge.yaml",
    "03-clusterspiffeid-hooks-mtls.yaml",
    "04-spiffe-helper-configmaps.yaml",
    "05-openclaw-hooks-mtls-proxy.yaml",
    "06-netobserv-grafana-bridge-spiffe.yaml",
):
    text = (Path(manifests) / rel).read_text()
    text = text.replace("SPIFFE_HELPER_IMAGE_PLACEHOLDER", helper_image)
    text = text.replace("CLUSTER_TRUST_DOMAIN_PLACEHOLDER", trust_domain)
    text = text.replace("SLACK_CHANNEL_ID_PLACEHOLDER", slack)
    chunks.append(text.rstrip() + "\n")
Path(out).write_text("\n---\n".join(chunks))
print(out)
PY
  RENDERED_MANIFEST="$out"
}

deploy_scripts_configmaps() {
  step "Refresh bridge + hooks-mtls script ConfigMaps"
  "$KUBECTL" -n "$OPENCLAW_NS" create configmap netobserv-grafana-bridge-scripts \
    --from-file=netobserv-grafana-bridge.py="$ROOT/openclaw-skills/netobserv-heal/scripts/netobserv-grafana-bridge.py" \
    --dry-run=client -o yaml | "$KUBECTL" apply -f -
  "$KUBECTL" -n "$OPENCLAW_NS" create configmap openclaw-hooks-mtls-scripts \
    --from-file=openclaw-hooks-mtls-proxy.py="$ROOT/openclaw-skills/netobserv-heal/scripts/openclaw-hooks-mtls-proxy.py" \
    --dry-run=client -o yaml | "$KUBECTL" apply -f -
  ok "script ConfigMaps applied"
}

apply_spiffe_manifests() {
  [[ -n "${SLACK_CHANNEL_ID:-}" ]] || fail "SLACK_CHANNEL_ID required for wire/all"
  ensure_redhat_pull_secret
  detect_spiffe_helper_image
  render_manifests
  step "Apply SPIFFE manifests (ClusterSPIFFEID + mTLS proxy + bridge upgrade)"
  "$KUBECTL" apply -f "$RENDERED_MANIFEST"
  "$KUBECTL" -n "$OPENCLAW_NS" rollout status deployment/openclaw-hooks-mtls --timeout=300s || \
    warn "openclaw-hooks-mtls rollout slow — check SPIFFE CSI + ClusterSPIFFEID"
  "$KUBECTL" -n "$OPENCLAW_NS" rollout status deployment/netobserv-grafana-bridge --timeout=300s || \
    warn "netobserv-grafana-bridge rollout slow — oc logs -n openclaw deploy/netobserv-grafana-bridge -c bridge"
  ok "SPIFFE workloads deployed"
}

smoke_test_mtls() {
  step "Smoke-test bridge → openclaw-hooks-mtls → OpenClaw"
  local attempt
  for attempt in 1 2; do
    if run_bridge_smoke; then
      ok "Bridge smoke test accepted (mTLS path)"
      return 0
    fi
    if [[ "$attempt" == "1" ]]; then
      warn "Smoke test failed — restarting hooks-mtls for fresh SPIFFE SVID (attempt $attempt/2)"
      "$KUBECTL" -n "$OPENCLAW_NS" rollout restart deployment/openclaw-hooks-mtls
      "$KUBECTL" -n "$OPENCLAW_NS" rollout status deployment/openclaw-hooks-mtls --timeout=180s || true
      sleep 5
    fi
  done
  warn "Smoke test failed — check oc logs -n $OPENCLAW_NS deploy/openclaw-hooks-mtls -c hooks-mtls"
  return 1
}

run_bridge_smoke() {
  "$KUBECTL" -n "$OPENCLAW_NS" run grafana-bridge-spiffe-smoke --rm -i --restart=Never \
    --image=registry.access.redhat.com/ubi9/ubi-minimal:latest \
    --command -- sh -lc "
command -v curl >/dev/null 2>&1 || microdnf install -y curl >/dev/null 2>&1
code=\$(curl -sS -o /tmp/out -w '%{http_code}' \\
  -X POST 'http://netobserv-grafana-bridge.openclaw.svc.cluster.local:8080/grafana' \\
  -H 'Content-Type: application/json' \\
  -d '{\"status\":\"firing\",\"title\":\"spiffe-smoke\",\"message\":\"SPIFFE mTLS smoke — ignore.\",\"commonLabels\":{\"alertname\":\"SpiffeSmoke\"}}')
echo HTTP=\$code
head -c 400 /tmp/out 2>/dev/null || true
test \"\$code\" = '200' || test \"\$code\" = '202'
"
}

cmd_status() {
  step "SPIFFE / ZTWI status"
  check_ztwi_ready || true
  detect_trust_domain 2>/dev/null || CLUSTER_TRUST_DOMAIN="${CLUSTER_TRUST_DOMAIN:-unknown}"
  echo "Trust domain: $CLUSTER_TRUST_DOMAIN"
  "$KUBECTL" get clusterspiffeid netobserv-grafana-bridge openclaw-hooks-mtls 2>/dev/null || \
    warn "ClusterSPIFFEID resources missing — run wire-openclaw-spiffe.sh all"
  "$KUBECTL" -n "$OPENCLAW_NS" get deploy/openclaw-hooks-mtls deploy/netobserv-grafana-bridge svc/openclaw-hooks-mtls 2>/dev/null || true
  local bridge_mtls
  bridge_mtls="$("$KUBECTL" -n "$OPENCLAW_NS" get deploy/netobserv-grafana-bridge \
    -o jsonpath='{.spec.template.spec.containers[?(@.name=="bridge")].env[?(@.name=="SPIFFE_MTLS")].value}' 2>/dev/null || echo "")"
  if [[ "$bridge_mtls" == "1" ]]; then
    ok "Bridge SPIFFE_MTLS=1 (Bearer token removed from bridge pod)"
  else
    warn "Bridge not upgraded to SPIFFE mTLS yet"
  fi
}

cmd_rollback() {
  step "Rollback to Bearer-token bridge (non-SPIFFE)"
  require_slack_channel_id || fail "SLACK_CHANNEL_ID required for rollback (env or site-secrets.local.yaml)"
  RECYCLE_POD=0 SLACK_CHANNEL_ID="$SLACK_CHANNEL_ID" "$ROOT/scripts/wire-openclaw-hooks.sh"
  "$KUBECTL" -n "$OPENCLAW_NS" delete deploy/openclaw-hooks-mtls svc/openclaw-hooks-mtls \
    configmap/openclaw-hooks-mtls-scripts openclaw-hooks-mtls-spiffe-helper \
    --ignore-not-found
  "$KUBECTL" delete clusterspiffeid/netobserv-grafana-bridge clusterspiffeid/openclaw-hooks-mtls --ignore-not-found
  "$KUBECTL" -n "$OPENCLAW_NS" delete rolebinding/netobserv-grafana-bridge-spiffe-privileged \
    rolebinding/openclaw-hooks-mtls-spiffe-privileged --ignore-not-found
  "$KUBECTL" -n "$OPENCLAW_NS" delete configmap netobserv-grafana-bridge-spiffe-helper --ignore-not-found
  ok "Rolled back to Bearer-token event path"
}

case "$CMD" in
  all|wire)
    check_ztwi_ready
    check_event_aiops_prereq
    deploy_scripts_configmaps
    apply_spiffe_manifests
    smoke_test_mtls
    ok "Phase 3d SPIFFE mTLS wired. Verify: ./scripts/wire-openclaw-spiffe.sh status"
    ok "Docs: docs/SPIFFE-WORKLOAD-IDENTITY-GUIDE.md · SECURITY-PROOF-GUIDE.md Layer 3"
    ;;
  status)
    cmd_status
    ;;
  rollback)
    cmd_rollback
    ;;
  *)
    echo "usage: $0 [all|wire|status|rollback]" >&2
    exit 1
    ;;
esac
