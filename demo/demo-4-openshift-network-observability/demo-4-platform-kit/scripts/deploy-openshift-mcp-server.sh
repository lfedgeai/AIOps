#!/usr/bin/env bash
# Deploy Red Hat openshift-mcp-server (kubernetes-mcp-server chart) for OpenClaw.
# Read-only netobserv toolset (core omitted for small-context workshops); ClusterIP only (no public Route).
#
# Chart install ≠ OpenClaw wiring. Leave mcp.servers.openshift-mcp.enabled=false unless
# ENABLE_OPENSHIFT_MCP=1 and the LLM has sufficient context (128k+ recommended).
# (seeder default). Opt in with ENABLE_OPENSHIFT_MCP=1 only on a larger context model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
RELEASE="${OPENSHIFT_MCP_RELEASE:-openshift-mcp}"
CHART="${OPENSHIFT_MCP_CHART:-oci://ghcr.io/containers/charts/kubernetes-mcp-server}"
CHART_VERSION="${OPENSHIFT_MCP_CHART_VERSION:-}"  # empty = latest
VALUES="${VALUES:-$ROOT/openclaw-skills/manifests/openshift-mcp-server-values.yaml}"

c_blue=$'\033[1;34m'; c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_reset=$'\033[0m'
step() { printf '%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }

command -v helm >/dev/null || { echo "helm required" >&2; exit 1; }
KUBECTL="$(command -v oc || command -v kubectl)"
[[ -n "$KUBECTL" ]] || { echo "oc/kubectl required" >&2; exit 1; }
[[ -f "$VALUES" ]] || { echo "missing values: $VALUES" >&2; exit 1; }

"$KUBECTL" get ns "$OPENCLAW_NS" >/dev/null

# NetObserv plugin soft-check
if ! "$KUBECTL" -n netobserv get svc netobserv-plugin >/dev/null 2>&1; then
  warn "netobserv/netobserv-plugin Service not found — netobserv toolset may fail until NetObserv is installed"
fi

step "Helm upgrade $RELEASE from $CHART (ns=$OPENCLAW_NS)"
ARGS=(upgrade -i "$RELEASE" "$CHART" -n "$OPENCLAW_NS" -f "$VALUES" --wait --timeout 5m)
if [[ -n "$CHART_VERSION" ]]; then
  ARGS+=(--version "$CHART_VERSION")
fi
helm "${ARGS[@]}"

step "Wait for rollout"
"$KUBECTL" -n "$OPENCLAW_NS" rollout status "deploy/$RELEASE" --timeout=180s \
  || "$KUBECTL" -n "$OPENCLAW_NS" rollout status deploy -l "app.kubernetes.io/instance=$RELEASE" --timeout=180s \
  || true

SVC="$("$KUBECTL" -n "$OPENCLAW_NS" get svc -l "app.kubernetes.io/instance=$RELEASE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
if [[ -z "$SVC" ]]; then
  SVC="$RELEASE"
fi
ok "OpenShift MCP Service: http://${SVC}.${OPENCLAW_NS}.svc.cluster.local:8080/mcp"
"$KUBECTL" -n "$OPENCLAW_NS" get deploy,svc,po -l "app.kubernetes.io/instance=$RELEASE" 2>/dev/null \
  || "$KUBECTL" -n "$OPENCLAW_NS" get deploy,svc,po | grep -i mcp || true
