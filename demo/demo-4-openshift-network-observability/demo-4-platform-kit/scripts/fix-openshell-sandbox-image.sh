#!/usr/bin/env bash
# Fix OpenShell agent sandbox ImagePullBackOff when the pinned digest was removed from Quay.
#
# Symptom: openclaw-agent-* stuck Init:ImagePullBackOff; Control UI exec hangs (stalled session).
# Cause: lab pins quay.io/ryan_nix/openclaw-openshift@sha256:… which may no longer exist;
#        OpenClaw config.yaml plugins.openshell.config.from is what spawns sandbox CRs
#        (openshell-config default_image alone is not enough for existing sessions).
#
# This script updates OpenShell to a pullable tag and restarts the gateway. Re-run seed if needed.
#
# Note: OpenClaw **gateway** pod (lab deployment.yaml) may still use a digest-pinned image
# that only exists on workers with a cached copy — pin deploy/openclaw nodeName if needed.
# This script updates the **sandbox agent** image only (openshell-config + lab values.yaml).
#
# Usage (bastion):
#   export DEMO_KIT_ROOT="${DEMO_KIT_ROOT:-$HOME/AIOps/demo/demo-4-platform-kit}"
#   cd "$DEMO_KIT_ROOT"
#   ./scripts/fix-openshell-sandbox-image.sh
#
# Env:
#   OPENSHELL_NS=openshell
#   SANDBOX_IMAGE=quay.io/ryan_nix/openclaw-openshift:2026.08.04
#   LAB_VALUES=~/labs/openshell-on-openshift-lab/manifests/openshell/values.yaml
#   LAB_CFG=~/labs/openshell-on-openshift-lab/manifests/openclaw/config.yaml
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=supply-chain-pins.env
source "$ROOT/supply-chain-pins.env" 2>/dev/null || true

OPENSHELL_NS="${OPENSHELL_NS:-openshell}"
OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
SANDBOX_IMAGE="${SANDBOX_IMAGE:-${OPENCLAW_SANDBOX_IMAGE:-quay.io/${QUAY_ORG:-your-org}/openclaw-openshift:openclaw-v2026.6.11}}"
GATEWAY_IMAGE="${GATEWAY_IMAGE:-${OPENCLAW_GATEWAY_IMAGE:-$SANDBOX_IMAGE}}"
OLD_DIGEST="${OLD_DIGEST:-sha256:a91dbc1cc1879a46137da1b73e5c270ea3094cdc3753f5ce8636168c9e1d8d0c}"
OLD_IMAGE="quay.io/ryan_nix/openclaw-openshift@${OLD_DIGEST}"
LAB_VALUES="${LAB_VALUES:-$HOME/labs/openshell-on-openshift-lab/manifests/openshell/values.yaml}"
LAB_CFG="${LAB_CFG:-$HOME/labs/openshell-on-openshift-lab/manifests/openclaw/config.yaml}"
LAB_DEPLOY="${LAB_DEPLOY:-$(dirname "$LAB_CFG")/deployment.yaml}"

c_blue=$'\033[1;34m'; c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_reset=$'\033[0m'
step() { printf '%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }

KUBECTL="$(command -v oc || command -v kubectl)"
[[ -n "$KUBECTL" ]] || { echo "oc/kubectl required" >&2; exit 1; }

step "Patch OpenShell config → sandbox image $SANDBOX_IMAGE"
"$KUBECTL" get cm openshell-config -n "$OPENSHELL_NS" -o yaml \
  | sed "s|${OLD_IMAGE}|${SANDBOX_IMAGE}|g" \
  | "$KUBECTL" apply -f -

if [[ -f "$LAB_VALUES" ]]; then
  step "Update lab values.yaml (if digest pinned)"
  if grep -q "$OLD_DIGEST" "$LAB_VALUES" 2>/dev/null; then
    sed -i.bak "s|${OLD_IMAGE}|${SANDBOX_IMAGE}|g" "$LAB_VALUES"
    ok "Updated $LAB_VALUES (backup: ${LAB_VALUES}.bak)"
  else
    warn "No old digest in $LAB_VALUES — skipped file patch"
  fi
else
  warn "Lab values not found at $LAB_VALUES — ConfigMap only"
fi

if [[ -f "$LAB_DEPLOY" ]]; then
  step "Update OpenClaw deployment.yaml gateway/init images → $GATEWAY_IMAGE"
  sed -i.bak \
    -e "s|${OLD_IMAGE}|${GATEWAY_IMAGE}|g" \
    -e "s|quay.io/ryan_nix/openclaw-openshift:2026\\.08\\.04|${GATEWAY_IMAGE}|g" \
    -e "s|quay.io/ryan_nix/openclaw-openshift:openclaw-v2026\\.7\\.1-2|${GATEWAY_IMAGE}|g" \
    -e "s|quay.io/ryan_nix/openclaw-openshift:latest|${GATEWAY_IMAGE}|g" \
    "$LAB_DEPLOY"
  ok "Updated $LAB_DEPLOY (backup: ${LAB_DEPLOY}.bak)"
fi

if [[ -f "$LAB_CFG" ]] && grep -qE "openshell|sandbox|${OLD_DIGEST}" "$LAB_CFG" 2>/dev/null; then
  step "Update OpenClaw config.yaml sandbox image → $SANDBOX_IMAGE"
  sed -i.bak \
    -e "s|${OLD_IMAGE}|${SANDBOX_IMAGE}|g" \
    -e 's|"from": "quay.io/ryan_nix/openclaw-openshift:[^"]*"|"from": "'"${SANDBOX_IMAGE}"'"|g' \
    "$LAB_CFG"
  ok "Updated $LAB_CFG"
fi

if [[ -d "$(dirname "$LAB_CFG")" ]]; then
  step "Re-apply OpenClaw deployment from lab manifests"
  if command -v kubectl >/dev/null 2>&1; then
    kubectl kustomize "$(dirname "$LAB_CFG")" \
      | sed "s/openshell\.openshell\.svc/openshell.${OPENSHELL_NS}.svc/" \
      | "$KUBECTL" -n "$OPENCLAW_NS" apply -f -
  else
    oc kustomize "$(dirname "$LAB_CFG")" \
      | sed "s/openshell\.openshell\.svc/openshell.${OPENSHELL_NS}.svc/" \
      | "$KUBECTL" -n "$OPENCLAW_NS" apply -f -
  fi
  step "Restart OpenClaw gateway (reload sandbox image in pod config)"
  "$KUBECTL" -n "$OPENCLAW_NS" rollout restart deploy/openclaw
  "$KUBECTL" -n "$OPENCLAW_NS" rollout status deploy/openclaw --timeout=900s
else
  warn "OpenClaw lab dir not found — skipped cluster re-apply"
fi

step "Restart OpenShell gateway"
"$KUBECTL" -n "$OPENSHELL_NS" rollout restart sts/openshell
"$KUBECTL" -n "$OPENSHELL_NS" rollout status sts/openshell --timeout=180s

step "Remove stuck agent sandbox pods and Sandbox CRs"
"$KUBECTL" -n "$OPENSHELL_NS" delete sandbox --all --ignore-not-found --wait=false 2>/dev/null || true
"$KUBECTL" -n "$OPENSHELL_NS" delete pod -l app.kubernetes.io/name=openclaw-agent --force --grace-period=0 2>/dev/null \
  || "$KUBECTL" -n "$OPENSHELL_NS" delete pod -l 'app in (openclaw-agent)' --force --grace-period=0 2>/dev/null \
  || true
for p in $("$KUBECTL" -n "$OPENSHELL_NS" get pods -o name 2>/dev/null | grep openclaw-agent || true); do
  phase="$("$KUBECTL" -n "$OPENSHELL_NS" get "$p" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  if [[ "$phase" != "Running" ]]; then
    "$KUBECTL" -n "$OPENSHELL_NS" delete "$p" --force --grace-period=0 2>/dev/null || true
  fi
done

ok "OpenShell sandbox image updated"
cat <<EOF

Next:
  1. ./scripts/seed-openclaw-netobserv-skills.sh   # if skills/MCP missing after gateway restart
  2. ./scripts/clear-openclaw-sessions.sh   # if Control UI was wedged on exec
  3. Hard refresh Control UI → /new → send one message
  4. ./scripts/openshell-sandbox-proof.sh wait

If ImagePullBackOff persists on multi-node clusters, pin deploy/openclaw to a worker
that already pulled the image:
  oc get pod -n $OPENCLAW_NS -l app.kubernetes.io/name=openclaw -o wide
  oc patch deploy openclaw -n $OPENCLAW_NS -p '{"spec":{"template":{"spec":{"nodeName":"NODE_NAME"}}}}'
EOF
