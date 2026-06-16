#!/usr/bin/env bash
# Optional: deploy an in-cluster nginx proxy in the EvalHub namespace that sets
# X-Tenant from the ?namespace= query when the header is empty (EvalHub auth otherwise
# returns "required header X-Tenant is missing").
#
# IMPORTANT — OpenShift AI: the Dashboard controller reconciles deployment/rhods-dashboard
# from the Dashboard CR and removes manual changes to the eval-hub-ui container (including
# --eval-hub-url=...) within seconds, even if the Deployment is rollout-paused. There is no
# supported field on OdhDashboardConfig / Dashboard to set the EvalHub BFF URL in current
# releases checked here. Until Red Hat ships a fix, use the EvalHub Route with
# Authorization + X-Tenant (see README §5) instead of relying on the federated UI list.
#
# This script only applies the proxy manifests (for lab use, e.g. if you can stop the
# reconciling operator temporarily and patch eval-hub-ui by hand).
#
# Usage:
#   EVALHUB_NAMESPACE=rhods-notebooks ./scripts/patch-rhods-dashboard-evalhub-ui-tenant-proxy.sh
#
# Remove proxy:
#   oc -n "${EVALHUB_NAMESPACE:-rhods-notebooks}" delete deploy,svc,cm -l app=evalhub-tenant-header-proxy
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS="${EVALHUB_NAMESPACE:-rhods-notebooks}"

subst() { sed "s/__EVALHUB_NAMESPACE__/${NS}/g" "$1"; }

echo "==> Applying optional EvalHub X-Tenant header proxy in namespace ${NS}"
subst "${ROOT}/evalhub/manifests/evalhub-tenant-header-proxy.configmap.yaml" | oc apply -f -
subst "${ROOT}/evalhub/manifests/evalhub-tenant-header-proxy.deployment.yaml" | oc apply -f -
oc rollout status "deploy/evalhub-tenant-header-proxy" -n "${NS}" --timeout=120s
echo "==> Proxy Service: https://evalhub-tenant-header-proxy.${NS}.svc.cluster.local:8443"
echo "==> Federated UI: OpenShift AI still cannot be fixed durably by patching rhods-dashboard (operator revert). Use EvalHub API with X-Tenant until a product fix exists."
