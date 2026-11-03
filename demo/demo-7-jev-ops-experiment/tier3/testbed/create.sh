#!/usr/bin/env bash
# Create the four-node kind cluster `jev` (k8s-testbed spec). Refuses if it exists.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/pins.env"
"$HERE/precheck.sh"

if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  echo "cluster '$CLUSTER' already exists; not recreating. To start over: $HERE/delete.sh"
  exit 1
fi
CONFIG="$HERE/kind.yaml"
[ "$CNI" = calico ] && CONFIG="$HERE/kind-calico.yaml"
kind create cluster --config "$CONFIG" --image "$NODE_IMAGE" --wait 5m
umask 077
kind get kubeconfig --name "$CLUSTER" > "$HERE/kubeconfig"
export KUBECONFIG="$HERE/kubeconfig"
if [ "$CNI" = calico ]; then
  kubectl apply -f "https://raw.githubusercontent.com/projectcalico/calico/${CALICO_VERSION}/manifests/calico.yaml"
  kubectl -n kube-system rollout status daemonset/calico-node --timeout=5m
fi
kubectl wait --for=condition=Ready nodes --all --timeout=5m
kubectl get nodes -L jev-role
