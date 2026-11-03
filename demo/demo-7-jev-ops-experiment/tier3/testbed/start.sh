#!/usr/bin/env bash
# Start the kind node containers stopped by stop.sh, then wait for the nodes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/pins.env"
"$HERE/precheck.sh"
export KUBECONFIG="$HERE/kubeconfig"
CP="${CLUSTER}-control-plane"
docker start "$CP" "${CLUSTER}-worker" "${CLUSTER}-worker2" "${CLUSTER}-worker3"

# Docker can hand the nodes different IPs on restart. The control plane's
# kubelet.conf may hold an old IP (now another node's); point it at the node
# name, as the workers do, so it reaches the API server whatever the IPs are.
# The API server certificate covers the name (docs/k8s-testbed-notes.md).
if ! docker exec "$CP" grep -q "server: https://${CP}:6443" /etc/kubernetes/kubelet.conf; then
  echo "control-plane kubelet.conf uses an IP; pointing it at ${CP}:6443"
  docker exec "$CP" sed -i "s#server: https://[0-9.]*:6443#server: https://${CP}:6443#" /etc/kubernetes/kubelet.conf
  docker exec "$CP" systemctl restart kubelet
fi

# kubectl wait fails at once while the API server is still starting; retry first.
for _ in $(seq 60); do
  kubectl get --raw /readyz >/dev/null 2>&1 && break
  sleep 5
done
kubectl get --raw /readyz >/dev/null || { echo "error: API server not ready after 5 minutes"; exit 1; }
kubectl wait --for=condition=Ready nodes --all --timeout=5m
echo "nodes ready. Wait for the demo pods, then run ./health-check-k8s.sh"
