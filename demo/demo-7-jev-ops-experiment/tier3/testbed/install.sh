#!/usr/bin/env bash
# Install the OpenTelemetry Demo chart release whose appVersion is 3.1.0, by values
# only, then the host NodePorts, then write VERSION from the running cluster.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/pins.env"
export KUBECONFIG="$HERE/kubeconfig"
PY="${PYTHON:-python3}"

helm repo add open-telemetry "$CHART_REPO" >/dev/null 2>&1 || true
helm repo update open-telemetry >/dev/null
if [ -z "$CHART_VERSION" ]; then
  CHART_VERSION=$(helm search repo "$CHART" --versions -o json | "$PY" -c '
import json, sys
want = sys.argv[1]
hits = [r["version"] for r in json.load(sys.stdin) if r.get("app_version") == want]
print(hits[0] if hits else "")' "$DEMO_APP_VERSION")
  [ -n "$CHART_VERSION" ] || { echo "error: no $CHART release has appVersion $DEMO_APP_VERSION"; exit 1; }
  echo "chart release for app $DEMO_APP_VERSION: $CHART_VERSION (pin it in pins.env)"
fi

kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || kubectl create namespace "$NAMESPACE"
helm upgrade --install "$RELEASE" "$CHART" --version "$CHART_VERSION" --namespace "$NAMESPACE" \
  -f "$HERE/values/demo.yaml" --wait --timeout 20m
kubectl apply -f "$HERE/nodeports.yaml"

app=$(helm list -n "$NAMESPACE" -o json | "$PY" -c 'import json,sys; print(json.load(sys.stdin)[0]["app_version"])')
k8s=$(kubectl version -o json | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["serverVersion"]["gitVersion"])')
image=$(docker inspect --format '{{.Config.Image}}' "${CLUSTER}-control-plane")
{
  echo "# Written by tier3/testbed/install.sh from the running cluster."
  echo "kind=$(kind version | awk '{print $2}')"
  echo "node_image=$image"
  echo "kubernetes=$k8s"
  echo "cni=$CNI"
  echo "chart=$CHART"
  echo "chart_version=$CHART_VERSION"
  echo "demo_app_version=$app"
} > "$HERE/VERSION"
cat "$HERE/VERSION"
echo "Placement (no app pod may run on ${CLUSTER}-worker or the control plane):"
kubectl -n "$NAMESPACE" get pods -o wide --no-headers | awk '{print $7, $1}' | sort
