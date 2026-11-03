#!/usr/bin/env bash
# Spread the demo's application pods back over both app nodes. node_taint drains one
# app node, and after the taint is removed nothing moves back, so a later node_taint
# on that node would evict nothing. This restarts every Deployment whose pods run on
# the app nodes; the scheduler places each new pod by the nodes' current load. Testbed upkeep
# only: never run during an experiment.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KUBECONFIG="${TIER3_KUBECONFIG:-$HERE/kubeconfig}"
NS="${TIER3_NAMESPACE:-otel-demo}"
deps=$(kubectl -n "$NS" get deploy -o jsonpath='{range .items[?(@.spec.template.spec.nodeSelector.jev-role=="app")]}{.metadata.name}{"\n"}{end}')
[ -n "$deps" ] || { echo "no app deployments found" >&2; exit 1; }
# One at a time: restarting all at once puts every new pod on the emptier node.
for d in $deps; do
  kubectl -n "$NS" rollout restart deploy/"$d" >/dev/null
  kubectl -n "$NS" rollout status deploy/"$d" --timeout=300s >/dev/null
done
kubectl -n "$NS" get pods -o wide --no-headers | awk '{print $7}' | sort | uniq -c
