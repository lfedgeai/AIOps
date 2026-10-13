#!/usr/bin/env bash
set -euo pipefail
EDGE_IP="${EDGE_IP:-10.0.0.20}"
EDGELAKE_URL="${EDGELAKE_URL:-http://${EDGE_IP}:32159}"
DASHBOARD_URL="${DASHBOARD_URL:-http://${EDGE_IP}:8080}"
echo '== Kubernetes nodes =='
if command -v kubectl >/dev/null; then kubectl get nodes -o wide; else echo 'kubectl unavailable; skipping'; fi
echo '== EdgeLake count =='
curl --fail --silent --show-error --max-time 10 "$EDGELAKE_URL" -H 'command: sql monitoring select count(*) from otel_edge_metrics'; echo
echo '== Dashboard health =='
curl --fail --silent --show-error --max-time 10 "$DASHBOARD_URL/health"; echo
echo '== Dashboard latest =='
curl --fail --silent --show-error --max-time 10 "$DASHBOARD_URL/api/latest"; echo
