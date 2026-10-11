#!/bin/bash
# Keep ClickHouse PF alive with a fresh kubeadmin login (token kubeconfig expires).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
# shellcheck disable=SC1091
source "$ROOT/config/.env"
set +a
if [[ -n "${OCP_API_URL:-}" && -n "${KUBEADMIN_PASSWORD:-}" ]]; then
  oc login "$OCP_API_URL" \
    --username=kubeadmin --password="$KUBEADMIN_PASSWORD" \
    --insecure-skip-tls-verify >/dev/null 2>&1 || true
fi
# Ensure CH deployment is up (may have been scaled to 0)
oc scale deploy/clickhouse -n agentic-aiops --replicas=1 >/dev/null 2>&1 || true
exec /usr/local/bin/oc port-forward -n agentic-aiops svc/clickhouse 38123:8123
