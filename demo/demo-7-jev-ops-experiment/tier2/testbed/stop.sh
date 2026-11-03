#!/usr/bin/env bash
# Stop the demo and remove its volumes so the next start is clean.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/opentelemetry-demo"
export DEMO_VERSION="3.1.0"
FILES=(-f compose.yaml -f compose.full.yaml -f compose.observability.yaml)
[ "${TIER2_PROFILE:-full}" = minimal ] && FILES=(-f compose.yaml -f compose.observability.yaml)
docker compose --env-file .env --env-file .env.override "${FILES[@]}" down --remove-orphans --volumes
