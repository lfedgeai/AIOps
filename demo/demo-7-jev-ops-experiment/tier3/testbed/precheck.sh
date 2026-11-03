#!/usr/bin/env bash
# Refuse to create or start the cluster without Docker, enough memory, or while
# Tier 2 runs (k8s-testbed spec: pre-check).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/pins.env"

command -v docker >/dev/null || { echo "error: docker is not installed"; exit 1; }
docker info >/dev/null 2>&1 || { echo "error: cannot talk to the docker daemon"; exit 1; }
if [ -n "$(docker ps -q --filter label=com.docker.compose.project=opentelemetry-demo)" ]; then
  echo "error: the Tier 2 demo (Docker Compose) is running. Stop it first: tier2/testbed/stop.sh"
  exit 1
fi
avail_gb=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
if [ "$avail_gb" -lt "$NEED_GB" ]; then
  echo "error: ${avail_gb} GB of memory available, Tier 3 needs ${NEED_GB} GB (kind nodes share host memory)."
  exit 1
fi
for tool in kind kubectl helm; do
  command -v "$tool" >/dev/null || { echo "error: $tool is not installed (see docs/k8s-testbed-notes.md)"; exit 1; }
done
have=$(kind version | awk '{print $2}')
[ "$have" = "$KIND_VERSION" ] || echo "warning: kind is $have, pinned $KIND_VERSION"
echo "pre-check ok: ${avail_gb} GB available, docker running, Tier 2 stopped"
