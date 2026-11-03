#!/usr/bin/env bash
# Clone the OpenTelemetry Demo at the pinned tag, check the host, pull images.
# Usage: tier2/testbed/setup.sh [--start]     TIER2_PROFILE=minimal for the small stack
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$HERE/opentelemetry-demo"
TAG="3.1.0"
PROFILE="${TIER2_PROFILE:-full}"
NEED_GB=8; [ "$PROFILE" = minimal ] && NEED_GB=4

command -v docker >/dev/null || { echo "error: docker is not installed"; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "error: docker compose plugin is missing"; exit 1; }
docker info >/dev/null 2>&1 || { echo "error: cannot talk to the docker daemon"; exit 1; }

avail_gb=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
if [ "$avail_gb" -lt "$NEED_GB" ]; then
  echo "error: ${avail_gb} GB of memory available, the '$PROFILE' profile needs about ${NEED_GB} GB."
  [ "$PROFILE" = full ] && echo "       Try: TIER2_PROFILE=minimal $0   (no Kafka, accounting or fraud-detection;"\
" the kafka scenario is not available)"
  exit 1
fi

if [ ! -d "$DEMO/.git" ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/open-telemetry/opentelemetry-demo.git "$DEMO"
fi
have=$(git -C "$DEMO" describe --tags 2>/dev/null || echo unknown)
[ "$have" = "$TAG" ] || { echo "error: demo checkout is at '$have', expected $TAG"; exit 1; }

# The demo's .env says DEMO_VERSION=latest. A shell variable overrides it without
# touching the checkout, so the images match the tag.
export DEMO_VERSION="$TAG"
cd "$DEMO"
FILES=(-f compose.yaml -f compose.observability.yaml)
[ "$PROFILE" = minimal ] || FILES=(-f compose.yaml -f compose.full.yaml -f compose.observability.yaml)
COMPOSE=(docker compose --env-file .env --env-file .env.override "${FILES[@]}")
"${COMPOSE[@]}" pull
echo "$TAG" > "$HERE/VERSION"

if [ "${1:-}" = "--start" ]; then
  "${COMPOSE[@]}" up --detach --remove-orphans
  "${COMPOSE[@]}" ps --format '{{.Service}} {{.Image}}' | sort > "$HERE/IMAGES"
  echo "started. Wait about 3 minutes for span metrics, then run ./health-check.sh"
else
  echo "images pulled. Start with: $0 --start"
fi
