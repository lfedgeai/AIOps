#!/usr/bin/env bash
# Build and push the harness image.
#   ./scripts/build-image.sh              # podman
#   CONTAINER_ENGINE=docker ./scripts/build-image.sh
#   PUSH=0 ./scripts/build-image.sh       # build only
set -euo pipefail

. "$(dirname "$0")/lib.sh"

ENGINE="${CONTAINER_ENGINE:-podman}"
need "$ENGINE"
[[ "$HARNESS_IMAGE" != quay.io/your-org/* ]] || die "set HARNESS_IMAGE in .env first"

log "building $HARNESS_IMAGE"
"$ENGINE" build -t "$HARNESS_IMAGE" "$ROOT"

if [[ "${PUSH:-1}" =~ ^(1|true|yes)$ ]]; then
  log "pushing $HARNESS_IMAGE"
  "$ENGINE" push "$HARNESS_IMAGE"
fi
log "done"
