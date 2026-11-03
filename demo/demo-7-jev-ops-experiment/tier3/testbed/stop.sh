#!/usr/bin/env bash
# Stop the kind node containers (keeps the cluster). Start again with start.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/pins.env"
docker stop "${CLUSTER}-control-plane" "${CLUSTER}-worker" "${CLUSTER}-worker2" "${CLUSTER}-worker3"
