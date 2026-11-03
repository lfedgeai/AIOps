#!/usr/bin/env bash
# Delete the kind cluster. Asks for confirmation unless --yes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/pins.env"
if [ "${1:-}" != "--yes" ]; then
  read -r -p "Delete kind cluster '$CLUSTER' and everything in it? Type yes: " answer
  [ "$answer" = yes ] || { echo "nothing deleted"; exit 1; }
fi
kind delete cluster --name "$CLUSTER"
rm -f "$HERE/kubeconfig" "$HERE/VERSION"
