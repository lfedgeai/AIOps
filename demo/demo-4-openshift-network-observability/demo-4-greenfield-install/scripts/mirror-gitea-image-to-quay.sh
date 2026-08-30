#!/usr/bin/env bash
# Mirror kwkoo Gitea image into your Quay namespace (one-time, any bastion with podman + quay login).
#
# Usage:
#   export QUAY_ORG=your-org
#   ./scripts/mirror-gitea-image-to-quay.sh
#
# Then install uses: quay.io/${QUAY_ORG}/gitea-openshift:gitea-openshift-v1 (supply-chain-pins.env)
set -euo pipefail

QUAY_ORG="${QUAY_ORG:-your-org}"
QUAY_REPO="${QUAY_REPO:-gitea-openshift}"
QUAY_TAG="${QUAY_TAG:-gitea-openshift-v1}"
SRC="${GITEA_UPSTREAM_IMAGE:-ghcr.io/kwkoo/gitea-openshift:latest}"
DEST="quay.io/${QUAY_ORG}/${QUAY_REPO}:${QUAY_TAG}"

c_blue=$'\033[1;34m'; c_green=$'\033[1;32m'; c_reset=$'\033[0m'
step() { printf '%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }

command -v podman >/dev/null || { echo "podman required" >&2; exit 1; }
podman login --get-login quay.io >/dev/null 2>&1 || {
  echo "Run: podman login quay.io (as ${QUAY_ORG})" >&2
  exit 1
}

step "Pull ${SRC}"
podman pull "$SRC"
step "Tag + push → ${DEST}"
podman tag "$SRC" "$DEST"
podman push "$DEST"
step "Verify"
skopeo inspect "docker://${DEST}" | jq -r '"digest=\(.Digest) created=\(.Created)"' 2>/dev/null \
  || podman pull "$DEST" >/dev/null
ok "Mirrored ${DEST}"

cat <<EOF

Install Gitea with:
  GITEA_IMAGE=${DEST%:*} GITEA_TAG=${DEST##*:} ./scripts/install-gitea.sh install
EOF
