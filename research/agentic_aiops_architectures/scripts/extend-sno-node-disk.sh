#!/usr/bin/env bash
# Grow the OpenShift SNO root disk (vda) to TARGET_SIZE and expand the XFS root filesystem.
#
# Prereq: hypervisor block device already >= TARGET_SIZE (live resize example):
#   virsh -c qemu:///system blockresize <vm-name> vda 500G
#
# Usage:
#   ./scripts/extend-sno-node-disk.sh [TARGET_SIZE] [NODE]
#   TARGET_SIZE defaults to 500G; NODE defaults to the sole cluster node.
set -euo pipefail

TARGET_SIZE="${1:-500G}"
NODE="${2:-$(oc get nodes -o jsonpath='{.items[0].metadata.name}')}"

log() { printf '[extend-sno-node-disk] %s\n' "$*"; }
die() { printf '[extend-sno-node-disk] ERROR: %s\n' "$*" >&2; exit 1; }

command -v oc >/dev/null || die "oc not found"
oc whoami >/dev/null 2>&1 || die "oc not logged in"

log "Growing partition/filesystem on node/${NODE} (target disk ${TARGET_SIZE})"
oc debug "node/${NODE}" --quiet -- chroot /host bash -s -- "${TARGET_SIZE}" <<'HOST'
set -euo pipefail
TARGET_SIZE="$1"
echo "Before:"
lsblk /dev/vda
df -h /var
growpart /dev/vda 4
mount -o remount,rw /sysroot
xfs_growfs /sysroot || xfs_growfs /var
echo "After:"
lsblk /dev/vda4
df -h /var
HOST

log "Done. Verify: oc debug node/${NODE} -- chroot /host df -h /var"
