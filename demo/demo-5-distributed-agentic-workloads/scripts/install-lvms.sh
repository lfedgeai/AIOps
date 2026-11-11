#!/usr/bin/env bash
# Install LVMS on the hub and wait for default StorageClass lvms-vg1.
# This SNO has only the OS disk — lab loop device /dev/loop7 (not reboot-persistent).
set -euo pipefail

DEMO5="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STOR="${DEMO5}/manifests/hub/storage"
NODE="$(oc get nodes -o jsonpath='{.items[0].metadata.name}')"

log() { printf '[install-lvms] %s\n' "$*"; }
die() { printf '[install-lvms] ERROR: %s\n' "$*" >&2; exit 1; }

command -v oc >/dev/null || die "oc not found"
oc whoami >/dev/null 2>&1 || die "oc not logged in"

log "Installing LVMS operator…"
oc apply -f "${STOR}/lvms-operator.yaml"

log "Waiting for LVMS CSV Succeeded…"
for i in $(seq 1 90); do
  phase="$(oc get csv -n openshift-storage -o jsonpath='{range .items[*]}{.metadata.name}{"="}{.status.phase}{"\n"}{end}' 2>/dev/null | awk -F= '/lvms/{print $2; exit}')"
  if [[ "${phase}" == "Succeeded" ]]; then
    log "CSV Succeeded"
    break
  fi
  if [[ "$i" -eq 90 ]]; then
    oc get csv -n openshift-storage >&2 || true
    die "LVMS CSV not Succeeded (last phase=${phase:-none})"
  fi
  sleep 10
done

log "Preparing lab loop device on node/${NODE}…"
oc debug "node/${NODE}" --quiet -- chroot /host bash -c '
  set -euo pipefail
  IMG=/var/lib/lvms-lab-disk.img
  LOOP=/dev/loop7
  if losetup "$LOOP" >/dev/null 2>&1; then
    echo "already: $(losetup "$LOOP")"
    lsblk "$LOOP"
    exit 0
  fi
  [[ -f "$IMG" ]] || truncate -s 40G "$IMG"
  losetup "$LOOP" "$IMG"
  losetup -a | grep loop7 || true
  lsblk "$LOOP"
'

log "Creating LVMCluster…"
oc apply -f "${STOR}/lvmcluster.yaml"

log "Waiting for StorageClass lvms-vg1…"
for i in $(seq 1 90); do
  if oc get sc lvms-vg1 >/dev/null 2>&1; then
    oc annotate sc lvms-vg1 storageclass.kubernetes.io/is-default-class=true --overwrite
    log "lvms-vg1 is default StorageClass"
    oc get sc
    oc get lvmcluster -n openshift-storage
    exit 0
  fi
  # surface progress
  if (( i % 6 == 0 )); then
    oc get lvmcluster -n openshift-storage -o jsonpath='{.items[0].status.state}{"\n"}' 2>/dev/null || true
  fi
  sleep 10
done
oc get lvmcluster -n openshift-storage -o yaml >&2 || true
die "StorageClass lvms-vg1 not created — check LVMCluster / loop device"
