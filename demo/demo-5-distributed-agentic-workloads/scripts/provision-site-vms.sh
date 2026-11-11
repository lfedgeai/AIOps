#!/usr/bin/env bash
# Provision CNV VM site-a (MicroShift). Requires CNV/CDI Available + lvms-vg1.
# RHSM: set in demo-5 .env (gitignored) or export RHSM_ORG+RHSM_ACTIVATION_KEY
#       (or RHSM_USER+RHSM_PASSWORD).
set -euo pipefail

DEMO5="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${DEMO5}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a && source "${DEMO5}/.env" && set +a
fi

NS=demo5-sites
SITE_ID="${SITE_ID:-site-a}"
VM=demo5-site-a
IMAGE_URL="${RHEL_GUEST_IMAGE:-docker://registry.redhat.io/rhel9/rhel-guest-image:9.6}"

log() { printf '[provision-site-vms] %s\n' "$*"; }
die() { printf '[provision-site-vms] ERROR: %s\n' "$*" >&2; exit 1; }

command -v oc >/dev/null || die "oc not found"
oc whoami >/dev/null 2>&1 || die "oc not logged in"
oc get sc lvms-vg1 >/dev/null || die "lvms-vg1 missing — run scripts/install-lvms.sh"
oc get hyperconverged kubevirt-hyperconverged -n openshift-cnv >/dev/null || die "CNV missing"
oc get crd datavolumes.cdi.kubevirt.io >/dev/null || die "CDI not ready — wait for CNV"

PASS_HASH="$(openssl passwd -6 microshift 2>/dev/null || python3 -c 'import crypt; print(crypt.crypt("microshift", crypt.METHOD_SHA512))')"

RHSM_BOOT=""
if [[ -n "${RHSM_ORG:-}" && -n "${RHSM_ACTIVATION_KEY:-}" ]]; then
  RHSM_BOOT=$(cat <<EOS
subscription-manager register --org='${RHSM_ORG}' --activationkey='${RHSM_ACTIVATION_KEY}' || true
EOS
)
elif [[ -n "${RHSM_USER:-}" && -n "${RHSM_PASSWORD:-}" ]]; then
  RHSM_BOOT=$(cat <<EOS
subscription-manager register --username='${RHSM_USER}' --password='${RHSM_PASSWORD}' --auto-attach || true
EOS
)
else
  log "WARN: set RHSM_ORG+RHSM_ACTIVATION_KEY (or RHSM_USER+RHSM_PASSWORD) so MicroShift repos can install"
fi

log "Namespace ${NS}"
oc create namespace "${NS}" --dry-run=client -o yaml | oc apply -f -
oc label namespace "${NS}" app.kubernetes.io/part-of=demo-5 --overwrite

log "Pull secret for registry.redhat.io"
oc -n openshift-config get secret pull-secret -o jsonpath='{.data.\.dockerconfigjson}' | base64 -d >/tmp/demo5-pull.json
oc -n "${NS}" create secret generic pull-secret \
  --from-file=.dockerconfigjson=/tmp/demo5-pull.json \
  --type=kubernetes.io/dockerconfigjson \
  --dry-run=client -o yaml | oc apply -f -
rm -f /tmp/demo5-pull.json

# Escape for YAML block
USERDATA=$(cat <<EOF
#cloud-config
hostname: demo5-site-a
manage_etc_hosts: true
users:
  - name: cloud-user
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    lock_passwd: false
    passwd: "${PASS_HASH}"
    groups: wheel
ssh_pwauth: true
chpasswd:
  expire: false
write_files:
  - path: /usr/local/bin/bootstrap-microshift.sh
    permissions: "0755"
    content: |
      #!/bin/bash
      set -euo pipefail
      exec > >(tee -a /var/log/bootstrap-microshift.log) 2>&1
      echo "=== demo-5 MicroShift bootstrap \$(date -Is) ==="
      ${RHSM_BOOT}
      ARCH=\$(uname -m)
      for ver in 4.18 4.17 4.16; do
        subscription-manager repos --enable="rhocp-\${ver}-for-rhel-9-\${ARCH}-rpms" --enable="fast-datapath-for-rhel-9-\${ARCH}-rpms" && break
      done
      dnf install -y microshift openshift-clients firewalld
      systemctl enable --now firewalld || true
      systemctl enable --now microshift
      echo "DONE kubeconfig=/var/lib/microshift/resources/kubeadmin/kubeconfig"
runcmd:
  - [/usr/local/bin/bootstrap-microshift.sh]
EOF
)

oc -n "${NS}" create secret generic site-a-cloudinit \
  --from-literal=userdata="${USERDATA}" \
  --from-literal=networkdata=$'version: 2\nethernets:\n  eth0:\n    dhcp4: true\n' \
  --dry-run=client -o yaml | oc apply -f -

log "DataVolume + VM (image ${IMAGE_URL})"
oc apply -f - <<EOF
apiVersion: cdi.kubevirt.io/v1beta1
kind: DataVolume
metadata:
  name: site-a-rootdisk
  namespace: ${NS}
  labels:
    site_id: ${SITE_ID}
spec:
  pvc:
    accessModes: ["ReadWriteOnce"]
    resources:
      requests:
        storage: 40Gi
    storageClassName: lvms-vg1-immediate
  source:
    registry:
      url: "${IMAGE_URL}"
      # pullMethod node avoids dockerconfig→accessKeyId CDI bug; image pulled on node.
      pullMethod: node
---
apiVersion: kubevirt.io/v1
kind: VirtualMachine
metadata:
  name: ${VM}
  namespace: ${NS}
  labels:
    site_id: ${SITE_ID}
    app.kubernetes.io/part-of: demo-5
spec:
  running: true
  template:
    metadata:
      labels:
        kubevirt.io/domain: ${VM}
        site_id: ${SITE_ID}
    spec:
      domain:
        cpu:
          cores: 2
        resources:
          requests:
            memory: 8Gi
            cpu: "2"
          limits:
            memory: 8Gi
            cpu: "2"
        devices:
          disks:
            - name: rootdisk
              disk:
                bus: virtio
            - name: cloudinitdisk
              disk:
                bus: virtio
          interfaces:
            - name: default
              masquerade: {}
        machine:
          type: q35
        firmware:
          bootloader:
            efi:
              secureBoot: false
      networks:
        - name: default
          pod: {}
      volumes:
        - name: rootdisk
          dataVolume:
            name: site-a-rootdisk
        - name: cloudinitdisk
          cloudInitNoCloud:
            secretRef:
              name: site-a-cloudinit
EOF

log "Waiting for DataVolume Ready (image import)…"
oc -n "${NS}" wait dv/site-a-rootdisk --for=condition=Ready --timeout=45m || {
  oc -n "${NS}" get dv,pvc,importerpods 2>&1 || oc -n "${NS}" get pods 2>&1 || true
  die "DataVolume not Ready"
}

log "Waiting for VMI Running…"
for i in $(seq 1 90); do
  phase=$(oc -n "${NS}" get vmi "${VM}" -o jsonpath='{.status.phase}' 2>/dev/null || true)
  echo "VMI phase=${phase:-pending}"
  [[ "${phase}" == "Running" ]] && break
  sleep 10
done
[[ "$(oc -n "${NS}" get vmi "${VM}" -o jsonpath='{.status.phase}' 2>/dev/null)" == "Running" ]] \
  || die "VMI not Running"

log "OK: ${VM} Running. console: oc -n ${NS} console vmi/${VM}"
log "Login cloud-user / microshift — check /var/log/bootstrap-microshift.log"
log "Then register with RHACM or Argo (scripts/bootstrap-site-k8s.sh)."
