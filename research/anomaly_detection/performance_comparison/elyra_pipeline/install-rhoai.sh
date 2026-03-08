#!/bin/bash
# Install Red Hat OpenShift AI (RHOAI) on an OpenShift cluster
# Target: api.sno1gpu.localdomain (or current KUBECONFIG cluster)
#
# Prerequisites (RHOAI 3.3):
# - OpenShift 4.20 or 4.21
# - Identity provider configured; cluster-admin user (kubeadmin NOT allowed)
# - Default storage class with dynamic provisioning
# - SNO: 32+ CPU, 128+ GiB RAM | Multi-node: 2+ workers with 8 CPU, 32 GiB each
# - Object storage (S3) for pipelines and model serving
#
# Docs: https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.3/

set -e
CHANNEL="${RHOAI_CHANNEL:-stable-3.3}"

echo "=== RHOAI Prerequisite Checks ==="

# 1. Cluster access
if ! oc whoami &>/dev/null; then
  echo "ERROR: Not logged in. Run: oc login"
  exit 1
fi
echo "✓ Logged in as: $(oc whoami)"

# 2. kubeadmin not allowed
USER=$(oc whoami 2>/dev/null)
if [[ "$USER" == "kube:admin" ]] || [[ "$USER" == "kubeadmin" ]]; then
  echo "ERROR: kubeadmin is not allowed for RHOAI. Configure an identity provider"
  echo "  and create a cluster-admin user. See:"
  echo "  https://docs.redhat.com/en/documentation/openshift_container_platform/latest/html/authentication_and_authorization/using-rbac#creating-cluster-admin_using-rbac"
  exit 1
fi
echo "✓ Not using kubeadmin"

# 3. Storage class
SC_DEFAULT=$(oc get storageclass -o jsonpath='{.items[?(@.metadata.annotations.storageclass\.kubernetes\.io/is-default-class=="true")].metadata.name}' 2>/dev/null || true)
if [ -z "$SC_DEFAULT" ]; then
  # Alternative: check for any storage class
  SC_ANY=$(oc get storageclass -o name 2>/dev/null | head -1)
  if [ -z "$SC_ANY" ]; then
    echo "ERROR: No storage class found. RHOAI requires a default storage class."
    echo "  Install ODF, local-storage, or another provisioner. See:"
    echo "  https://docs.redhat.com/en/documentation/openshift_container_platform/latest/html/storage/dynamic-provisioning"
    exit 1
  fi
  echo "WARNING: No default storage class. Consider setting one:"
  echo "  oc annotate storageclass <name> storageclass.kubernetes.io/is-default-class=true --overwrite"
fi
echo "✓ Storage class available"

# 4. Node resources (informational for SNO)
NODE_CPU_RAW=$(oc get node -o jsonpath='{.items[0].status.allocatable.cpu}' 2>/dev/null)
NODE_MEM_KI=$(oc get node -o jsonpath='{.items[0].status.allocatable.memory}' 2>/dev/null | sed 's/Ki$//')
NODE_COUNT=$(oc get node -o name 2>/dev/null | wc -l)
# Parse CPU: "7500m" -> 7, "32" -> 32
NODE_CPU=$([[ "$NODE_CPU_RAW" =~ ^([0-9]+)m$ ]] && echo $(( ${BASH_REMATCH[1]} / 1000 )) || echo "${NODE_CPU_RAW:-0}")

if [ -n "$NODE_MEM_KI" ] && [ "$NODE_MEM_KI" -gt 0 ] 2>/dev/null; then
  NODE_MEM_GI=$((NODE_MEM_KI / 1024 / 1024))
  if [ "$NODE_COUNT" -eq 1 ]; then
    if [ "${NODE_CPU:-0}" -lt 32 ] 2>/dev/null || [ "$NODE_MEM_GI" -lt 128 ]; then
      echo "WARNING: SNO requires 32+ CPU and 128+ GiB RAM for RHOAI."
      echo "  Current: ${NODE_CPU}m CPU, ~${NODE_MEM_GI} GiB RAM. Installation may fail or be unstable."
      if [ "${RHOAI_SKIP_RESOURCE_CHECK:-}" != "1" ]; then
        echo "  Set RHOAI_SKIP_RESOURCE_CHECK=1 to proceed anyway."
        exit 1
      fi
    fi
  fi
fi
echo "✓ Resource check done"

echo ""
echo "=== Installing RHOAI Operator (channel: $CHANNEL) ==="

# Namespace
oc create namespace redhat-ods-operator --dry-run=client -o yaml | oc apply -f -

# OperatorGroup
cat <<EOF | oc apply -f -
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata:
  name: rhods-operator
  namespace: redhat-ods-operator
spec:
  targetNamespaces:
  - redhat-ods-operator
EOF

# Subscription
cat <<EOF | oc apply -f -
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: rhods-operator
  namespace: redhat-ods-operator
spec:
  channel: $CHANNEL
  name: rhods-operator
  source: redhat-operators
  sourceNamespace: openshift-marketplace
EOF

echo ""
echo "Waiting for RHOAI Operator to succeed (this may take 5–10 minutes)..."
STATUS=""
for i in $(seq 1 60); do
  if oc get csv -n redhat-ods-operator --no-headers 2>/dev/null | grep -q Succeeded; then
    STATUS=Succeeded
    break
  fi
  STATUS=$(oc get csv -n redhat-ods-operator -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "pending")
  echo "  [$i/60] Phase: $STATUS"
  sleep 10
done

if [ "$STATUS" != "Succeeded" ]; then
  echo "ERROR: Operator did not reach Succeeded. Check: oc get csv -n redhat-ods-operator"
  exit 1
fi

echo ""
echo "=== Configuring Data Science Cluster with Workbenches + KServe ==="

# RHOAI 3.x: DataScienceCluster enables components. KServe requires cert-manager
# and Service Mesh (operator installs deps). Use RawDeployment for our Elyra pipeline.
oc create namespace redhat-ods-applications --dry-run=client -o yaml | oc apply -f -

# DataScienceCluster - enable workbenches (Elyra) and KServe (model serving)
cat <<'DSC' | oc apply -f -
apiVersion: datasciencecluster.opendatahub.io/v2
kind: DataScienceCluster
metadata:
  name: default-dsc
  namespace: redhat-ods-applications
spec:
  components:
    aipipelines:
      argoWorkflowsControllers:
        managementState: Removed
      managementState: Managed
    dashboard:
      managementState: Managed
    feastoperator:
      managementState: Removed
    kserve:
      managementState: Managed
      defaultDeploymentMode: RawDeployment
      serving:
        managementState: Removed
        name: knative-serving
    kueue:
      defaultClusterQueueName: default
      defaultLocalQueueName: default
      managementState: Removed
    llamastackoperator:
      managementState: Removed
    modelregistry:
      managementState: Removed
      registriesNamespace: rhoai-model-registries
    ray:
      managementState: Removed
    trainingoperator:
      managementState: Removed
    trustyai:
      managementState: Removed
    workbenches:
      managementState: Managed
      workbenchNamespace: rhods-notebooks
DSC

echo ""
echo "=== RHOAI installation initiated ==="
echo "Monitor progress:"
echo "  oc get dsc -n redhat-ods-applications"
echo "  oc get pods -n redhat-ods-applications"
echo "  oc get pods -n istio-system"
echo ""
echo "KServe InferenceService CRD will be available once installation completes."
echo "Then re-run your Elyra pipeline test (04_deploy_to_serving.ipynb)."
