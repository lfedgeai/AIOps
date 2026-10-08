#!/usr/bin/env bash
#
# install-netobserv-aws.sh
# ---------------------------------------------------------------------------
# Interactive installer for Network Observability on an IPI-on-AWS OpenShift
# cluster, backed by a dedicated LokiStack on AWS S3 (non-STS / static creds).
#
# Deploys:  Loki Operator + LokiStack (S3)  ->  Network Observability Operator
#           + FlowCollector (eBPF agent, pipeline, console plugin)
#
# The script PROMPTS for all credentials at runtime. Nothing sensitive is
# passed on the command line, written to a temp file, or echoed to the screen.
# AWS keys reach the cluster only via a Secret applied over stdin (heredoc),
# so they never appear in the process list.
#
# Requirements on the machine running this: oc, aws, jq
# Run as a user who can obtain cluster-admin on the target cluster.
# ---------------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

# ---------------------------------------------------------------------------
# Tunables (override via environment before running, e.g. LOKI_SIZE=1x.medium)
# ---------------------------------------------------------------------------
LOKI_NS="${LOKI_NS:-netobserv-loki}"
NETOBSERV_NS="${NETOBSERV_NS:-netobserv}"
OPERATOR_NS="${OPERATOR_NS:-openshift-netobserv-operator}"
LOKI_SIZE="${LOKI_SIZE:-1x.extra-small}"          # 1x.pico | 1x.extra-small | 1x.small | 1x.medium
SAMPLING="${SAMPLING:-50}"                  # 1-in-N packet sampling
STORAGE_CLASS="${STORAGE_CLASS:-gp3-csi}"   # storage class for Loki PVCs
# Loki Operator's supported install namespace on OCP 4.20+ is openshift-operators-redhat.
LOKI_OPERATOR_NS="${LOKI_OPERATOR_NS:-openshift-operators-redhat}"
# Channels are auto-detected from the cluster's package manifest below.
# Set LOKI_CHANNEL / NETOBSERV_CHANNEL explicitly to override auto-detection.
LOKI_CHANNEL="stable-6.5"            # e.g. stable-6.3 ; empty => auto-detect newest stable-6.x
NETOBSERV_CHANNEL="${NETOBSERV_CHANNEL:-}" # empty => auto-detect (falls back to stable)
CATALOG_SOURCE="${CATALOG_SOURCE:-redhat-operators}"
CATALOG_NS="${CATALOG_NS:-openshift-marketplace}"
KMS_KEY_ID="${KMS_KEY_ID:-}"               # optional CMK ARN; empty => SSE-S3 (AES256)

LOKI_CSV_PREFIX="loki-operator"
NETOBSERV_CSV_PREFIX="network-observability-operator"

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
c_reset=$'\033[0m'; c_blue=$'\033[1;34m'; c_green=$'\033[1;32m'
c_yellow=$'\033[1;33m'; c_red=$'\033[1;31m'
step() { printf '%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
die()  { printf '%s[fail]%s %s\n' "$c_red" "$c_reset" "$*" >&2; exit 1; }

prompt() {  # prompt <varname> <message> [default]
  local __var="$1" __msg="$2" __def="${3:-}" __in
  if [[ -n "$__def" ]]; then read -rp "$__msg [$__def]: " __in; __in="${__in:-$__def}"
  else read -rp "$__msg: " __in; fi
  printf -v "$__var" '%s' "$__in"
}
prompt_secret() {  # prompt_secret <varname> <message>
  local __var="$1" __msg="$2" __in
  read -rsp "$__msg: " __in; echo
  [[ -n "$__in" ]] || die "A value is required for: $__msg"
  printf -v "$__var" '%s' "$__in"
}
confirm() {  # confirm <message> ; returns 0 on yes
  local __in; read -rp "$1 [y/N]: " __in; [[ "${__in,,}" == y || "${__in,,}" == yes ]]
}

# ---------------------------------------------------------------------------
# 0. Preflight: required tooling
# ---------------------------------------------------------------------------
step "Checking required CLIs"
for bin in oc aws jq; do
  command -v "$bin" >/dev/null 2>&1 || die "'$bin' not found in PATH."
done
ok "oc, aws, jq present."

# ---------------------------------------------------------------------------
# 1. OpenShift authentication
# ---------------------------------------------------------------------------
step "OpenShift cluster login"
if oc whoami >/dev/null 2>&1 && ! confirm "Already logged in as $(oc whoami) at $(oc whoami --show-server). Re-login?"; then
  ok "Reusing existing session."
else
  prompt OCP_API "OpenShift API server URL (https://api.<cluster>.<domain>:6443)"
  SKIP_TLS_FLAG=""
  if confirm "Skip TLS verification? (not recommended for FSI/Gov)"; then
    SKIP_TLS_FLAG="--insecure-skip-tls-verify=true"
  fi
  echo "Authentication method:"
  echo "  1) Token"
  echo "  2) Username / password"
  prompt AUTH_METHOD "Choose" "2"
  case "$AUTH_METHOD" in
    1) prompt_secret OCP_TOKEN "Login token (input hidden)"
       oc login "$OCP_API" --token="$OCP_TOKEN" $SKIP_TLS_FLAG >/dev/null \
         || die "Token login failed." ;;
    2) prompt OCP_USER "Username"
       # oc prompts for the password itself, keeping it out of argv/history.
       oc login "$OCP_API" -u "$OCP_USER" $SKIP_TLS_FLAG \
         || die "Username/password login failed." ;;
    *) die "Invalid authentication choice." ;;
  esac
  ok "Logged in as $(oc whoami)."
fi

# ---------------------------------------------------------------------------
# 2. Cluster preflight
# ---------------------------------------------------------------------------
step "Validating cluster prerequisites"
[[ "$(oc auth can-i '*' '*' --all-namespaces 2>/dev/null)" == "yes" ]] \
  || die "Current user is not cluster-admin."
ok "cluster-admin confirmed."

cni="$(oc get network.operator cluster -o jsonpath='{.spec.defaultNetwork.type}' 2>/dev/null || true)"
[[ "$cni" == "OVNKubernetes" ]] || die "Primary CNI is '$cni'; Network Observability requires OVN-Kubernetes."
ok "CNI is OVN-Kubernetes."

cred_mode="$(oc get cloudcredential cluster -o jsonpath='{.spec.credentialsMode}' 2>/dev/null || echo Unknown)"
if [[ "$cred_mode" == "Manual" ]]; then
  warn "Cloud credential mode is 'Manual' — this cluster may be STS-based."
  warn "This script uses STATIC S3 credentials. Continue only if that is intended."
  confirm "Proceed with static credentials?" || die "Aborted by user."
else
  ok "Cloud credential mode: ${cred_mode:-mint/passthrough}."
fi

if ! oc get storageclass "$STORAGE_CLASS" >/dev/null 2>&1; then
  warn "Storage class '$STORAGE_CLASS' not found. Available:"
  oc get storageclass -o name | sed 's#storageclass.storage.k8s.io/#  - #'
  prompt STORAGE_CLASS "Enter a storage class to use for Loki PVCs" "$STORAGE_CLASS"
  oc get storageclass "$STORAGE_CLASS" >/dev/null 2>&1 || die "Storage class '$STORAGE_CLASS' does not exist."
fi
ok "Loki PVC storage class: $STORAGE_CLASS."

CLUSTER_ID="$(oc get infrastructure cluster -o jsonpath='{.status.infrastructureName}')"
DEFAULT_REGION="$(oc get infrastructure cluster -o jsonpath='{.status.platformStatus.aws.region}' 2>/dev/null || true)"

# ---------------------------------------------------------------------------
# 3. AWS inputs
# ---------------------------------------------------------------------------
step "AWS S3 configuration"
prompt AWS_REGION "AWS region" "${DEFAULT_REGION:-ap-southeast-1}"
prompt S3_BUCKET "S3 bucket name for Loki" "netobserv-loki-${CLUSTER_ID}"
prompt AWS_ACCESS_KEY_ID "AWS Access Key ID"
prompt_secret AWS_SECRET_ACCESS_KEY "AWS Secret Access Key (input hidden)"

# Export for the aws CLI (env, not argv -> not visible in the process list)
export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_REGION AWS_DEFAULT_REGION="$AWS_REGION"

step "Verifying AWS credentials"
aws sts get-caller-identity >/dev/null 2>&1 || die "AWS credentials failed to authenticate."
ok "AWS credentials valid."

# ---------------------------------------------------------------------------
# 4. Create + harden the S3 bucket
# ---------------------------------------------------------------------------
step "Creating S3 bucket: $S3_BUCKET"
if aws s3api head-bucket --bucket "$S3_BUCKET" >/dev/null 2>&1; then
  warn "Bucket already exists and is owned by you — reusing."
else
  if [[ "$AWS_REGION" == "us-east-1" ]]; then
    aws s3api create-bucket --bucket "$S3_BUCKET" --region "$AWS_REGION" >/dev/null
  else
    aws s3api create-bucket --bucket "$S3_BUCKET" --region "$AWS_REGION" \
      --create-bucket-configuration LocationConstraint="$AWS_REGION" >/dev/null
  fi
  ok "Bucket created."
fi

step "Applying bucket hardening (public-access block + encryption + TLS-only)"
aws s3api put-public-access-block --bucket "$S3_BUCKET" \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

if [[ -n "$KMS_KEY_ID" ]]; then
  aws s3api put-bucket-encryption --bucket "$S3_BUCKET" \
    --server-side-encryption-configuration \
    "{\"Rules\":[{\"ApplyServerSideEncryptionByDefault\":{\"SSEAlgorithm\":\"aws:kms\",\"KMSMasterKeyID\":\"${KMS_KEY_ID}\"}}]}"
  ok "Default encryption: SSE-KMS ($KMS_KEY_ID)."
else
  aws s3api put-bucket-encryption --bucket "$S3_BUCKET" \
    --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
  ok "Default encryption: SSE-S3 (AES256). Set KMS_KEY_ID for a customer-managed key."
fi

aws s3api put-bucket-policy --bucket "$S3_BUCKET" --policy \
  "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Sid\":\"DenyInsecureTransport\",\"Effect\":\"Deny\",\"Principal\":\"*\",\"Action\":\"s3:*\",\"Resource\":[\"arn:aws:s3:::${S3_BUCKET}\",\"arn:aws:s3:::${S3_BUCKET}/*\"],\"Condition\":{\"Bool\":{\"aws:SecureTransport\":\"false\"}}}]}" >/dev/null
ok "TLS-only bucket policy applied."

# ---------------------------------------------------------------------------
# 5. (Optional) dedicated least-privilege IAM user for Loki
# ---------------------------------------------------------------------------
LOKI_KEY_ID="$AWS_ACCESS_KEY_ID"
LOKI_KEY_SECRET="$AWS_SECRET_ACCESS_KEY"

if confirm "Create a dedicated, bucket-scoped IAM user for Loki? (requires IAM rights on the supplied key)"; then
  step "Provisioning least-privilege IAM user"
  IAM_USER="loki-netobserv-${CLUSTER_ID}"
  POLICY_JSON="{\"Version\":\"2012-10-17\",\"Statement\":[{\"Sid\":\"LokiObjectStore\",\"Effect\":\"Allow\",\"Action\":[\"s3:ListBucket\",\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\"],\"Resource\":[\"arn:aws:s3:::${S3_BUCKET}\",\"arn:aws:s3:::${S3_BUCKET}/*\"]}]}"
  aws iam get-user --user-name "$IAM_USER" >/dev/null 2>&1 || aws iam create-user --user-name "$IAM_USER" >/dev/null
  aws iam put-user-policy --user-name "$IAM_USER" --policy-name loki-s3-access --policy-document "$POLICY_JSON"
  KEY_JSON="$(aws iam create-access-key --user-name "$IAM_USER")"
  LOKI_KEY_ID="$(jq -r '.AccessKey.AccessKeyId' <<<"$KEY_JSON")"
  LOKI_KEY_SECRET="$(jq -r '.AccessKey.SecretAccessKey' <<<"$KEY_JSON")"
  unset KEY_JSON
  ok "Scoped IAM user '$IAM_USER' created; Loki will use its access key."
  warn "This user has no key rotation automation — track it in your credential inventory."
fi

# ---------------------------------------------------------------------------
# 6. Namespaces + Loki storage Secret
# ---------------------------------------------------------------------------
step "Creating namespaces"
for ns in "$LOKI_NS" "$NETOBSERV_NS"; do
  oc get namespace "$ns" >/dev/null 2>&1 || oc create namespace "$ns" >/dev/null
done
ok "Namespaces ready: $LOKI_NS, $NETOBSERV_NS."

step "Creating Loki S3 Secret (applied via stdin; keys not in process list)"
oc apply -f - >/dev/null <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: loki-s3
  namespace: ${LOKI_NS}
stringData:
  access_key_id: ${LOKI_KEY_ID}
  access_key_secret: ${LOKI_KEY_SECRET}
  bucketnames: ${S3_BUCKET}
  endpoint: https://s3.${AWS_REGION}.amazonaws.com
  region: ${AWS_REGION}
EOF
unset LOKI_KEY_SECRET AWS_SECRET_ACCESS_KEY   # scrub secrets from the environment
ok "Secret loki-s3 created."

# ---------------------------------------------------------------------------
# Helper: wait for an operator CSV to reach Succeeded
# ---------------------------------------------------------------------------
wait_for_csv() {  # wait_for_csv <namespace> <csv-name-prefix> [timeout_s]
  local ns="$1" prefix="$2" timeout="${3:-600}" elapsed=0 phase
  step "Waiting for operator CSV '$prefix' in $ns (Succeeded)"
  while true; do
    phase="$(oc get csv -n "$ns" -o json 2>/dev/null \
      | jq -r --arg p "$prefix" '.items[] | select(.metadata.name|startswith($p)) | .status.phase' | head -1)"
    [[ "$phase" == "Succeeded" ]] && { ok "$prefix is Succeeded."; return 0; }
    (( elapsed >= timeout )) && die "Timed out waiting for CSV '$prefix' (last phase: ${phase:-none})."
    sleep 10; (( elapsed += 10 ))
  done
}

# ---------------------------------------------------------------------------
# Helper: resolve a subscription channel from the cluster's package manifest
# ---------------------------------------------------------------------------
csv_exists() {  # csv_exists <csv-name-prefix> ; true if a Succeeded CSV exists in any ns
  oc get csv -A -o json 2>/dev/null \
    | jq -e --arg p "$1" '.items[] | select(.metadata.name|startswith($p)) | select(.status.phase=="Succeeded")' >/dev/null 2>&1
}
resolve_channel() {  # resolve_channel <package> <regex-preferred> <fallback>
  local pkg="$1" prefer="$2" fallback="$3" chans picked default
  chans="$(oc get packagemanifest "$pkg" -n "$CATALOG_NS" \
            -o jsonpath='{range .status.channels[*]}{.name}{"\n"}{end}' 2>/dev/null || true)"
  if [[ -z "$chans" ]]; then echo "$fallback"; return; fi
  # newest channel matching the preferred pattern (version-sorted), else defaultChannel, else fallback
  picked="$(grep -E "$prefer" <<<"$chans" | sort -V | tail -1 || true)"
  if [[ -n "$picked" ]]; then echo "$picked"; return; fi
  default="$(oc get packagemanifest "$pkg" -n "$CATALOG_NS" -o jsonpath='{.status.defaultChannel}' 2>/dev/null || true)"
  echo "${default:-$fallback}"
}

# ---------------------------------------------------------------------------
# 7. Install Loki Operator + LokiStack
# ---------------------------------------------------------------------------
if csv_exists "$LOKI_CSV_PREFIX"; then
  ok "Loki Operator already installed on this cluster — reusing it."
else
  [[ -n "$LOKI_CHANNEL" ]] || LOKI_CHANNEL="$(resolve_channel loki-operator '^stable-6\.' stable-6.2)"
  step "Installing Loki Operator into $LOKI_OPERATOR_NS (channel: $LOKI_CHANNEL)"
  # openshift-operators-redhat is not a global OperatorGroup namespace by default,
  # so create the namespace (with monitoring label) and an AllNamespaces OperatorGroup.
  oc apply -f - >/dev/null <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: ${LOKI_OPERATOR_NS}
  labels:
    openshift.io/cluster-monitoring: "true"
---
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata:
  name: loki-operator
  namespace: ${LOKI_OPERATOR_NS}
spec: {}
---
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: loki-operator
  namespace: ${LOKI_OPERATOR_NS}
spec:
  channel: ${LOKI_CHANNEL}
  name: loki-operator
  source: ${CATALOG_SOURCE}
  sourceNamespace: ${CATALOG_NS}
  installPlanApproval: Automatic
EOF
  wait_for_csv "$LOKI_OPERATOR_NS" "$LOKI_CSV_PREFIX"
fi
oc wait --for=condition=established --timeout=300s crd/lokistacks.loki.grafana.com >/dev/null
ok "LokiStack CRD established."

step "Creating LokiStack (size: $LOKI_SIZE, tenant: openshift-network)"
oc apply -f - >/dev/null <<EOF
apiVersion: loki.grafana.com/v1
kind: LokiStack
metadata:
  name: loki
  namespace: ${LOKI_NS}
spec:
  size: ${LOKI_SIZE}
  storage:
    schemas:
      - version: v13
        effectiveDate: '2024-01-01'
    secret:
      name: loki-s3
      type: s3
  storageClassName: ${STORAGE_CLASS}
  tenants:
    mode: openshift-network
EOF

step "Waiting for LokiStack to become Ready (this can take a few minutes)"
if oc wait --for=condition=Ready lokistack/loki -n "$LOKI_NS" --timeout=600s >/dev/null 2>&1; then
  ok "LokiStack is Ready."
else
  warn "LokiStack not Ready within timeout — check: oc get pods -n $LOKI_NS"
fi

# ---------------------------------------------------------------------------
# 8. Install Network Observability Operator + FlowCollector
# ---------------------------------------------------------------------------
if csv_exists "$NETOBSERV_CSV_PREFIX"; then
  ok "Network Observability Operator already installed — reusing it."
else
  [[ -n "$NETOBSERV_CHANNEL" ]] || NETOBSERV_CHANNEL="$(resolve_channel netobserv-operator '^stable$' stable)"
  step "Installing Network Observability Operator into $OPERATOR_NS (channel: $NETOBSERV_CHANNEL)"
  oc get namespace "$OPERATOR_NS" >/dev/null 2>&1 || oc create namespace "$OPERATOR_NS" >/dev/null
  oc apply -f - >/dev/null <<EOF
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata:
  name: netobserv
  namespace: ${OPERATOR_NS}
spec: {}
---
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: netobserv-operator
  namespace: ${OPERATOR_NS}
spec:
  channel: ${NETOBSERV_CHANNEL}
  name: netobserv-operator
  source: ${CATALOG_SOURCE}
  sourceNamespace: ${CATALOG_NS}
  installPlanApproval: Automatic
EOF
  wait_for_csv "$OPERATOR_NS" "$NETOBSERV_CSV_PREFIX"
fi
oc wait --for=condition=established --timeout=300s crd/flowcollectors.flows.netobserv.io >/dev/null
ok "FlowCollector CRD established."

# Optional advanced eBPF features (privileged) for deeper Telco diagnostics
EBPF_BLOCK="      sampling: ${SAMPLING}"
if confirm "Enable advanced eBPF features (PacketDrop, DNSTracking, FlowRTT)? Requires privileged agent"; then
  EBPF_BLOCK="      sampling: ${SAMPLING}
      privileged: true
      features:
        - PacketDrop
        - DNSTracking
        - FlowRTT"
fi

step "Creating FlowCollector 'cluster'"
oc apply -f - >/dev/null <<EOF
apiVersion: flows.netobserv.io/v1beta2
kind: FlowCollector
metadata:
  name: cluster
spec:
  namespace: ${NETOBSERV_NS}
  deploymentModel: Direct
  agent:
    type: eBPF
    ebpf:
${EBPF_BLOCK}
  loki:
    mode: LokiStack
    lokiStack:
      name: loki
      namespace: ${LOKI_NS}
  consolePlugin:
    enable: true
EOF
ok "FlowCollector applied."

# ---------------------------------------------------------------------------
# 9. Verify
# ---------------------------------------------------------------------------
step "Verifying deployment"
sleep 15
echo "--- Operator CSVs ---"
oc get csv -A 2>/dev/null | grep -iE 'loki-operator|network-observability' || true
echo "--- Loki pods ($LOKI_NS) ---"
oc get pods -n "$LOKI_NS" 2>/dev/null || true
echo "--- Network Observability pods ($NETOBSERV_NS) ---"
oc get pods -n "$NETOBSERV_NS" 2>/dev/null || true
echo "--- FlowCollector status ---"
oc get flowcollector cluster -o jsonpath='{range .status.conditions[*]}{.type}={.status} {end}{"\n"}' 2>/dev/null || true

console_url="$(oc whoami --show-console 2>/dev/null || true)"
cat <<SUMMARY

${c_green}=============================================================${c_reset}
 Network Observability install complete.
   Loki namespace   : ${LOKI_NS}  (size ${LOKI_SIZE}, class ${STORAGE_CLASS})
   NetObserv ns     : ${NETOBSERV_NS}  (sampling 1-in-${SAMPLING})
   S3 bucket        : ${S3_BUCKET} (${AWS_REGION})

 View flows: ${console_url:-<console>}  ->  Observe -> Network Traffic
   (allow a minute or two for the eBPF agents to start streaming flows)

 If Loki pods are not yet Ready:  oc get pods -n ${LOKI_NS}
 If flows do not appear:          oc logs -n ${NETOBSERV_NS} -l app=flowlogs-pipeline
${c_green}=============================================================${c_reset}
SUMMARY
