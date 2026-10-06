#!/usr/bin/env bash
set -euo pipefail
NS=openclaw
SA=netobserv-grafana-bridge
IMG=registry.redhat.io/zero-trust-workload-identity-manager/spiffe-helper-rhel9

oc get secret pull-secret -n openshift-config -o jsonpath='{.data.\.dockerconfigjson}' | base64 -d > /tmp/pull.json
oc -n "$NS" create secret generic redhat-io-pull \
  --from-file=.dockerconfigjson=/tmp/pull.json \
  --type=kubernetes.io/dockerconfigjson \
  --dry-run=client -o yaml | oc apply -f -
oc -n "$NS" secrets link "$SA" redhat-io-pull --for=pull 2>/dev/null || true
oc -n "$NS" secrets link openclaw-hooks-mtls redhat-io-pull --for=pull 2>/dev/null || true

for tag in 1.1.0-1 1.1.0-2 1.1.0-3 1.1.0-4 1.1.0-5 1.1.0-6 1.1.0-7 1.1.0-8 1.1.0-9 1.1.0-10 \
           1.1.0-11 1.1.0-12 1.1.0-13 1.1.0-14 1.1.0-15 1.1.0-16 1.1.0-17 1.1.0-18 1.1.0-19 1.1.0-20; do
  name="tag-${tag//./-}"
  overrides=$(cat <<EOF
{"spec":{"serviceAccountName":"$SA","containers":[{"name":"t","image":"$IMG:$tag","command":["/spiffe-helper","-version"],"securityContext":{"allowPrivilegeEscalation":false,"capabilities":{"drop":["ALL"]},"runAsNonRoot":true,"seccompProfile":{"type":"RuntimeDefault"}}}]}}
EOF
)
  if out=$(oc run "$name" --rm -i --restart=Never -n "$NS" --image="$IMG:$tag" \
      --overrides="$overrides" --command -- /spiffe-helper -version 2>&1); then
    if ! echo "$out" | grep -qE 'manifest unknown|ErrImagePull|ImagePullBackOff'; then
      echo "GOOD_TAG=$tag"
      echo "$out" | tail -2
      exit 0
    fi
  fi
done
echo "NO_TAG"
exit 1
