#!/usr/bin/env bash
# Render and apply the in-cluster harness Job (one cell).
# Edit HARNESS_ARGS in .env or the environment to change the cell.
set -euo pipefail

. "$(dirname "$0")/lib.sh"
oc_login
# Job specs are immutable; recreate on each invocation.
"$(kube)" delete job aiops-harness -n "$AIOPS_NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
apply_manifest "$ROOT/manifests/harness-job.yaml"
log "Job submitted. Follow it with:"
log "  oc logs -n $AIOPS_NAMESPACE job/aiops-harness -f"
log "  oc get job,pod -n $AIOPS_NAMESPACE -l app=aiops-harness"
