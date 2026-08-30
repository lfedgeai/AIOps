#!/usr/bin/env bash
#
# netobserv-krkn-fault.sh
# ---------------------------------------------------------------------------
# Kraken-based fault injection for the NetObserv todo demo.
#
# Replaces the former Toxiproxy path with infrastructure chaos so the app
# topology stays clean (todo -> postgresql) and the incident is less
# deterministic for AI analysis.
#
# Mechanism:
# * Start a high-concurrency load generator against the todo API (healthy baseline)
# * On demand, inject network degradation with Kraken via podman/docker:
#     - MODE=pod  (default): pod_egress_shaping on the todo (or postgresql) pod
#       via tc/netem — elevates FlowRTT on the DB path without a proxy hop
#     - MODE=node: classic network-chaos (netem on the worker's br-ex). Prefer
#       MODE=pod when app pods are co-located on one worker (east-west may not
#       traverse br-ex).
# * Optional COMPOSITE=1 also applies mild CPU pressure on that worker
# * Capture while chaos is active via agent in Control UI (netobserv_capture_flows ~60s)
#
# Subcommands:
#   inject          Start heavy load only (HEALTHY baseline — no chaos yet)
#   slow [ms]       Inject latency (default LATENCY_MS) via Kraken
#   heal            Stop Kraken, delete chaos Jobs, clear residual tc/netem
#   status          Show loadgen, Kraken run state, and app health
#   restore         Full rollback: stop chaos + remove heavy loadgen
#
# Prerequisites:
#   - oc logged into the cluster
#   - podman or docker with pull access to quay.io/krkn-chaos/*
#
# Notes from bastion validation (OpenShift 4.20 + krknctl):
#   - Prefer direct podman mounts of a flattened kubeconfig. krknctl on this
#     host failed with "No configuration found" for /home/krkn/.kube/config.
#   - krkn-hub:pod-network-chaos ignores NETWORK_PARAMS and applies OVS drops
#     (outage), not latency — do not use it for slow.
# ---------------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=supply-chain-pins.env
source "$ROOT/scripts/supply-chain-pins.env"

APP_NS="${APP_NS:-todo-demo}"
CLIENT_NS="${CLIENT_NS:-todo-client}"
REGISTRY="${REGISTRY:-registry.access.redhat.com}"
CLIENT_IMAGE="${CLIENT_IMAGE:-${REGISTRY}/ubi9/ubi:latest}"
CONCURRENCY="${CONCURRENCY:-80}"
LATENCY_MS="${LATENCY_MS:-800}"
TARGET_PATH="${TARGET_PATH:-/api}"

MODE="${MODE:-pod}"                         # pod | node
TARGET="${TARGET:-todo}"                    # todo | postgresql
COMPOSITE="${COMPOSITE:-0}"                 # 0|1
TEST_DURATION="${TEST_DURATION:-${CHAOS_DURATION:-300}}"
WAIT_DURATION="${WAIT_DURATION:-$(( TEST_DURATION + 60 ))}"
LOSS="${LOSS:-}"
BANDWIDTH="${BANDWIDTH:-}"
CPU_LOAD_PERCENTAGE="${CPU_LOAD_PERCENTAGE:-40}"
CPU_HOG_DURATION="${CPU_HOG_DURATION:-$TEST_DURATION}"

KRKN_IMAGE="${KRKN_IMAGE}"
KRKN_HUB_NETWORK_IMAGE="${KRKN_HUB_NETWORK_IMAGE:-containers.krkn-chaos.dev/krkn-chaos/krkn-hub:network-chaos}"
KRKN_HUB_CPU_IMAGE="${KRKN_HUB_CPU_IMAGE:-containers.krkn-chaos.dev/krkn-chaos/krkn-hub:node-cpu-hog}"
KRKN_TOOLS_IMAGE="${KRKN_TOOLS_IMAGE}"

STATE_DIR="${STATE_DIR:-/tmp/netobserv-krkn-state}"
CONTAINER_NAME="${CONTAINER_NAME:-netobserv-krkn-slow}"
CPU_CONTAINER_NAME="${CPU_CONTAINER_NAME:-netobserv-krkn-cpu}"
KUBECONFIG_COPY="${KUBECONFIG_COPY:-/tmp/krkn-kubeconfig}"

c_reset=$'\033[0m'; c_blue=$'\033[1;34m'; c_green=$'\033[1;32m'
c_yellow=$'\033[1;33m'; c_red=$'\033[1;31m'
step() { printf '%s==>%s %s\n' "$c_blue" "$c_reset" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$*" >&2; }
die()  { printf '%s[fail]%s %s\n' "$c_red" "$c_reset" "$*" >&2; exit 1; }

command -v oc >/dev/null 2>&1 || die "'oc' not found in PATH."
oc whoami >/dev/null 2>&1 || die "Not logged in. Run 'oc login ...' first."

mkdir -p "$STATE_DIR/scenarios/custom"

# --- runtime helpers -------------------------------------------------------
pick_runtime() {
  if command -v podman >/dev/null 2>&1 && podman info >/dev/null 2>&1; then
    echo podman; return 0
  fi
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    echo docker; return 0
  fi
  return 1
}

RUNTIME="$(pick_runtime || true)"

ensure_runtime() {
  if ! RUNTIME="$(pick_runtime)"; then
    die "Need a working podman or docker runtime to run Kraken containers."
  fi
}

prep_kubeconfig() {
  # krkn container runs non-root; needs a world-readable flattened kubeconfig.
  rm -f "$KUBECONFIG_COPY" 2>/dev/null || true
  oc config view --flatten >"$KUBECONFIG_COPY" || die "Could not export kubeconfig to $KUBECONFIG_COPY"
  chmod 444 "$KUBECONFIG_COPY"
  echo "$KUBECONFIG_COPY"
}

target_label() {
  case "$TARGET" in
    todo|postgresql) echo "app=${TARGET}" ;;
    *) die "TARGET must be 'todo' or 'postgresql' (got: $TARGET)" ;;
  esac
}

target_pod() {
  local label pod
  label="$(target_label)"
  pod="$(oc get pods -n "$APP_NS" -l "$label" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  [[ -n "$pod" ]] || die "No running pod with $label in $APP_NS"
  echo "$pod"
}

target_node() {
  local pod node
  pod="$(target_pod)"
  node="$(oc get pod -n "$APP_NS" "$pod" -o jsonpath='{.spec.nodeName}')"
  [[ -n "$node" ]] || die "Could not resolve node for pod $pod"
  echo "$node"
}

stop_container() {
  local name="$1"
  ensure_runtime
  if "$RUNTIME" inspect "$name" >/dev/null 2>&1; then
    step "Stopping Kraken container $name"
    "$RUNTIME" stop -t 30 "$name" >/dev/null 2>&1 || true
    "$RUNTIME" rm -f "$name" >/dev/null 2>&1 || true
    ok "Stopped $name"
  fi
}

# Kraken schedules Jobs in the default namespace; leftover Completed jobs cause
# AlreadyExists on the next slow, and killing only the local podman container
# does not stop an already-running shaping Job (tc rules can linger).
delete_chaos_jobs() {
  step "Removing Kraken chaos Jobs/Pods in default"
  oc get jobs -n default --no-headers 2>/dev/null \
    | awk '/^chaos/ {print $1}' \
    | xargs -r oc delete job -n default --ignore-not-found --wait=false >/dev/null 2>&1 || true
  oc get pods -n default --no-headers 2>/dev/null \
    | awk '/chaos|krkn|network-chaos/ {print $1}' \
    | xargs -r oc delete pod -n default --ignore-not-found --force --grace-period=0 >/dev/null 2>&1 || true
}

# Best-effort: clear any residual netem qdiscs on the node that hosted the target.
clear_residual_tc() {
  local node=""
  [[ -f "$STATE_DIR/target-node" ]] && node="$(tr -d '[:space:]' <"$STATE_DIR/target-node")"
  [[ -n "$node" ]] || node="$(target_node 2>/dev/null || true)"
  [[ -n "$node" ]] || { warn "No target node recorded; skip residual tc clear."; return 0; }

  step "Best-effort clear of residual netem on node $node"
  oc delete job/netobserv-tc-cleanup -n default --ignore-not-found >/dev/null 2>&1 || true
  oc apply -n default -f - >/dev/null <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: netobserv-tc-cleanup
  namespace: default
spec:
  ttlSecondsAfterFinished: 60
  backoffLimit: 1
  template:
    spec:
      hostNetwork: true
      hostPID: true
      restartPolicy: Never
      nodeName: ${node}
      containers:
        - name: cleanup
          image: ${KRKN_TOOLS_IMAGE}
          securityContext:
            privileged: true
          command: ["/bin/bash", "-c"]
          args:
            - |
              set +e
              for i in \$(ls /sys/class/net 2>/dev/null); do
                tc qdisc show dev "\$i" 2>/dev/null | grep -q netem || continue
                echo "clearing netem on \$i"
                tc qdisc del dev "\$i" root 2>/dev/null
              done
              echo DONE
EOF
  oc wait --for=condition=complete job/netobserv-tc-cleanup -n default --timeout=90s >/dev/null 2>&1 \
    || warn "tc cleanup job did not complete in time (check: oc logs -n default job/netobserv-tc-cleanup)"
  oc delete job/netobserv-tc-cleanup -n default --ignore-not-found >/dev/null 2>&1 || true
}

write_pod_scenario() {
  local ms="$1" scenario="$STATE_DIR/scenarios/custom/pod_egress_shaping.yml"
  local loss_line="" bw_line=""
  [[ -n "$LOSS" ]] && loss_line="      loss: '${LOSS}'"
  [[ -n "$BANDWIDTH" ]] && bw_line="      bandwidth: ${BANDWIDTH}"

  cat >"$scenario" <<EOF
- id: pod_egress_shaping
  config:
    namespace: ${APP_NS}
    label_selector: '$(target_label)'
    instance_count: 1
    execution_type: parallel
    network_params:
      latency: ${ms}ms
${loss_line}
${bw_line}
    wait_duration: ${WAIT_DURATION}
    test_duration: ${TEST_DURATION}
    image: ${KRKN_TOOLS_IMAGE}
EOF
  echo "$scenario"
}

write_krkn_config() {
  local cfg="$STATE_DIR/config.yaml"
  local composite_block=""
  if [[ "$COMPOSITE" == "1" ]]; then
    composite_block=$(cat <<EOF
    - hog_scenarios:
        - scenarios/custom/cpu-hog.yml
EOF
)
  fi

  cat >"$cfg" <<EOF
kraken:
  distribution: openshift
  kubeconfig_path: /home/krkn/.kube/config
  exit_on_failure: False
  publish_kraken_status: False
  signal_state: RUN
  signal_address: 0.0.0.0
  port: 8081
  auto_rollback: True
  generate_pdf_report: False
  chaos_scenarios:
    - pod_network_scenarios:
        - scenarios/custom/pod_egress_shaping.yml
${composite_block}
cerberus:
  cerberus_enabled: False
performance_monitoring:
  prometheus_url: ''
  enable_alerts: False
  enable_metrics: False
  check_critical_alerts: False
tunings:
  wait_duration: 1
  iterations: 1
  daemon_mode: False
telemetry:
  enabled: False
  events_backup: False
  logs_backup: False
  prometheus_backup: False
elastic:
  enable_elastic: False
health_checks:
  interval: 2
  config: []
EOF
  echo "$cfg"
}

write_cpu_scenario() {
  local node="$1" scenario="$STATE_DIR/scenarios/custom/cpu-hog.yml"
  cat >"$scenario" <<EOF
duration: ${CPU_HOG_DURATION}
workers: ''
hog-type: cpu
image: ${KRKN_HOG_IMAGE}
namespace: default
cpu-load-percentage: ${CPU_LOAD_PERCENTAGE}
cpu-method: all
node-name: "${node}"
number-of-nodes: 1
taints: []
EOF
  echo "$scenario"
}

run_krkn_pod_mode() {
  local ms="$1"
  local kubeconfig scenario cfg node
  ensure_runtime
  kubeconfig="$(prep_kubeconfig)"
  scenario="$(write_pod_scenario "$ms")"
  node="$(target_node)"
  echo "$node" >"$STATE_DIR/target-node"
  echo "$(target_pod)" >"$STATE_DIR/target-pod"
  echo "$ms" >"$STATE_DIR/latency-ms"
  echo "pod" >"$STATE_DIR/mode"

  if [[ "$COMPOSITE" == "1" ]]; then
    write_cpu_scenario "$node" >/dev/null
  fi
  cfg="$(write_krkn_config)"

  stop_container "$CONTAINER_NAME"
  delete_chaos_jobs

  step "Starting Kraken pod_egress_shaping (${ms}ms) on $(target_label) for ${TEST_DURATION}s"
  local -a vols=(
    -v "${kubeconfig}:/home/krkn/.kube/config:Z"
    -v "${cfg}:/home/krkn/kraken/config/config.yaml:Z"
    -v "${scenario}:/home/krkn/kraken/scenarios/custom/pod_egress_shaping.yml:Z"
  )
  if [[ "$COMPOSITE" == "1" ]]; then
    vols+=(-v "${STATE_DIR}/scenarios/custom/cpu-hog.yml:/home/krkn/kraken/scenarios/custom/cpu-hog.yml:Z")
  fi

  "$RUNTIME" run -d --name "$CONTAINER_NAME" --net=host \
    "${vols[@]}" \
    "$KRKN_IMAGE" --config=config/config.yaml \
    >/dev/null || die "Failed to start $KRKN_IMAGE"

  echo "$CONTAINER_NAME" >"$STATE_DIR/container"
  ok "Kraken running as '$CONTAINER_NAME' (MODE=pod). Expect ~${ms}ms+ DB-path latency after ~30s settle."
}

run_krkn_node_mode() {
  local ms="$1"
  local kubeconfig node egress
  ensure_runtime
  kubeconfig="$(prep_kubeconfig)"
  node="$(target_node)"
  echo "$node" >"$STATE_DIR/target-node"
  echo "$(target_pod)" >"$STATE_DIR/target-pod"
  echo "$ms" >"$STATE_DIR/latency-ms"
  echo "node" >"$STATE_DIR/mode"

  egress="{latency: ${ms}ms}"
  [[ -n "$LOSS" ]] && egress="{latency: ${ms}ms, loss: ${LOSS}}"
  [[ -n "$BANDWIDTH" && -n "$LOSS" ]] && egress="{latency: ${ms}ms, loss: ${LOSS}, bandwidth: ${BANDWIDTH}}"
  [[ -n "$BANDWIDTH" && -z "$LOSS" ]] && egress="{latency: ${ms}ms, bandwidth: ${BANDWIDTH}}"

  stop_container "$CONTAINER_NAME"
  stop_container "$CPU_CONTAINER_NAME"
  delete_chaos_jobs

  warn "MODE=node shapes br-ex. Same-node east-west traffic may not see the delay — prefer MODE=pod."
  step "Starting Kraken network-chaos on $node (${ms}ms, ${TEST_DURATION}s)"
  "$RUNTIME" run -d --name "$CONTAINER_NAME" --net=host \
    -v "${kubeconfig}:/home/krkn/.kube/config:Z" \
    -e TRAFFIC_TYPE=egress \
    -e DURATION="$TEST_DURATION" \
    -e NODE_NAME="$node" \
    -e "EGRESS=${egress}" \
    -e EXECUTION=parallel \
    "$KRKN_HUB_NETWORK_IMAGE" \
    >/dev/null || die "Failed to start $KRKN_HUB_NETWORK_IMAGE"

  echo "$CONTAINER_NAME" >"$STATE_DIR/container"

  if [[ "$COMPOSITE" == "1" ]]; then
    step "Starting mild node CPU hog on $node (${CPU_LOAD_PERCENTAGE}% for ${CPU_HOG_DURATION}s)"
    "$RUNTIME" run -d --name "$CPU_CONTAINER_NAME" --net=host \
      -v "${kubeconfig}:/home/krkn/.kube/config:Z" \
      -e TOTAL_CHAOS_DURATION="$CPU_HOG_DURATION" \
      -e NODE_CPU_PERCENTAGE="$CPU_LOAD_PERCENTAGE" \
      -e NODE_SELECTOR="kubernetes.io/hostname=${node}" \
      -e NUMBER_OF_NODES=1 \
      "$KRKN_HUB_CPU_IMAGE" \
      >/dev/null || warn "CPU hog container failed to start."
    echo "$CPU_CONTAINER_NAME" >"$STATE_DIR/cpu-container"
  fi

  ok "Kraken running as '$CONTAINER_NAME' (MODE=node, COMPOSITE=$COMPOSITE)"
}

# --- subcommands -----------------------------------------------------------
cmd_inject() {
  step "Starting high-concurrency load (${CONCURRENCY} workers -> todo${TARGET_PATH})"
  oc apply -n "$CLIENT_NS" -f - >/dev/null <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: loadgen-heavy
  namespace: ${CLIENT_NS}
  labels:
    app: loadgen-heavy
spec:
  replicas: 1
  selector:
    matchLabels:
      app: loadgen-heavy
  template:
    metadata:
      labels:
        app: loadgen-heavy
    spec:
      securityContext:
        runAsNonRoot: true
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: loadgen
          image: ${CLIENT_IMAGE}
          env:
            - { name: CONCURRENCY, value: "${CONCURRENCY}" }
            - { name: TARGET,      value: "http://todo.${APP_NS}:8080${TARGET_PATH}" }
          command: ["/bin/bash", "-c"]
          args:
            - |
              echo "heavy loadgen: \$CONCURRENCY workers -> \$TARGET"
              run_worker() {
                while true; do
                  code=\$(curl -s -o /dev/null -w "%{http_code}" -m 6 "\$TARGET")
                  if [ "\$code" = "200" ]; then echo "200"; else echo "ERR \$code"; fi
                done
              }
              for i in \$(seq 1 "\$CONCURRENCY"); do run_worker & done
              wait
          resources:
            requests: { cpu: 50m, memory: 64Mi }
            limits:   { cpu: 500m, memory: 256Mi }
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
EOF
  oc rollout status deployment/loadgen-heavy -n "$CLIENT_NS" --timeout=120s || warn "loadgen-heavy not ready."

  cat <<MSG

${c_green}Baseline is live and HEALTHY${c_reset} (no Toxiproxy; topology remains todo -> postgresql).
  Induce the incident:   ./$(basename "$0") slow ${LATENCY_MS}
  Watch the app:         oc logs -n ${CLIENT_NS} deploy/loadgen-heavy -f
  Measure latency:       oc exec -n ${CLIENT_NS} deploy/loadgen-heavy -- \\
                           curl -s -o /dev/null -w 'code=%{http_code} time=%{time_total}\\n' -m 10 \\
                           http://todo.${APP_NS}:8080${TARGET_PATH}
  Chaos defaults:        MODE=${MODE} TARGET=${TARGET} COMPOSITE=${COMPOSITE} DURATION=${TEST_DURATION}s
MSG
}

cmd_slow() {
  local ms="${1:-$LATENCY_MS}"
  [[ "$ms" =~ ^[0-9]+$ ]] || die "Latency must be an integer number of milliseconds (got: $ms)"
  oc get deployment/loadgen-heavy -n "$CLIENT_NS" >/dev/null 2>&1 || die "Run 'inject' first."

  case "$MODE" in
    pod)  run_krkn_pod_mode "$ms" ;;
    node) run_krkn_node_mode "$ms" ;;
    *)    die "MODE must be 'pod' or 'node' (got: $MODE)" ;;
  esac

  cat <<MSG

${c_green}Chaos injected${c_reset} — allow ~30s for tc rules, then expect elevated DB-path RTT.
  Seed + UI:    ./scripts/netobserv-e2e-openclaw-test.sh seed
  Watch load:   oc logs -n ${CLIENT_NS} deploy/loadgen-heavy -f
  Kraken logs:  ${RUNTIME} logs -f ${CONTAINER_NAME}
  Recover:      ./$(basename "$0") heal
MSG
}

cmd_heal() {
  if RUNTIME="$(pick_runtime || true)" && [[ -n "$RUNTIME" ]]; then
    if [[ -f "$STATE_DIR/container" ]]; then
      stop_container "$(cat "$STATE_DIR/container")"
    fi
    if [[ -f "$STATE_DIR/cpu-container" ]]; then
      stop_container "$(cat "$STATE_DIR/cpu-container")"
    fi
    stop_container "$CONTAINER_NAME"
    stop_container "$CPU_CONTAINER_NAME"
  else
    warn "No container runtime available; will still clear cluster-side chaos Jobs/tc."
  fi

  delete_chaos_jobs
  clear_residual_tc

  ok "Chaos stopped. Verify recovery: oc exec -n ${CLIENT_NS} deploy/loadgen -- curl -s -o /dev/null -w 'time=%{time_total}s\\n' -m 10 http://todo.${APP_NS}:8080${TARGET_PATH}"
}

cmd_status() {
  step "Heavy loadgen"
  oc get pods -n "$CLIENT_NS" -l app=loadgen-heavy -o wide 2>/dev/null || echo "  (not deployed — run inject)"
  echo
  step "App pods"
  oc get pods -n "$APP_NS" -l 'app in (todo,postgresql)' -o wide 2>/dev/null || true
  echo
  step "Kraken state (${STATE_DIR})"
  if [[ -f "$STATE_DIR/mode" ]]; then
    echo "  mode:     $(cat "$STATE_DIR/mode")"
    echo "  latency:  $(cat "$STATE_DIR/latency-ms" 2>/dev/null || echo '?')ms"
    echo "  target:   $(cat "$STATE_DIR/target-pod" 2>/dev/null || echo '?') @ $(cat "$STATE_DIR/target-node" 2>/dev/null || echo '?')"
  else
    echo "  (no recorded slow run)"
  fi
  if RUNTIME="$(pick_runtime || true)" && [[ -n "$RUNTIME" ]]; then
    echo
    step "Kraken containers"
    "$RUNTIME" ps -a --filter "name=netobserv-krkn" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' 2>/dev/null || true
  fi
  echo
  step "Sample request latency"
  local probe_deploy=""
  if oc get deployment/loadgen-heavy -n "$CLIENT_NS" >/dev/null 2>&1; then
    probe_deploy=loadgen-heavy
  elif oc get deployment/loadgen -n "$CLIENT_NS" >/dev/null 2>&1; then
    probe_deploy=loadgen
  fi
  if [[ -n "$probe_deploy" ]]; then
    oc exec -n "$CLIENT_NS" "deploy/${probe_deploy}" -- \
      curl -s -o /dev/null -w 'code=%{http_code} time=%{time_total}\n' -m 10 \
      "http://todo.${APP_NS}:8080${TARGET_PATH}" 2>/dev/null || echo "  (probe failed)"
  else
    echo "  (no loadgen)"
  fi
  echo
  step "Recent loadgen sample"
  oc logs -n "$CLIENT_NS" deploy/loadgen-heavy --tail=15 2>/dev/null \
    || oc logs -n "$CLIENT_NS" deploy/loadgen --tail=15 2>/dev/null \
    || echo "  (no loadgen logs)"
}

cmd_restore() {
  cmd_heal
  step "Removing heavy load generator"
  oc delete deployment/loadgen-heavy -n "$CLIENT_NS" --ignore-not-found >/dev/null
  ok "Restored: chaos stopped, loadgen-heavy removed, todo -> postgresql unchanged."
}

case "${1:-}" in
  inject)  cmd_inject ;;
  slow)    shift; cmd_slow "${1:-}" ;;
  heal)    cmd_heal ;;
  status)  cmd_status ;;
  restore) cmd_restore ;;
  *)
    cat <<EOF
Usage: $(basename "$0") {inject|slow [ms]|heal|status|restore}

Environment (common):
  APP_NS=$APP_NS  CLIENT_NS=$CLIENT_NS  CONCURRENCY=$CONCURRENCY
  LATENCY_MS=$LATENCY_MS  TEST_DURATION=$TEST_DURATION
  MODE=$MODE          # pod (default) | node
  TARGET=$TARGET      # todo (default) | postgresql
  COMPOSITE=$COMPOSITE  # 0|1 — also apply mild CPU hog on the target node
  LOSS=               # optional packet loss percent for tc/netem (use 5 for clear drops)
  BANDWIDTH=          # optional, e.g. 10mbit
EOF
    exit 1
    ;;
esac
