# Tier 3 testbed notes (kind, LF Edge Kubernetes faults)

This file lists the facts Tier 3 depends on, where each came from, and what is still unverified. Task group 1 in `openspec/changes/archive/2026-11-02-k8s-faults-tier3/tasks.md` fills in the open items on a real cluster. Change `tier3/testbed/pins.env`, the scenarios or the code only together with this file.

## LF Edge harness (read 2026-10-14, task 1.5)

Source: `lfedgeai/AIOps`, `research/agentic_aiops_architectures/code/harness/run_harness.py`, commit `a408708` (2026-10-12). We only read it. We run nothing from it.

| Item | Their harness | Tier 3 |
|---|---|---|
| Fault selection | `--flag` is the fault type (default `scale_zero`), `--variant` the target deployment (default `cart`) | `fault.type`, `fault.target`; exported as `fault_flag`, `fault_variant` |
| Wait before the first poll | 15 s after injection (2 s without injection) | `telemetry_lag_s` 15 |
| Poll interval | `--poll-interval`, default 15 s | `poll_interval_s` 15 in every scenario |
| Detection window | `--detection-timeout`, default 1500 s (wall clock; skips a last poll that cannot finish) | `detection_window_s` 240: Jev answers in about a second, not minutes |
| Fault duration | `--fault-duration`, default 90 s, then recovery | fault held until detection or the window ends, then reset |
| MTTD | injection time to first alert (the agent's own timestamp if it reports one, else the harness clock) | `t_inject` to the decision time of the first detecting poll |
| Healthy control | `--skip-fault-injection` ("analyze existing telemetry only"); false alarms are not scored | `K000`, every poll scored for false alarms |

Per-run fields (`HarnessRun`) used by `python -m tier3 export-lfedge`: `run_id`, `fault_flag`, `fault_variant`, `fault_injection_time`, `first_alert_time`, `mttd_seconds`, `mttr_seconds`, `detected`, `suggested_remediations`, `suggested_root_cause`, `expected_root_cause`, `rca_correct`, `remediation_correct`, `remediation_declared`, `remediation_executed`, `recovery_verified`, `mttr_verified_seconds`, `approach`, `llm_invoked`, `agent_error`. Their MLflow logging writes `-1` for an unmeasured time; our export writes `null`.

### Their fault code, and what Tier 3 does differently

| Fault | Their inject (`inject_k8s_fault`) | Tier 3 |
|---|---|---|
| `scale_zero` | replicas 0 | same; restore the saved count (theirs restores 1) |
| `memory_limit` | every container's memory limit `10Mi` | same; a memory request above the limit is lowered too, or the API server rejects the change |
| `readiness_probe_fail` | every container: `exec: ["false"]` probe | same (period 5 s, threshold 1) |
| `config_corruption` | every container: `DATABASE_HOST=invalid-host-that-does-not-exist.local` **and `HARNESS_INJECTED_FAULT=config_corruption`** | `DATABASE_HOST` only; the second variable names the fault to any agent that reads the pod spec |
| `pvc_full` | `sh -c "dd ... of=/tmp/fill_disk bs=1M count=500 2>/dev/null \|\| true"` in a pod | same write to `/tmp/jev-fault-fill`, but a failed write fails the inject instead of being ignored |
| `node_taint` | taints the **first listed node** with `harness-fault=true:NoExecute`, replacing its taint list | adds `jev-fault=true:NoExecute` to an application worker only, keeps other taints, removes only its own |
| `replica_overload` | target replicas 0, `load-generator` replicas 3 | same; both saved counts restored |
| `dependency_removal` | target replicas 0 | same, target `valkey-cart` |
| `network_partition`, `kill_pod` | deny-ingress NetworkPolicy; delete the target's pods | same; policy named `jev-fault-np` |

## Host (task 1.1, 2026-10-14)

The host has 30 GB RAM, with about 9.2 GB available while the owner's other work runs. It has 16 cores (load about 1.2), an NVMe SSD with 359 GB free, and Docker 29.8.2. Tier 2 was stopped. Swap is zram (compressed RAM), so memory pressure costs CPU, not disk. We lowered the pre-check from 12 to 8 GB after trimming the demo (below). The measured footprint is recorded under task 1.4.

## Pinned versions (tasks 1.2-1.4)

| Component | Pin | Verified |
|---|---|---|
| kind | `v0.29.0` | installed to `~/.local/bin`, SHA-256 checked against the release file (2026-10-14). Newer releases exist (v0.33.0). The pin stays |
| node image | `kindest/node:v1.33.1@sha256:0500722...b34f` | the default image listed in the v0.29.0 release notes |
| kubectl | `v1.33.1` | from dl.k8s.io, SHA-256 checked |
| helm | `v3.22.0` | from get.helm.sh, SHA-256 checked (Helm 4 exists. v3 stays) |
| CNI | kindnet (kind default) | **enforces NetworkPolicy** (task 1.3): kind has shipped `kube-network-policies` since v0.24.0 |
| Demo chart | `open-telemetry/opentelemetry-demo` `0.42.4` (app 3.1.0) | 0.42.0-0.42.4 all carry app 3.1.0; newest taken |

## NetworkPolicy check (task 1.3, 2026-10-14)

busybox `httpd` pod on jev-worker2, client on jev-worker3. Before: `wget` gets the page. With a deny-all ingress policy on the server: `wget: download timed out` (packets dropped, not refused). After deleting the policy: the page again. So `network_partition` makes calls to the target time out. We expect callers to show latency and timeouts, not immediate connection errors.

## Chart facts (task 1.4, checked with `helm template` before install)

- Every demo service is a Deployment named after it (`cart`, `valkey-cart`, `load-generator`, ...). Pods carry `opentelemetry.io/name`, `app.kubernetes.io/component` and `app.kubernetes.io/name` all equal to the service name.
- Subchart pods carry only `app.kubernetes.io/name` and `/component`, with component values such as `server` (Prometheus) and `all-in-one` (Jaeger). The pod-to-service mapping therefore reads `opentelemetry.io/name` first (fixed 2026-10-14. Before that, it read `component` first).
- The collector is a DaemonSet `otel-collector-agent` (presets: hostMetrics, kubeletMetrics, kubernetesAttributes, clusterMetrics). It is pinned to `jev-worker`. Applications reach it through its Service, so it does not need a pod per node.
- Placement keys: `default.schedulingRules.nodeSelector`, `components.<name>.schedulingRules.nodeSelector`, `opentelemetry-collector.nodeSelector`, `jaeger.jaeger.nodeSelector` (Jaeger chart 4.13.1), `prometheus.server.nodeSelector`, `opensearch.nodeSelector`. The render puts every application Deployment on `jev-role=app` and `jaeger`, `prometheus`, `opensearch`, the collector, `flagd`, `load-generator` and `opamp-server` on `jev-role=observability`.
- Host NodePorts: `frontend-proxy` pods (`opentelemetry.io/name=frontend-proxy`, port 8080), Prometheus (`app.kubernetes.io/name=prometheus, component=server`, 9090), OpenSearch (`app.kubernetes.io/name=opensearch`, 9200). The selectors in `nodeports.yaml` match.
- Demo 3.1.0 in the chart adds `agent`, `chatbot`, `mcp` (LLM assistant), `astronomy-db` (Postgres for product-catalog), `opamp-server` and `telemetry-docs`.

### Trimmed for memory

Disabled in `values/demo.yaml`: `kafka`, `accounting` and `fraud-detection` (Kafka consumers), `agent`, `chatbot`, `mcp` (also not run in Tier 2: they live in `compose.agent.yaml`), `telemetry-docs` and Grafana. Together they carry about 3 GB of memory limits. None is on the path of any Tier 3 scenario. `checkout` publishes to Kafka only when `KAFKA_ADDR` is set (`src/checkout/main.go`), so its `KAFKA_ADDR` is overridden to empty and its `wait-for-kafka` init container removed. The trim keeps `astronomy-db` and `opamp-server` (product-catalog depends on both).

Tier 3 therefore has its own service list (`tier3/vocab.py`, 15 services). It is not Tier 2's list, which has `kafka`, `accounting`, `fraud-detection` and `postgresql`.

## Install and measured footprint (task 1.4, 2026-10-14)

`install.sh` with chart 0.42.4 took about 5 minutes (image pulls inside the kind nodes). Placement after install matched the render: every application pod on jev-worker2 or jev-worker3, every observability, flag, load and OpAMP pod on jev-worker, nothing on the control plane. `python -m tier3 health`: all eight checks ok.

Problems found and fixed in `values/demo.yaml`:

1. `ad` (Java) was OOMKilled 5 times in its first 5 minutes with the chart's 300Mi limit. We raised it to 500Mi. It has not restarted since.
2. No application telemetry arrived. Span metrics had only `flagd` and `frontend-web`, both on jev-worker. The collector is a DaemonSet, and in daemonset mode the chart gives its Service `internalTrafficPolicy: Local`. Because it is pinned to jev-worker, it has no pod on the application nodes, so their spans, metrics and logs were dropped. Set `opentelemetry-collector.service.internalTrafficPolicy: Cluster`. Afterwards, Jaeger lists every application service and span metrics cover the services under load.
3. The collector logged errors every few seconds (kafkametrics without Kafka, kubeletstats TLS on kind, and DNS for the disabled firepit) and shipped its own logs to OpenSearch as service `otelcol-contrib`: 461 warn/error lines in 5 minutes, which would have topped every observation's `error_logs`. We removed the kafkametrics receiver and the kubeletMetrics preset. `tier3/collect.py` now drops `otelcol-contrib`, `jaeger` (self-traces) and `opamp-server` from logs, traces, edges and service stats. The firepit DNS warning remains in the collector's own logs only.

Baseline after the fixes, with 10 users: span metrics exist for ad, cart, currency, frontend, frontend-proxy, image-provider, product-catalog, quote, recommendation and shipping (checkout, payment and email appear when orders are placed), all with error rate 0.000. Frontend is about 11 req/s, p95 about 37 ms.

Measured after settling (`docker stats`, 17:51):

| Node | Memory | CPU |
|---|---|---|
| jev-control-plane | 774 MiB | 10% of a core |
| jev-worker (observability, load) | 2.5 GiB | 16% |
| jev-worker2 (app) | 888 MiB | 8% |
| jev-worker3 (app) | 598 MiB | 15% |
| **total** | **about 4.7 GiB** | **about half a core** |

Startup peaked at load average 8.2 (16 cores) for a few minutes. Docker volumes grew by about 11 GB (images stored inside the kind nodes). The 8 GB pre-check leaves about 3 GB of headroom.

## Fault behaviour on the cluster (tasks 1.6-1.10, 2026-10-14)

Each fault was injected by hand with `tier3/testbed/probe_fault.py` (the real injector, ledger and dead-man timer, no engine), held for 240 s and sampled every 15 s: the harness's own collectors (span metrics over 120 s, Jaeger, OpenSearch, Kubernetes state), every new event of any type, and a user request to `/api/cart` through frontend-proxy. Records are in `results/tier3/probe/` (git-ignored). Times are seconds after inject. Exact restore read back cleanly every time.

| Fault (target) | User request | First signals | Later | Events | Recovered after reset |
|---|---|---|---|---|---|
| scale_zero (cart) | 500 from 15 s | cart 0/0; frontend error logs and failing traces at 15 s | span-metric errors from about 90 s: frontend about 20%, checkout 100%; cart drops out of span metrics | Normal only | 151 s |
| kill_pod (cart) | 200 throughout | new cart pod Ready by 15 s ("pods replaced") | frontend errors 1% between about 60 and 105 s, then baseline | Normal only | 0 s |
| memory_limit (cart) | 500 from 15 s | rollout completes (the 10Mi pod starts and passes readiness), old pod gone | checkout errors 67% then 100%, frontend about 20% from about 75 s; container exits Error at about 90 s, OOMKilled at about 165 s, then CrashLoopBackOff, 0 available | BackOff (Warning) from about 150 s | 120 s |
| network_partition (cart) | 200 throughout | nothing | nothing for 240 s | none | 0 s |
| readiness_probe_fail (cart) | 200 throughout | new pod never Ready, old pod keeps serving, rollout stuck "progressing" | unchanged; error rates at baseline | Unhealthy (Warning) every poll | 15 s |
| config_corruption (cart) | 200 throughout | a routine rolling update, new pod Ready by 15 s | nothing | Normal only | 15 s |
| dependency_removal (valkey-cart) | times out (over 10 s) from 15 s | valkey-cart 0/0 | frontend p95 about 3.6 s from about 105 s; cart errors 100% only from about 165 s; error rates otherwise near zero (calls hang, they do not fail) | Normal only | 75 s |
| replica_overload (cart) | 500 from 15 s | as scale_zero: cart 0/0 (the group's fault scales the target to zero and the load generator to three) | checkout 100%, frontend about 20% from about 60 s; more frontend error logs | Normal only | 120 s |
| node_taint (jev-worker2) | 200 throughout | 5 services evicted, replacements Ready on jev-worker3 by 15 s; node shows NoExecute and 0 pods | product-catalog errors about 15%, frontend about 3%, gone by about 150 s | Normal only (TaintManagerEviction) | 0 s |
| pvc_full (cart) | 500 at 15 s, 200 from 30 s | container OOMKilled mid-write and restarted ("pod restarted", last termination OOMKilled) | frontend errors about 8% between about 75 and 120 s, then baseline | one BackOff (Warning) | 0 s |

### What the probes settled

- Rolling updates (task 1.7). With one replica, the Deployment keeps the old pod until the new one is Ready. With `readiness_probe_fail`, the new pod is never Ready, so the old pod serves and users see nothing. The evidence is the stuck rollout and Unhealthy events. With `memory_limit`, the 10Mi pod does start and pass readiness, so the old pod is removed and users get errors at once. The OOM kills come later. `config_corruption` is an ordinary rollout.
- `config_corruption` (task 1.8). No service in demo 3.1.0 reads `DATABASE_HOST` (cart reads `VALKEY_ADDR`), and on the cluster the fault is a rolling restart and nothing else. For the group, their fault is a no-op on this demo version.
- `network_partition` is latent. kindnet (kube-network-policies) enforces the policy for new connections: a new TCP connect from the load-generator pod to `cart:8080` timed out after 5 s while the policy was in place, and connected at once after reset. But the frontend keeps a long-lived gRPC connection to cart, and the policy lets established connections through, so user traffic never notices. Calico and most iptables-based enforcers also accept established flows, so the group's harness likely sees the same unless the caller reconnects.
- `kill_pod` (task 1.9). The replacement is Ready before the first poll. A 1% error blip shows in span metrics for about 45 s. It heals itself.
- `pvc_full` (task 1.9). The cart pod has no PersistentVolumeClaim (no Tier 3 service has one) and `/tmp` is the container's writable layer on the host disk (929 GB, 344 GB free), so the write cannot fill a disk. The cart image has `sh` and `dd`. The 500 MB write fills page cache, which counts against cart's 160Mi memory limit. The container is OOMKilled mid-write (exit 137). It restarts with a fresh writable layer, so the file is gone. The injector treats exit 137 followed by a reported restart as the fault taking effect, and restore treats a restarted container as the file removed (it waits up to 30 s for the kubelet to report the restart). On the cluster, the result is one OOMKill and restart, a 500 for about 15 s, and a short error burst. It heals itself.
- Placement drifts. After `node_taint`, the evicted pods stay on jev-worker3 when the taint goes. A second `node_taint` on the empty node evicted nothing (the first rerun showed no effect at all). The health check now requires application pods on both app nodes, and `tier3/testbed/rebalance.sh` restarts the app Deployments one at a time to spread them again (restarting them all at once put every new pod on the empty node).
- Collection must not depend on an app node. The first `node_taint` probe crashed: Jaeger was queried through frontend-proxy (`/jaeger/ui`), and frontend-proxy was evicted. Jaeger now has its own NodePort (`host-jaeger`, 30686, host port 19686 in `kind.yaml`; `TIER3_JAEGER_URL`). A cluster created before this change has no host port for it. Set `TIER3_JAEGER_URL` to `http://<node IP>:30686` or recreate the cluster. Node IPs change when the containers restart (next item), so read the IP from `docker inspect` each time. The control plane's IP is a safe choice, since `node_taint` never targets it.
- Node IPs change on restart (2026-10-18). After `stop.sh` and `start.sh`, Docker gave the four nodes new IPs in a different order (the control plane went back to 172.27.0.2, and jev-worker3 took 172.27.0.5). The control plane's `/etc/kubernetes/kubelet.conf` held `server: https://172.27.0.5:6443`, so its kubelet hit worker3, got connection refused, and the node stayed `NotReady` (coredns pods on it stuck `Terminating`). The workers' kubelet.conf uses `https://jev-control-plane:6443` and was unaffected. The API server certificate lists `DNS:jev-control-plane`, so `start.sh` now points the control plane's kubelet at the name and restarts it. `start.sh` also waits for `/readyz` before `kubectl wait`, which otherwise fails at once while the API server is starting (this broke the scheduled 02:00 run).
- Most faults emit only Normal events. The observation carries Warning events only, so for scale_zero, dependency_removal, replica_overload, kill_pod, config_corruption and node_taint the engine sees replica counts, pod and node state and telemetry, not events. The only Warning events seen in all ten probes were `BackOff` (memory_limit) and `Unhealthy` with the message `Readiness probe failed: ` (readiness_probe_fail). No `OOMKilling` or `FailedScheduling` event appeared (jev-worker3 had room).
- No self-describing strings (task 1.10). Every event captured in the probes (all types) was scanned for `jev-fault`, `harness` and the ten fault names: none found. The NetworkPolicy and the taint key never appear in an event, and the readiness probe's command does not appear in the Unhealthy message. The leak-test fixtures in `tests/tier3/helpers.py` stay as written: they are a superset (they also plant the taint key and fault names) of what the cluster produced.

## Dropped or changed scenarios

No scenario was dropped: all ten of the group's faults stay in the catalogue so the export covers their full list. Expected outcomes changed after the probes (a change to the experiment, not a fix):

| ID | Was | Now | Why |
|---|---|---|---|
| K002 kill_pod | incident | no incident (still cart, pod_restart) | heals before the first poll; only a 1% blip |
| K004 network_partition | incident, cart, network_blocked | no incident, none, none | nothing observable; existing connections survive the policy |
| K006 config_corruption | incident, cart, misconfiguration | no incident, none, none | the variable is not read; only a routine rollout |
| K010 pvc_full | incident, cart, disk_full | no incident, cart, oom_killed | no volume to fill; the write gets the container OOMKilled once and it heals itself |

K003, K005, K007, K008 and K009 keep their expected outcomes. Their rationales now quote the measured behaviour. K005 stays an incident although users are not hurt: the rollout is broken and needs attention.

## Tier 4: remediation and MTTR (change `k8s-remediation-tier4`)

### How the group's harness measures MTTR (read 2026-10-18, task 1.1)

Source: `lfedgeai/AIOps`, `research/agentic_aiops_architectures/code/harness/run_harness.py` and `fault_verification.py`, commit `a408708`. We only read it.

| Field | Their harness | Tier 4 |
|---|---|---|
| `mttr_seconds` | fault injection to the agent's first successful remediation write (`remediation_executed_time`; "legacy" in their code) | `t_inject` to `t_action_done` (our write accepted; for restart and rollback, the rollout complete) |
| `mttr_verified_seconds` | fault injection to the first moment `verify_fault_resolved` passes, polled every 5 s for up to `RECOVERY_VERIFY_TIMEOUT_SEC` (120 s, capped at `fault_duration`, at least 30 s), no hold | `t_inject` to `t_recovered`: the first check of the final run of passing service-level checks, which must last at least 60 s and reach the end of a 360 s watch (checks every 15 s) |
| `recovery_verified` | true only if a successful write targeted the injected deployment (`write_targeted_expected`) and the check passed | true only for `recovered_by_action` |
| What the check looks at | the injected target's spec and readiness, per fault type: `scale_zero` replicas >= 1 and ready; `config_corruption`, `memory_limit`, `readiness_probe_fail` the fault's value gone and ready >= 1; `network_partition` their policy gone; `dependency_removal`, `replica_overload` ready >= 1; `kill_pod` target pods Running and Ready; any other type (`pvc_full`, `node_taint`) never passes | every Deployment at baseline availability and rollout complete, every app pod Ready, no service deviating from baseline; fault-agnostic, so `pvc_full` and `node_taint` can verify, and a fixed spec with a still-erroring service does not |
| Fault while verifying | stays injected until `fault_duration` after injection, then the harness reverts it | stays injected for the whole recovery window, then reset |

### Span metrics arrive every 60 s (2026-10-18)

`traces_span_metrics_calls_total` has one sample every 60 s (five samples over 5 minutes, gaps of exactly 60 s). `increase()` needs two samples in its window, so a service-stats query over 15, 30, 45, 60 or 90 s returns no series at all. A 120 s window returns every service. Tier 3 never noticed because its windows are 120 to 180 s. Tier 4's first 10 probes used a 30 s window, so the service part of their recovery check saw no services and could never fail. Kubernetes state alone decided those outcomes, and we discarded them. The recovery check now uses 120 s (about the last minute of traffic), fails when the baseline had services and the current window has none, and holds for 60 s so every verified recovery spans a fresh sample. A shorter metrics interval would need a collector or Prometheus change on the testbed (it would also change Tier 3's inputs), so we left it for a later change.

### A pass can relapse (2026-10-18)

Probe K004 (`network_partition` on cart) with `restart_service` on cart: every check passed for 60 s after the restart, so the first version of the watch stopped and called the run recovered. Prometheus afterwards showed frontend p95 at 15 s and checkout errors on 100 % of calls in the 120 s windows ending 120 and 180 s after the restart: the new cart pods take no new connections under the policy, and the callers' failures reached the span metrics only after their timeouts and the 60 s sample interval. The watch now runs the full 360 s, recovery needs the final passing run to reach the end, and an earlier pass that fails later is recorded as a relapse. The 10 probes run with the early stop were discarded too (`results/tier4/probe-discarded-early-stop/`).

### Action probes (tasks 1.3 to 1.5, 2026-10-18)

`python -m tier4 probe <ID> <action> <component>` runs the real Tier 4 path with a scripted engine (always "incident", the given action and component; no Jev). The matrix covers each fault on its Tier 3 target and each allowlisted action on that target, plus `restart_service` and `rollback_change` on `frontend` during the K000 control. There were 32 probes, all `ok`. Every action was saved, read back and restored, and no ledger was left. Records are in `results/tier4/probe/`, and the log is `results/tier4/probe-matrix.log`. MTTR is fault to action done. Verified is fault to the start of the final passing run (which must reach the end of the 360 s watch). Self is the self-recovery time when the check just before acting already passed.

| Fault (target) | `scale_up` | `restart_service` | `rollback_change` | Reading |
|---|---|---|---|---|
| K001 scale_zero (cart) | by action, MTTR 16 s, verified 106 s | not recovered | not recovered | fixed by `scale_up` only |
| K002 kill_pod (cart) | by action, verified 121 s | without action, self 17 s | without action, self 16 s | heals itself; "by action" when the new pod is not Ready yet at the pre-action check |
| K003 memory_limit (cart) | not recovered | not recovered (rollout never completes, 136 s) | by action, verified 18 s | fixed by `rollback_change` only |
| K004 network_partition (cart) | by action, verified 31 s | not recovered, relapsed | not recovered | latent: nothing removes the policy; restart and rollback replace cart's pods, which then take no new connections, so they turn the latent fault into an outage (checkout and frontend fail about 2 minutes after) |
| K005 readiness_probe_fail (cart) | not recovered | not recovered (136 s) | by action, verified 18 s | fixed by `rollback_change` only |
| K006 config_corruption (cart) | by action, verified 31 s | without action, self 16 s | without action, self 15 s | no effect beyond its own rolling restart; heals itself |
| K007 dependency_removal (valkey-cart) | by action, MTTR 16 s, verified 91 s | not recovered | not recovered | fixed by `scale_up` only |
| K008 replica_overload (cart) | by action, MTTR 16 s, verified 121 s | not recovered | not recovered | fixed by `scale_up` only; one cart replica copes with the tripled load |
| K009 node_taint (jev-worker2) | by action, verified 166 s | by action, verified 80 s | without action, self 17 s | heals itself by rescheduling onto jev-worker3 |
| K010 pvc_full (cart) | by action, verified 33 s | without action, self 18 s | without action, self 16 s | heals itself (the write is OOMKilled, the container restarts) |
| K000 control (frontend) | | no disturbance (25 of 25 checks passed) | no disturbance | a gradual rollout on a healthy Deployment is invisible to users |

What follows for the live runs:

- Right action, right component, one fix each. Replica faults (K001, K007, K008) are fixed only by `scale_up` on the right Deployment. Template faults (K003, K005) only by `rollback_change`. `restart_service` fixed nothing in any probe. No allowlisted action fixes K004.
- Self-healing faults (K002, K006, K009, K010) do not measure remediation. Whether a run is "by action" or "without action" depends only on whether the system is already passing at the check just before acting (about 16 to 20 s after injection). Report them separately and never count them toward an engine's verified MTTR.
- Verified MTTR equals MTTR when the fix lands before the fault's errors reach the span metrics (K003, K005 at about 18 s). The service part of the check does not yet see the errors, so it rests on Kubernetes state. Replica faults show errors first and verify 75 to 105 s after the fix (metrics every 60 s, 120 s window).
- `disturbed_healthy_system` is clean only on K000. On a no-incident fault scenario a failing check after the action can come from the fault itself (K002, K006, K010) or from the action (K004).
- Rollout history (1.4). `kubectl rollout undo` on cart picked the previous ReplicaSet revision every time (cart's revision grew from 80 to 132 over the matrix, from the faults, restores, restarts and rebalances). On the healthy `frontend` it rolled back to the revision `rebalance.sh` left (revision 12), which differs only in the restart annotation, so nothing visible changed. The first records stored the revision before the controller bumped it. We fixed this in `308eeca`.
- No self-describing markers (1.5). A rollout restart writes only `kubectl.kubernetes.io/restartedAt` on the pod template (read back by the executor). `kubectl rollout undo` copies the previous ReplicaSet's template and adds nothing. Neither names the fault or the experiment, and the harness's restore removes the restart annotation again.

### Recovery-check noise on the healthy cluster (task 1.6, 2026-10-18)

`python -m tier4 probe K000 no_action none` ran with no fault, no action, and the recovery check for the full 360 s. 25 of 25 checks passed (`results/tier4/noise/`). Together with the two K000 frontend probes (25 of 25 each), 75 checks ran on a healthy cluster and none failed, so the check does not fail on noise at the Tier 2 deviation thresholds. Nothing was tuned.
