# TOOLS.md - NetObserv demo

## Tool scope

| Question | Use |
|----------|-----|
| User reports slowness / DB problems (first message) | **netobserv-investigate** skill |
| “Is the **OpenShift cluster** healthy?” / operators / nodes | **`netobserv-openshift`** `openshift_cluster_health` **and** **`openshift-mcp`** core read (e.g. `events_list`) during investigate |
| What’s in **todo-demo** app tier (pods, deploys) | **`netobserv-openshift`** `k8s_list_pods`, `k8s_list_deployments`, `k8s_get_deployment` in **`todo-demo` only** |
| Measure todo API latency during triage | **`netobserv-openshift`** `netobserv_probe_latency` |
| Policy / microsegmentation check | **`netobserv-openshift`** `netobserv_policy_status` |
| Capture flows after triage (60s default) | **`netobserv-openshift`** `netobserv_capture_flows` |
| Analyze capture bundle | **`netobserv-openshift`** `netobserv_analyze_evidence` (preferred) or `analyze-evidence.sh` |
| Live NetObserv RTT / recent flows on todo→postgresql | **`netobserv-live-flows`** skill → **`openshift-mcp`** `netobserv_get_flow_metrics` first (`dataSource=prom`, `timeRange` 60–120); `netobserv_list_flows` only as narrow fallback |

**Never** say “the OpenShift cluster is healthy” based only on demo-path latency probes.

Allowlisted namespaces for `k8s_*` tools: `todo-demo`, `todo-client`, `default`, `openclaw`.
During **investigation**, use **`todo-demo` only** in user-facing turns — do not discuss
`todo-client` workloads (load generators) with the audience.

## Investigation flow (symptom → capture → diagnose)

When the user reports symptoms (slow app, timeouts, DB errors):

1. **netobserv-openshift:** `k8s_list_pods` / `k8s_list_deployments` in `todo-demo` +
   **`openshift_cluster_health`** (platform operators/nodes)
2. **openshift-mcp:** one cluster-wide read (`events_list` or `netobserv_get_flow_metrics`)
3. **`netobserv-openshift`:** `netobserv_probe_latency` **or** `netobserv_policy_status`
4. **`netobserv-openshift`:** `netobserv_capture_flows` (duration_seconds=60, burst_load=true)
5. **`netobserv-openshift`:** `netobserv_analyze_evidence`
6. Summarize diagnosis; ask once for remediation permission — **do not heal in the same turn**

Do **not** use `read` on `summarize-evidence.py`.

See **netobserv-investigate** skill for reply format and spoiler rules.

## Evidence analysis (diagnosis only)

When the user explicitly asks to analyze evidence **or** step 5 above completed:

```bash
bash "$(find /sandbox -name analyze-evidence.sh ! -path '*/.openclaw/*' | head -1)"
```

Reply headings: Impact → Network → Application → Likely cause class →
**Suggested follow-up (diagnostic only)** → Confidence.

**During evidence analysis:** no remediation MCP tools, no “Next Steps”, no chaos/Kraken spoilers.

## Remediation (separate turn only)

| Incident | MCP tool |
|----------|----------|
| Policy / connectivity (Scenario B) | `ansible_launch_job` → `netobserv-restore-policy` (confirmed=true) |
| Latency / path degradation (Scenario A) | `ansible_launch_job` → `netobserv-heal-db-path` (confirmed=true) |

Follow the **netobserv-heal** skill.

## Cluster / platform health

Call MCP `openshift_cluster_health` when the user asks about OpenShift cluster health.

## Live flows / elevated RTT (ad-hoc)

When the user asks for **live** NetObserv data or **elevated RTT** on todo→postgresql:

1. **`openshift-mcp`** `netobserv_get_flow_metrics` — `type=TimeFlowRttNs`, `function=avg` or `p90`,
   `aggregateBy=resource`, `namespace=todo-demo`, `filters=DstK8S_Name=postgresql&DstPort=5432&Proto=6`,
   `dataSource=prom`, `timeRange=60`–`120`
2. Only if needed: **`netobserv_list_flows`** with the same filters, `timeRange=60`, `limit=15`

**Do not** start with `list_flows` for RTT questions — it hits Loki and often times out.
See **netobserv-live-flows** skill.
