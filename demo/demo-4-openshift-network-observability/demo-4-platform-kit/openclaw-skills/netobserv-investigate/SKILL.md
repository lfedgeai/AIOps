---
name: netobserv-investigate
description: "Triage todo-demo performance or connectivity issues, capture NetObserv flow traces, then hand off to evidence analysis. Use when users report slowness, timeouts, or DB problems — before analyzing evidence."
user-invocable: true
metadata:
  { "openclaw": { "always": true } }
---

# NetObserv investigation (triage → capture → analyze)

## When this applies

The user reports **symptoms** (slow todo app, timeouts, database errors, “the app is broken”)
and has **not** asked to analyze an existing evidence bundle yet.

This skill owns the **investigation narrative**: rule out obvious causes → measure latency →
capture flows → analyze evidence → optional pod sanity check → **stop before heal**.

Do **not** jump straight to analysis until flow capture completes (Step 4) and
`netobserv_analyze_evidence` returns.

## Step 1 — Workload & platform sanity (quick)

Narrate briefly what you are checking, then call MCP tools from **both** servers during triage
(proves full-platform SRE breadth, not only the todo→DB network path). Keep pre-capture triage
lean — **≤5 MCP calls** before Step 2 (probe/capture).

### Application scope (`netobserv-openshift`)

1. `k8s_list_pods` in **`todo-demo` only** (todo + postgresql Running? high restart counts?)
2. `k8s_list_deployments` in **`todo-demo` only** (replicas ready?)

### Cluster-wide scope (both MCP servers)

3. **`netobserv-openshift`** → `openshift_cluster_health` — ClusterVersion, ClusterOperators,
   node Ready (compact platform summary).
4. **`openshift-mcp`** (Red Hat) → **one** read-only tool that adds cluster-wide context beyond
   step 3, for example:
   - `events_list` (core) — recent Warning events cluster-wide or in `openshift-*` / `netobserv`
   - **or** `netobserv_get_flow_metrics` (netobserv) — confirm the observability pipeline is
     returning flow data (not an app-path latency probe)

Mention in user-facing text that you checked **application workloads** and **OpenShift platform
health** (operators/nodes/events or observability pipeline) — not only the database network path.

**Do NOT call during triage:** `netobserv_list_chaos`, `netobserv_status`, `k8s_recent_events`,
`ansible_launch_job`, or other remediation tools. Do **not** list pods/deployments in `todo-client`
or other consumer namespaces — that exposes lab traffic generators and breaks the demo narrative.

**Demo narrative (user-facing text):** Respond like an SRE to a real user-reported incident.
Never mention loadgen, loadgen-heavy, synthetic traffic, traffic generators, `todo-client`
workloads, demo/lab setup, or “we turned up load to simulate errors.” If the idle latency probe
looks fast while users report slowness, say user-visible impact may differ under real concurrent
usage and proceed to NetObserv capture — do **not** name internal load tooling.

Summarize in **one short paragraph**: app workloads healthy vs not; **platform** (operators/nodes
and/or cluster events / NetObserv pipeline) OK vs not.

## Step 2 — Symptom path check (latency vs connectivity)

Call **one** of:

| Symptom | Tool |
|---------|------|
| Slow / timeouts / high latency | `netobserv_probe_latency` |
| Cannot connect / DB denied / policy incident suspected | `netobserv_policy_status` |

Quote key numbers (probe `time=` lines or `POLICY_VERDICT`).

**Branch:**

- **Policy/connectivity denial** (POLICY_VERDICT does not admit todo): skip latency narrative;
  go to Step 4 capture (policy signals still appear in the bundle), then Step 5 analysis.
- **Elevated latency** (probe max sample clearly high, e.g. >0.5s) or ambiguous slowness with
  healthy pods: proceed to Step 4.
- **Healthy idle probe + healthy pods** but user still reports slowness: note the symptom may
  show under real user traffic even when a spot-check looks fine; proceed to Step 4 capture on
  the DB path (do not mention load generators or lab namespaces).

## Step 3 — Tell the user you are capturing flows

Before calling capture, one sentence for the audience, e.g.:

> Workloads and cluster health look normal, but latency to the todo API is elevated. I will
> collect a short NetObserv flow capture on the database path to see RTT and drops.

## Step 4 — Capture flow traces (MCP)

1. Call MCP `netobserv_capture_flows` with `duration_seconds=60` (default; do not exceed 90).
2. When the tool returns `EVIDENCE_READY`, immediately call MCP `netobserv_analyze_evidence`.

Do **not** use the `read` tool on `summarize-evidence.py` — it is not at a flat sandbox path.

If capture fails, reply with the tool error and suggest the operator check
`oc -n openclaw get deploy netobserv-capture-proxy` — do not invent kubeconfig steps.

## Step 5 — Present the diagnosis

Use the output from `netobserv_analyze_evidence` and present the **executive diagnosis**:

Impact → Network (NetObserv) → Application → Likely cause class →
**Suggested follow-up (diagnostic only)** → Confidence.

## Step 6 — Post-diagnosis sanity (optional, one call)

If not done in Step 1, optionally re-check `k8s_list_pods` in `todo-demo` to confirm app
tiers are still Running after analysis.

## Step 7 — Close the investigation turn (no heal yet)

End with a **plain-language summary** and ask whether the user wants you to **attempt
remediation** on the cluster — one sentence only, e.g.:

> The evidence points to network path degradation on todo→postgresql:5432. Application pods
> look healthy. Would you like me to attempt remediation on the cluster?

- Do **not** call `ansible_launch_job` or other automation in this turn.
- Do **not** use the heading **“Next Steps”**.
- No Kraken, chaos Jobs, loadgen, bastion scripts, or fault-injection names unless the user said them.

## Remediation (separate turn)

When the user confirms heal/restore in a **new** message (`/new` recommended after long investigate turns):

- Latency / path degradation → **netobserv-heal** skill → `ansible_launch_job` / `netobserv-heal-db-path`
- Policy / connectivity denial → **netobserv-heal** skill → `ansible_launch_job` / `netobserv-restore-policy`
