---
name: netobserv-evidence
description: "Analyze NetObserv evidence for todo→PostgreSQL issues (latency, drops, JDBC, or policy denial). Use after flow capture or when the user explicitly asks to analyze evidence / /netobserv-evidence."
user-invocable: true
---

# NetObserv evidence analysis

## When this applies

The user asks to **analyze evidence** explicitly, or you have just completed flow capture
(`netobserv_analyze_evidence` succeeded) as part of an investigation.

For **initial symptom reports** (“the app is slow”, “users complain”, “DB timeouts”), use
**netobserv-investigate** first — do not run this skill until capture has populated
`evidence/latest.json`.

## Step 1 — Always run analysis (MCP preferred)

**Do this immediately.** Prefer MCP `netobserv_analyze_evidence` (works in-cluster after capture).

Fallback only if MCP is unavailable:

```bash
bash "$(find /sandbox -name analyze-evidence.sh ! -path '*/.openclaw/*' | head -1)"
```

- Do **not** use the `read` tool on `summarize-evidence.py`.

If the script exits with `ERROR: evidence file not found`, reply in **one short paragraph**:

> No flow evidence is in the workspace yet. Run the investigation path first: triage workloads,
> call `netobserv_capture_flows`, then `netobserv_analyze_evidence`. If capture fails, check
> `netobserv-capture-proxy` in the openclaw namespace on the cluster.

Do **not** ask “Would you like me to check?” — state the fix and stop.

## Step 2 — Optional live context (workloads only)

If helpful, inspect workloads with MCP `netobserv-openshift`: `k8s_list_pods` and/or
`k8s_list_deployments` in `todo-demo` only — **one** quick call is enough.

**During evidence analysis, do NOT call:**

- `k8s_recent_events` (slow; policy/path signals are in the evidence bundle)
- `ansible_launch_job` (remediation — user must ask separately; ansible-automation MCP)
- `netobserv_list_chaos` (surfaces demo/lab job names)
- `netobserv_status` (includes batch-job inventory that reads like a scripted demo)
- `openshift_cluster_health` (not needed for todo→DB evidence turns)

For **live** NetObserv flows/metrics (follow-up after capture, or complex flow checks), use read-only
**openshift-mcp** tools when wired. Prefer capture → `netobserv_analyze_evidence` for the primary
diagnosis — do not mention those tools in your reply unless you
actually called them.

## Step 3 — Reply using this executive format

Use the script’s **Executive summary** as your backbone. Present these headings in **plain
language** (numbers from the script — do not invent):

### Impact
What users/operators see (timeouts, slow API, pool exhaustion). One or two sentences.

### Network (NetObserv)
Path `todo → postgresql:5432`. Quote RTT (avg/p95), drops, top talkers if present.

### Application
JDBC/Agroal errors from evidence; datasource still points at PostgreSQL (no proxy host).

### Likely cause class
Path degradation on an **open** route vs **connectivity denied** by NetworkPolicy — follow the script’s classification. Do not default to “latency” when the bundle indicates policy/microsegmentation denial.

### Suggested follow-up (diagnostic only)
Copy the bullet list from the script’s **Suggested follow-up** section. Use operator language
(verify workloads, re-measure latency, escalate with bundle, compare baseline).

- Do **not** use the heading **“Next Steps”** for remediation.
- Do **not** name MCP tools, slash commands, heal scripts, or batch-job cleanup here.
- Do **not** ask “Would you like to heal?” or “proceed with healing?” — stop after diagnosis.

### Confidence
What is **measured** (RTT, drops, log lines) vs **inferred** (cause class). Copy only the Measured/Inferred bullets from the script — **not** the “Agent notes” section.

**Spoiler rules:** No Kraken, Toxiproxy, chaos/lab job names, or fault-injection tooling unless
the **user** named them first. Do not connect elevated RTT to “a chaos job” or “network shaping
we can clear” — describe **observed path degradation** only.

If the script flags **policy_connectivity_denial_suspected** or **Connectivity denied**, describe
enforcement/microsegmentation — **not** path latency. Do **not** suggest `ansible_launch_job` or tc cleanup.

If the user later asks to heal for a **latency/Kraken** scenario (“heal”, `/netobserv-heal`), that is a
**different** turn — use the heal skill/MCP then, not this evidence reply.
