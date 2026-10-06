---
name: netobserv-live-flows
description: "Query live NetObserv flow metrics or recent flow records on the todo→postgresql path (RTT, throughput, drops). Use when the user asks for live/recent flows, elevated RTT, or console-style observability — not for first-time symptom triage."
user-invocable: true
---

# NetObserv live flows (todo→postgresql)

## When this applies

The user asks for **live** or **recent** NetObserv data, for example:

- "Show recent flows on todo→postgresql with elevated RTT"
- "What's the current RTT on the database path?"
- "Live NetObserv metrics for todo-demo → postgresql:5432"

Use **openshift-mcp** read-only tools. Do **not** use `netobserv_capture_flows` unless the user
explicitly wants a timed capture bundle for deeper diagnosis (see **netobserv-investigate**).

For **first symptom reports** ("app is slow", "users complain"), use **netobserv-investigate**
instead — capture → analyze gives richer evidence than ad-hoc live queries.

## Step 1 — Prefer aggregated RTT metrics (fast, Prometheus)

For **elevated RTT**, **current latency**, or **path health** questions, call
**`openshift-mcp`** → **`netobserv_get_flow_metrics`** first — **not** `netobserv_list_flows`.

`list_flows` queries Loki and often times out on this lab; metrics use Prometheus when possible.

Example parameters for todo→postgresql:

| Parameter | Value |
|-----------|--------|
| `namespace` | `todo-demo` |
| `filters` | `DstK8S_Name=postgresql&DstPort=5432&Proto=6` |
| `aggregateBy` | `resource` (or `DstK8S_Name`) |
| `type` | `TimeFlowRttNs` |
| `function` | `avg` or `p90` |
| `dataSource` | `prom` |
| `timeRange` | `60`–`120` (seconds — keep short) |
| `limit` | `20` |

Summarize RTT in **milliseconds** for the user (divide nanoseconds by 1e6). Call out whether
p90/avg looks elevated (e.g. >500 ms) on the todo→postgresql path.

## Step 2 — Sample individual flows only if metrics succeeded and user wants records

If metrics show elevated RTT **and** the user asked to **see flows** (not just aggregates), call
**`netobserv_list_flows`** as a **second** step with a **narrow** window:

| Parameter | Value |
|-----------|--------|
| `namespace` | `todo-demo` |
| `filters` | `DstK8S_Name=postgresql&DstPort=5432&Proto=6&TimeFlowRttNs>500000000` |
| `timeRange` | `60` (not 300) |
| `limit` | `15` |

Do **not** put `SrcK8S_Namespace` in `filters` when `namespace=todo-demo` is already set — use
indexed namespace scope via the parameter.

If `list_flows` returns timeout (`-32001`), reply with the **metrics result from Step 1** and
note that individual flow records are unavailable right now; suggest a timed capture
(`netobserv_capture_flows`) if they need packet-level evidence.

## Step 3 — Reply format

Keep it short:

1. **Path** — todo-demo → postgresql:5432
2. **Live RTT** — quote avg/p90 from metrics (ms)
3. **Sample flows** — only if Step 2 succeeded (top few src pods / RTT values)
4. **Interpretation** — elevated vs normal; do not mention loadgen, Kraken, or lab tooling

Do **not** call heal or policy tools in this turn unless the user explicitly asks to remediate.
