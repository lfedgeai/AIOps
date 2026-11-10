---
name: aiops-detect
description: >-
  Detect whether a fault is present from live telemetry and Kubernetes signals.
  Use for all AIOps detector agents before RCA or remediation.
---

# AIOps — Detect

## Goal

Decide `detected=true|false` from **live** signals in the fault window. Do not invent a fault from docs alone.

## Required evidence (use available tools)

1. Cluster: `get_pod_status`, `get_events` (and `get_pod_logs` if needed).
2. Telemetry (if profile allows): `compare_telemetry`, `search_logs` / traces / metrics since `{since_ts}` / T0.
3. Treat “no pods / 0 replicas / CrashLoop / ImagePull / config mount errors” as strong detect signals.

## Rules

- `detected=false` only when available signals look normal for the fault window.
- Prefer cluster state over RAG for detection on K8s faults.
- Call `log_action` once with what you inspected and the detect decision.
- If detected, continue to diagnose — do not finish yet when remediation tools exist.

## Anti-patterns

- Declaring detect from a README/CHANGELOG hit with no cluster/telemetry check.
- Finishing after a single RAG call.
