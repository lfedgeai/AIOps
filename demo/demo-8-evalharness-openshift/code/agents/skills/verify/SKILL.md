---
name: aiops-verify
description: >-
  Verify remediation worked via cluster and telemetry re-checks before final JSON.
  Use after every AIOps write-tool call.
---

# AIOps — Verify

## Goal

Confirm the workload recovered. Do not claim success from the write-tool ACK alone.

## After each write tool

1. Re-run `get_pod_status` (and `get_events` if needed) on the affected workload.
2. If telemetry tools exist, spot-check errors/latency since the remediation time.
3. Success signals: desired replicas ready, pods Running, no fresh CrashLoop for the fault class.
4. If still broken: one retry with an alternate write action justified by live state, then report honestly.

## Final JSON

Only then emit:

```json
{"detected": true, "confidence": 0.0-1.0, "suggested_root_cause": "workload", "suggested_remediations": ["steps taken or recommended"]}
```

List **steps actually taken** (tool calls) in `suggested_remediations`, not hypothetical advice.

## Rules

- `log_action` verification outcome (recovered / not recovered).
- Never set an implied success in prose when pods are still down.

## Anti-patterns

- Finalizing immediately after `restart_deployment` without a status re-check.
- `detected=false` after you already remediated a real fault.
