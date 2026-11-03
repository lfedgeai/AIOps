## Why

The Tier 2 demo did not convince the LF Edge AIOps group. Its faults are flagd feature flags, which the group's own harness (`lfedgeai/AIOps`, `research/agentic_aiops_architectures`) moved away from: their default faults are Kubernetes API changes against the OpenTelemetry Demo, and they score agents on detection, RCA, remediation and MTTD across matrices of about 300 runs. Our results could not be compared with theirs, had no time-to-detect, and used a fault source they consider legacy. Tier 3 runs Jev against the same fault catalogue, on Kubernetes, and reports in the same terms, so Jev's results sit next to the group's model table.

## What Changes

- Add **Tier 3**: the OpenTelemetry Demo from its official Helm chart (release matching app version 3.1.0) on a local **kind** cluster with one control-plane node and three workers. One worker hosts observability and load; two host the application. Configured by values only.
- Add a fault injector for the group's ten Kubernetes fault types, using their names: `scale_zero`, `kill_pod`, `memory_limit`, `network_partition`, `readiness_probe_fail`, `config_corruption`, `dependency_removal`, `replica_overload`, `node_taint`, `pvc_full`. Our injector saves the exact prior state and restores it, adds no self-describing labels, and targets an application worker for `node_taint`.
- Add a scenario catalogue `K000` (healthy control) to `K010`, targeting `cart` as in the group's latest matrix, with hand-written ground truth and rationale. Each is verified on the cluster before it is kept.
- Add Kubernetes state to the observation (deployments and replica counts, unhealthy pods, nodes and taints, Warning events) next to the service telemetry, and extend the scrubber and leak test for Kubernetes leak paths.
- Add a **polling mode**: after injection the harness asks Jev at a fixed interval until it reports an incident or the detection window ends. This gives a real MTTD. The same polling during healthy-control runs gives false positives and detection precision, which the group's harness does not measure.
- Report a **scorecard in the group's terms** (detect, RCA, remediation correct, MTTD, agent errors, tokens and cost) and export per-run JSON with their field names. Remediation executed, MTTR and recovery verified are reported as "not measured": the harness still never acts on Jev's recommendation.
- Keep the Tier 2 lifecycle: four independent verdicts, no composite, reset in `finally`, contamination stop.
- **BREAKING (policy only):** `CLAUDE.md` says "Do not add Kubernetes." This change allows Kubernetes (a local kind cluster) for the Tier 3 testbed only.
- Real node faults on virtual machines (CPU, memory, disk, network, node down) are **not** in this change. They are a planned follow-up that reuses the Tier 3 harness.

## Capabilities

### New Capabilities
- `k8s-testbed`: kind cluster config, pinned versions, Helm install of the demo by values, NetworkPolicy support, memory pre-check, health check, teardown.
- `k8s-fault-injection`: the ten fault types, exact save and restore, read-back, dead-man revert, `reset-faults`, manifestation and recovery checks.
- `tier3-scenarios`: scenario file format, Tier 3 vocabularies, and the `K000` to `K010` catalogue.
- `tier3-observation`: Kubernetes state collectors, observation shape and byte budget, extended scrubber and leak test.
- `tier3-engine-adapter`: Tier 3 question set, strict parsing, reasoning text recorded.
- `tier3-runner`: lifecycle, polling detection, verdicts, timing, results, scorecard and export, batch runs.
- `tier3-reproducibility`: scripts, README, `CLAUDE.md` update, `docs/k8s-testbed-notes.md`, offline CI.

### Modified Capabilities
<!-- None. openspec/specs/ is empty; earlier changes are not archived. Tier 1 and Tier 2 behaviour does not change. -->

## Impact

- **Code:** new `tier3/` package and `tests/tier3/`. Tier 3 reuses Tier 2 helpers (`Scrubber`, timing, deviation check, Prometheus, Jaeger and OpenSearch collectors) by import, without changing Tier 2 behaviour.
- **Host:** Docker (present), plus `kind`, `kubectl` and `helm`. kind nodes share host memory; the demo needs about 7 GB plus cluster overhead. The pre-check requires 12 GB available and refuses while Tier 2 runs.
- **External systems:** kind and its node image, the OpenTelemetry Demo Helm chart, and a NetworkPolicy-capable CNI if kind's default does not enforce policies, all pinned. The Jev API as before.
- **Jev usage:** polling makes several Jev calls per run instead of one. The count per run is recorded.
- **Docs and policy:** `README.md`, `CLAUDE.md`, `.env.example`, `.gitignore`, new `docs/k8s-testbed-notes.md`.
- **CI:** stays offline. No cluster, no Docker, no Jev in tests.
