## 1. Verify the platform and every fault before coding against them

- [x] 1.1 Free host memory and stop Tier 2; record available RAM and Docker version (spec: k8s-testbed pre-check)
- [x] 1.2 Install pinned `kind`, `kubectl` and `helm`; record versions
- [x] 1.3 By hand, create the four-node `jev` cluster with role labels and NodePort mappings; confirm whether the default CNI enforces NetworkPolicy, else recreate with pinned Calico (design D4)
- [x] 1.4 Find the demo chart release whose `appVersion` is 3.1.0; install it with node selectors and NodePorts; confirm placement and measured memory; confirm Prometheus, Jaeger, OpenSearch and Locust are reachable from the host
- [x] 1.5 Read the LF Edge harness (`lfedgeai/AIOps`, `research/agentic_aiops_architectures`) for its per-run JSON field names, its poll cadence and detection window, and record them for the export and D11 defaults
- [x] 1.6 Run each of the ten faults by hand on `cart` (and `valkey-cart`, `jev-worker2`); for each, record what shows in span metrics, Jaeger, OpenSearch, deployments, pods, nodes and events, how long it takes, and that exact restore works
- [x] 1.7 Settle the rolling-update question for `memory_limit`, `readiness_probe_fail` and `config_corruption`: does the old pod keep serving? Decide each scenario's `incident` ground truth with a rationale
- [x] 1.8 Check whether `cart` reads `DATABASE_HOST`; if not, drop or retarget `config_corruption` and note it for the group
- [x] 1.9 Check whether `kill_pod` and `pvc_full` leave a symptom long enough to observe; keep, retune or drop
- [x] 1.10 Capture real Warning events from these runs as leak-test fixtures and list every string that names a fault or harness object
- [x] 1.11 Write `docs/k8s-testbed-notes.md`: versions, ports, queries, fault behaviour, dropped scenarios and reasons, and the differences from the group's fault code (D5); delete or stop the cluster (notes written 2026-10-14; cluster stopped with `stop.sh` 2026-10-18 after group 10)

## 2. Testbed scripts

- [x] 2.1 `tier3/testbed/precheck.sh`: Docker running, 12 GB available, Tier 2 not running
- [x] 2.2 `tier3/testbed/kind.yaml` and `create.sh`: pinned node image, four nodes, labels, `extraPortMappings`, CNI choice from 1.3; refuse if the cluster exists
- [x] 2.3 `tier3/testbed/install.sh` and `values/`: demo chart at the pinned version by values only; write `VERSION` from the running cluster (value keys unverified until 1.4)
- [x] 2.4 `tier3/testbed/health-check.sh` with every check in the spec, naming the first failure
- [x] 2.5 `stop.sh`, `start.sh`, `delete.sh` (confirmation unless `--yes`)
- [x] 2.6 Root wrappers `setup-k8s.sh` and `health-check-k8s.sh`; `.gitignore` for the kubeconfig; new variables in `.env.example`

## 3. Kubernetes access and collectors

- [x] 3.1 `tier3/collectors/kube.py`: thin HTTPS client from the kubeconfig with injectable `get`, `patch`, `put`, `post`, `delete`; deployments, pods, nodes, events, NetworkPolicies, pod exec for `pvc_full`
- [x] 3.2 Reuse Tier 2's Prometheus, Jaeger and OpenSearch collectors against the Tier 3 NodePorts; `tier3/collect.py` assembles one `Telemetry` value
- [x] 3.3 Offline tests with fixtures from 1.10 and fake HTTP (hand-made fixtures in `tests/tier3/helpers.py`, extended with the two Warning events captured in 1.10)

## 4. Scenarios and vocabularies

- [x] 4.1 `tier3/vocab.py` (design D8)
- [x] 4.2 `tier3/scenarios.py`: frozen dataclasses, YAML loader, enum and target validation, unknown-field rejection
- [x] 4.3 Write the surviving `K000` to `K010` files with hand-written expected outcomes and rationales from task 1 (written as provisional; revisit after 1.6-1.9)
- [x] 4.4 Tests: unknown fault type, taint on `jev-worker`, missing rationale, out-of-vocabulary value, `K000` shape, fault names match the group's spelling

## 5. Fault injection

- [x] 5.1 `tier3/faults.py`: the ten fault types through `kube.py`, no self-describing markers, `jev-fault` naming
- [x] 5.2 Save state to the ledger before inject; read back after inject; exact restore and read back on reset
- [x] 5.3 Host dead-man timer (`systemd-run --user --on-active`) and `python -m tier3 reset-faults` (ledger restore, policies, taints; idempotent; reports changes)
- [x] 5.4 Manifestation polling and the cluster recovery check
- [x] 5.5 Tests with a fake API: round trip per fault type, two-replica restore, other taints kept, read-back mismatch, reset failure gives contaminated, idempotent reset-faults, no marker added by `config_corruption`

## 6. Observation and scrubber

- [x] 6.1 `tier3/observation.py`: shape from design D9, named caps, ranking, byte budget, trim notes; no scenario types imported or accepted; no specs, env or limits
- [x] 6.2 Extend the scrubber (reusing Tier 2's `Scrubber`) for excluded namespaces, `jev-fault` and taint keys
- [x] 6.3 Tests: import and signature guards, `scale_zero` visible as replica counts only, empty `pods` when healthy, budget trimming, taint key removed from a scheduling event

## 7. Engine adapter

- [x] 7.1 `tier3/engine.py`: `Decision` (with `affected_component` and `reasoning`) and `AIOpsEngine`
- [x] 7.2 `tier3/jev_adapter.py`: four questions, strict parsing, reasoning stored, usage via `apply_usage`; the only Tier 3 importer of `jev.client`
- [x] 7.3 Tests with canned Jev responses: valid, out-of-set component, bad probability, HTTP error, with reasoning; questions identical across scenarios

## 8. Runner, polling, scoring and export

- [x] 8.1 `tier3/evaluate.py`: Tier 2 verdict rules on Tier 3 fields; false-positive runs and polls for `K000`
- [x] 8.2 `tier3/runner.py`: lifecycle with reset in `finally`, `poll` and `single` modes, `t_detect` and MTTD, every poll stored, contamination stop, results under `results/tier3/<run-id>/`
- [x] 8.3 `scorecard` (design D12), with denominators and "not measured (recommendation only)" rows
- [x] 8.4 `export-lfedge` with the field names from 1.5, and `export-reasoning`
- [x] 8.5 `tier3/cli.py`, `python -m tier3`, root wrappers `run-k8s-experiment` and `run-all-k8s-experiments`
- [x] 8.6 Whole-path leak test over every scenario and every poll with contaminated fixtures
- [x] 8.7 Runner tests with a virtual clock: detect on the third poll, never detected, `K000` false-positive counting, engine error mid-poll still resets, stuck rollout gives contaminated, missed incident still scores other fields

## 9. Docs, policy and CI

- [x] 9.1 Update `CLAUDE.md`: Kubernetes allowed for the Tier 3 kind testbed only; recommendation-only rule unchanged; Tier 3 rules, layout and commands
- [x] 9.2 Update `README.md`: Tier 3, relation to the LF Edge harness and what is not comparable, host requirements, commands, scenario table, environment variables
- [x] 9.3 Add Tier 3 tests to `./run-all.sh --check`; confirm CI passes offline on Python 3.10 to 3.13 (included automatically via `testpaths = tests`; CI run 39106546717 on 2026-10-18: tests pass on 3.10, 3.11, 3.12, 3.13)
- [x] 9.4 Confirm Tier 1 and Tier 2 tests pass unchanged (375 passed, Python 3.13, 2026-10-14)

## 10. Live verification

- [x] 10.1 Free memory, run `./setup-k8s.sh`, confirm health (2026-10-18: restarted the existing cluster with the fixed `start.sh`; all health checks ok)
- [x] 10.2 Run each scenario once with Jev in `poll` mode; read each request by hand for leaks (K000-K005 2026-10-14 run 20261014-195159-3f10; K006-K009 20261018-055547-2407; K010 20261018-061800-a5b2; no leaks: effects only, taint shown as effect not key)
- [x] 10.3 Batch with `--repeats 3`, plus at least 10 `K000` runs for the false-positive rate (2026-10-18: 30 fault runs in 20261018-070851-60ee, 10 K000 in 20261018-100756-41e8; all ok, 0 contaminated, 0 invalid of 351 polls)
- [x] 10.4 Print the scorecard, export for the group, export reasoning and hand-score it if Jev returned any (scorecard, export-lfedge and export-reasoning over both run directories; Jev returned no reasoning text, so nothing to hand-score)
- [x] 10.5 Write `docs/results-<date>-tier3-jev.md` from the run records only, stating what is not comparable with the group's agents; delete or stop the cluster (`docs/results-2026-10-18-tier3-jev.md`; cluster stopped, not deleted)
