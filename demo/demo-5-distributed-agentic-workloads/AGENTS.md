# AGENTS.md — Build instructions for demo-5

Instructions for an AI coding agent implementing **Demo 5: Distributed agentic workloads + AIOps overlay**, aligned with LF Edge **SuperAI SuperBlueprint** and InfiniEdge AI Ops (AIOps).

Read this file fully before creating manifests, scripts, or docs. Prefer small, reviewable increments. Do not invent a second product narrative that conflicts with the decisions below.

---

## 1. What we are building

### Problem

OpenShell (and similar runtimes) host **agentic and non-agentic workloads** across distributed sites. Those workloads are **not** the AIOps system. We need an **AIOps overlay** that fault-finds and operates:

1. The **OpenShell / site platform** (gateway, supervisors, sandboxes, policy, inference routing, cluster health)
2. The **workloads inside** those sandboxes/sites (apps and agents under management)

### Product framing (do not invert)

| Plane | Role |
|-------|------|
| **Managed system** | Distributed OpenShell environments + workloads on lightweight Kubernetes sites |
| **AIOps overlay** | Observe → engage → act (detect, RCA, remediate) **against** that managed system |

AIOps agents run in a **trusted ops plane** (hub cluster / dedicated namespace). They must **not** share privilege domains with tenant sandboxes under management.

### Blueprint alignment

- [SuperAI SuperBlueprint](https://lf-edge.atlassian.net/wiki/spaces/IA/pages/1152974856/SuperAI+SuperBlueprint)
- [Technology Stack](https://lf-edge.atlassian.net/wiki/spaces/IA/pages/1282015236) (Cloud & Edge Platform MVP)
- Related: cloud-native agent blueprint (OpenShell sandbox ≈ supervisor + workload child; MCP; OTEL)

**Site abstraction in the MVP:** lightweight Kubernetes (`MicroShift`, `K3s`, `MicroK8s`, `k0s`), fleet via **Open Cluster Management**, delivery via **Flux**, observability via **OTEL** toward **AIOps** (MTTD/MTTR). “Swarm virtualization” means a **logical** agent execution zone across sites — not “replace sites with bare VMs.”

---

## 2. Lab topology (decided)

**One OpenShift cluster** is enough for the PoC lab:

```text
┌────────────────────── OpenShift (lab host / hub) ──────────────────────┐
│  CNV / KubeVirt VMs  =  stand-ins for edge / remote hardware           │
│  Each VM runs a lightweight Kubernetes: MicroShift, K3s, …             │
│                                                                        │
│  Hub ns (examples):                                                    │
│    • Open Cluster Management hub                                       │
│    • Flux (optional central)                                           │
│    • OpenShell Gateway (central)                                       │
│    • AIOps overlay (collector, ClickHouse, AIOps agents, harness)      │
│                                                                        │
│  Guest VMs (sites):                                                    │
│    • MicroShift / K3s cluster                                          │
│    • OCM klusterlet                                                    │
│    • OpenShell supervisors + sandboxes                                 │
│    • Test workloads (agentic + non-agentic) + OTEL agents              │
└────────────────────────────────────────────────────────────────────────┘
```

### Rules for this topology

1. **CNV VMs = site simulators (machines), not the workload unit.** Document clearly: in production, sites are physical or appliance MicroShift/K3s nodes; VMs are a lab stand-in.
2. **Run OpenShell sandboxes and demo apps inside the guest clusters**, not only as pods on the OCP host (except hub-side Gateway / AIOps).
3. **Register each guest cluster with OCM** so placement, inventory, and site labels (`site_id`) are first-class.
4. Start with **2–3 site VMs** (e.g. one MicroShift, one K3s) sized for nested virt — do not scale VM count before the AIOps loop works.
5. Optional later: Kata / OpenShell MicroVM **inside** a site for stronger sandbox isolation — separate from “how we model sites.”

### What not to do

- Do not treat OpenShift Virtualization VMs as a substitute for lightweight Kubernetes sites.
- Do not run AIOps remediating agents inside the same sandboxes as tenant workloads.
- Do not require multi-cluster bare metal or a second OCP to complete Phase 0–1.
- Do not replace the existing InfiniEdge AIOps evaluation ideas with a heavy agent framework (LangChain/AutoGen) for the first harness — prefer simple tool-calling loops consistent with `research/agentic_aiops_architectures`.

---

## 3. OpenShell concepts (for implementers)

A **sandbox** is the local execution boundary (container/pod/microVM) containing:

- **Supervisor** (`openshell-sandbox`): starts first, outbound session to Gateway, policy proxy, credential injection, launches children
- **Workload child**: agentic or non-agentic app, unprivileged, constrained by Landlock / seccomp / egress / inference routing

AIOps must correlate telemetry with at least: `sandbox_id`, `policy_revision`, `site_id`, and workload identity.

---

## 4. Build phases (implement in order)

### Phase 0 — Hub skeleton

- Namespaces for `openshell`, `aiops`, `openshift-cnv` (or use existing CNV).
- Confirm CNV available; define VM templates (vCPU/RAM/disk) suitable for MicroShift/K3s nested.
- OTEL collector + ClickHouse (or reuse patterns from `research/agentic_aiops_architectures` / demo-4 style scripts).
- Stub AIOps tool surface: read ClickHouse + K8s API (hub); gated write later.

### Phase 1 — Site VMs + lightweight K8s

- Provision 2–3 VMs; install **MicroShift** and **K3s** (diversity matters for the blueprint story).
- Networking: stable IPs / NADs so Gateway, OCM, and OTEL paths work across VMs.
- Join sites to **OCM**; label `site_id=site-a|site-b|…`.
- Smoke: deploy a hello Deployment in each site; scrape metrics to hub ClickHouse with `site_id`.

### Phase 2 — OpenShell on sites

- Deploy OpenShell Gateway on hub (or designated central site).
- Supervisors + sample sandboxes on each site (agentic + non-agentic).
- Emit platform signals: sandbox lifecycle, supervisor session up/down, policy deny counts, inference proxy errors — into the same OTEL pipeline.

### Phase 3 — AIOps overlay

- AIOps agents on hub with tools:
  - **Read:** ClickHouse, OpenShell Gateway API, OCM/site K8s APIs
  - **Write (narrow, gated):** recreate sandbox, restart supervisor workload, restore last-known-good network/inference policy, reschedule via OCM — human-gate high blast-radius policy/identity changes
- Fault injector + scoring harness (MTTD / MTTR / RCA) modeled on `research/agentic_aiops_architectures`.
- Faults at three scopes: **platform**, **runtime/security**, **workload** (see §6).

### Phase 4 — Hardening (only after Phase 3 works)

- Disconnect/reconcile tests (stop VM NIC / shut down site VM).
- Flux per-site delivery; EdgeLake evaluation if in scope.
- Kata / confidential path as optional isolation demo.

---

## 5. Recommended test workloads (inside site K8s)

Purpose: give AIOps **diverse, injectable, observable** targets — both OpenShell-hosted and plain K8s — without boiling the ocean.

Prefer **small, well-known, OTEL-friendly** apps. Tag everything with `site_id`, `workload_class`, and (if sandboxed) `sandbox_id`.

### 5.1 Must-have set (Phase 1–3)

| # | Workload | Class | Why it belongs | AIOps / fault hooks |
|---|----------|-------|----------------|---------------------|
| 1 | **OpenTelemetry Demo** (trimmed: frontend + 3–5 services, e.g. cart / checkout / product-catalog) | Non-agentic microservice app | Same fault vocabulary as existing InfiniEdge AIOps harness; rich OTEL logs/traces/metrics | `scale_zero`, `kill_pod`, `memory_limit`, `network_partition`, `config_corruption`, dependency down |
| 2 | **Simple HTTP service** (e.g. `nginx` or tiny Go/Python “echo” with `/health` + custom metrics) | Non-agentic baseline | Cheap canary per site; proves site + scrape path before OpenShell | Kill pod, break Service/Endpoints, fill disk on pod |
| 3 | **Sandboxed non-agentic batch job** (cron/Job writing results to PVC or object stub) | Non-agentic **inside OpenShell sandbox** | Exercises sandbox lifecycle without LLM flakiness | Kill sandbox, deny egress to storage, corrupt mount, OOM sandbox |
| 4 | **Sandboxed agentic worker** (minimal tool-calling loop: e.g. “poll metrics → call one MCP/tool → write status”) | Agentic **inside OpenShell** | Real agentic tenant workload; **not** the AIOps agent | Break inference backend, revoke tool egress, policy deny storm, stuck loop (CPU/tokens) |
| 5 | **OpenShell supervisor / Gateway canaries** | Platform | Health of the managed runtime itself | Stop supervisor, break Gateway session, bad policy revision |

### 5.2 Strongly recommended additions

| # | Workload | Why |
|---|----------|-----|
| 6 | **MCP tool server** (tiny read-only “skills” server: `get_time`, `read_status_file`) used by workload (4) | Matches SuperBlueprint MCP story; AIOps can detect tool-path vs model-path failures separately |
| 7 | **Synthetic load generator** aimed at (1) or (2) | Turns silent failures into SLO burn (latency/error rate) so detection isn’t guesswork |
| 8 | **Second-site replica** of (2) or trimmed (1) | Cross-site correlation: “only site-b cart is down” vs global outage |

### 5.3 Explicitly defer / avoid for v1

- Full OTEL Demo (all services) on every tiny VM — too heavy for nested MicroShift; use a **trimmed** topology.
- Training / multi-node GPU post-training stacks (belongs to SuperBlueprint PoC lab hardware, not this AIOps demo).
- Heavy multi-agent frameworks as the **tenant** workload — keep one thin agent loop.
- Running the **AIOps** remediator as a sandbox peer of tenant agents.

### 5.4 Workload design requirements (enforce in manifests)

Every demo workload must provide:

1. **OTEL** (or scrapeable metrics + structured logs) to the hub pipeline  
2. **Stable labels:** `site_id`, `app`, `workload_class=agentic|non-agentic|platform`  
3. **Documented fault + expected signal + recovery** entry in ground truth (YAML)  
4. **Resource requests/limits** fit for nested VMs  
5. For sandboxed apps: policy YAML allowing only required egress (OTEL collector, MCP, approved inference)

---

## 6. Fault catalog (minimum for harness)

Implement inject/heal scripts or harness flags covering:

### Platform / site

- Power off or stop site VM; restore  
- Cut VM NIC / NetworkPolicy path to Gateway; restore  
- Delete OCM klusterlet or mark site unavailable  
- Scale OpenShell Gateway to zero / break Gateway Service  

### Runtime / security (OpenShell)

- Kill supervisor pod/process; recreate sandbox  
- Apply deny-all egress policy revision; roll back LKG  
- Point inference router at dead backend; restore  
- Exhaust sandbox ephemeral disk  

### Workload

- Classic K8s faults against OTEL Demo / HTTP canary (align with `research/agentic_aiops_architectures` flags where practical)  
- Stuck agentic loop (busy-spin or infinite tool retry)  
- MCP server down while agent still running  

Score **workload MTTD/MTTR** and **fabric MTTD/MTTR** separately when both apply.

---

## 7. AIOps overlay expectations

- **Observe:** OTEL → ClickHouse (hub); include OpenShell + site + workload signals  
- **Engage:** AIOps agent(s) with tool allow-lists; optional multi-role split later (platform vs workload)  
- **Act:** constrained remediation via Gateway / site K8s / OCM; prefer recreate-sandbox and LKG policy before broad cluster surgery  
- **Metrics:** MTTD, MTTR, remediation success, RCA accuracy — reuse semantics from `research/agentic_aiops_architectures`  
- **Trust:** AIOps credentials and network path isolated from tenant sandbox policies  

Reuse patterns from:

- `research/agentic_aiops_architectures` (harness, faults, ClickHouse tools, MLflow scoring)  
- `demo/demo-4-openshift-network-observability` (scripted install + inject + evidence bundle style)

---

## 8. Repository layout (target)

Keep this demo self-contained under `demo/demo-5-distributed-agentic-workloads/`:

```text
AGENTS.md                 # this file — agent build contract
README.md                 # human quickstart (write after Phase 0 works)
docs/
  ARCHITECTURE.md         # topology + trust boundaries
  FAULTS.md               # fault catalog + ground truth pointers
manifests/
  hub/                    # AIOps, Gateway, OCM hub bits
  site-workloads/         # OTEL demo trim, canaries, agent sandbox apps
  openshell/              # policies + sandbox examples
scripts/
  provision-site-vms.sh
  bootstrap-site-k8s.sh
  install-aiops-hub.sh
  inject-fault.sh
config/
  sites.yaml              # site_id, distro (microshift|k3s), sizing
  fault_ground_truth.yaml
```

Do not dump large upstream clones into this tree; pin versions and document fetch/submodule steps in README.

---

## 9. Success criteria (PoC “done”)

1. One OCP cluster hosts ≥2 CNV VMs running **different** lightweight K8s distros.  
2. Each site appears in OCM with `site_id`; workloads emit OTEL to hub with that label.  
3. OpenShell Gateway + ≥1 agentic and ≥1 non-agentic sandboxed workload are healthy.  
4. AIOps on hub detects and remediates ≥3 injected faults spanning platform and workload scopes, with recorded MTTD/MTTR.  
5. README states clearly that VMs simulate edge hardware; blueprint sites remain MicroShift/K3s-class clusters.

---

## 10. Decision log (do not regress)

| Decision | Choice |
|----------|--------|
| OpenShell role | Managed runtime for tenant agentic/non-agentic workloads |
| AIOps role | Overlay that operates OpenShell environments + workloads |
| Lab distribution | CNV VMs on one OCP, each running MicroShift/K3s/… |
| Fleet | Open Cluster Management |
| Primary workload for AIOps parity | Trimmed OpenTelemetry Demo + sandboxed agent + non-agentic canary |
| AIOps placement | Hub / trusted ops plane, not inside tenant sandboxes |
| Frameworks for v1 AIOps agents | Simple tool-calling loops; no LangChain/AutoGen requirement |

When uncertain, prefer alignment with SuperBlueprint Technology Stack MVP and existing InfiniEdge AIOps harness semantics over novel abstractions.
