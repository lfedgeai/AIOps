# AGENTS.md — Build instructions for demo-5

Instructions for an AI coding agent implementing **Demo 5: Distributed agentic workloads + AIOps overlay**, aligned with LF Edge **SuperAI SuperBlueprint** and InfiniEdge AI Ops (AIOps).

Read this file fully before creating manifests, scripts, or docs. Prefer small, reviewable increments. Do not invent a second product narrative that conflicts with the decisions below.

**First build slice (kick off here):** Phase 0 → Phase 2 only — Multicloud GitOps hub (`ValidatedPattern/`) + ≥1 site VM (**MicroShift** first; K3s later for full PoC) with `site_id` + todo/Postgres/loadgen with verified OTEL (**traces, metrics, and logs**) → ClickHouse via hub **OpenShift Route**. Todo image from the **OpenShift internal registry** (Route). Site registration: try **RHACM import**, else **ArgoCD kubeconfig destination** (do not block the slice on ACM). Do **not** start OpenShell or AIOps remediation until that slice works. The **LLM / model** is a Phase 4+ AIOps component — wire it when building engage/act, not here. Phases 3–5 and §9.2 are the complete PoC, not the first PR.

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

**AIOps placement (decided for v1): central on the hub.** Remediating AIOps agents run only on the OpenShift hub (trusted ops plane). Do **not** deploy remediating AIOps agents onto site VMs for v1.

**AIOps observe / act split (decided):**

| Plane | Handover / mechanism |
|-------|----------------------|
| **Observe** | Hub **ClickHouse** is the AIOps observe handover. OTEL (site collectors → hub Route → hub collector) streams traces, metrics, and logs into ClickHouse; AIOps **reads** that backend for detect / RCA evidence. |
| **Act** | Remediation is **not** written back through ClickHouse. AIOps engages and remediates via its own tool surface — **MCP servers** and other gated tools (site kube APIs via RHACM and/or Argo-registered kubeconfigs; OpenShell Gateway **when Phase 3 exists**; Vault-backed credentials when needed). |

**LLM / model endpoint:** a later **AIOps solution component** (Phase 4+). Not in scope for Phase 0–2; when building the overlay, stub the agent harness until the model is wired.

Optional later (Phase 5+): thin **site-local observers** (detect-only or suggest-only) for disconnected operation — still not co-resident with tenant OpenShell sandboxes, and remediation stays hub-gated or explicitly policy-scoped.

### Blueprint alignment

- [SuperAI SuperBlueprint](https://lf-edge.atlassian.net/wiki/spaces/IA/pages/1152974856/SuperAI+SuperBlueprint)
- [Technology Stack](https://lf-edge.atlassian.net/wiki/spaces/IA/pages/1282015236) (Cloud & Edge Platform MVP)
- Related: cloud-native agent blueprint (OpenShell sandbox ≈ supervisor + workload child; MCP; OTEL)

**Site abstraction:** lightweight Kubernetes on CNV VMs (lab stand-in for edge hardware). **v1 site distros:** **MicroShift** and **K3s** (one of each once two sites exist). Observability via **OTEL** toward **AIOps** (MTTD/MTTR). “Swarm virtualization” means a **logical** agent execution zone across sites — not “replace sites with bare VMs.”

**Hub control plane (decided):** Multicloud GitOps in this repo’s **`ValidatedPattern/`** directory: **OpenShift GitOps (ArgoCD)** + **RHACM** + **HashiCorp Vault**. See §2.1 and [getting started](https://validatedpatterns.io/learn/getting-started-multi-cloud-gitops/).

---

## 2. Lab topology (decided)

**One OpenShift cluster** is enough for the PoC lab:

```text
┌────────────────────── OpenShift (lab host / hub) ──────────────────────┐
│  ValidatedPattern/ (Multicloud GitOps, hub clusterGroup):              │
│    • Validated Patterns Operator + pattern.sh / Pattern CR             │
│    • OpenShift GitOps (ArgoCD)                                         │
│    • RHACM hub                                                         │
│    • HashiCorp Vault (+ load-secrets)                                  │
│  Plus demo-5 hub apps (Argo Applications / values overlays):           │
│    • Hub OTEL collector — OTLP exposed via OpenShift Route             │
│    • ClickHouse                                                        │
│    • OpenShell Gateway (Phase 3+)                                      │
│    • Central AIOps overlay (Phase 4+; model = later AIOps component)   │
│                                                                        │
│  CNV / KubeVirt VMs  =  stand-ins for edge / remote hardware           │
│  Each VM runs MicroShift or K3s                                        │
│  Register site: RHACM import (preferred) or Argo cluster kubeconfig    │
│                                                                        │
│  Guest VMs (sites):                                                    │
│    • MicroShift / K3s                                                  │
│    • RHACM agent after successful import (optional if using Argo-only) │
│    • Python todo (FastAPI + OTEL T/M/L) + Postgres                     │
│    • postgres_exporter + loadgen                                       │
│    • Site OTEL Collector → hub Route (with site_id)                    │
│    • OpenShell supervisors + sandboxes (Phase 3+)                      │
│    • Optional thin tenant agentic sandbox (not AIOps)                  │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Multicloud GitOps (`ValidatedPattern/`)

**Decision: use Multicloud GitOps in `ValidatedPattern/` as the hub skeleton; extend it for demo-5 sites and workloads.**

| Pattern provides | Demo-5 use |
|------------------|------------|
| Hub OpenShift + Validated Patterns Operator / `pattern.sh` | Install/manage hub GitOps framework |
| OpenShift GitOps (**ArgoCD**) | Declarative delivery for hub and site apps |
| **RHACM** | Preferred fleet registration, placement, governance |
| **Vault** + home-dir `values-secret-*.yaml` | Pattern secrets (never commit the home-dir file). Phase 2 todo DB: lab Secret → pod **env**; Vault/ESO later (same env keys) |
| Sample `hello-world` / `config-demo` | **Not** AIOps targets; leave or remove |

**References:**

- Getting started: https://validatedpatterns.io/learn/getting-started-multi-cloud-gitops/
- Pattern overview: https://validatedpatterns.io/patterns/multicloud-gitops/
- Upstream reference: https://github.com/validatedpatterns/multicloud-gitops
- **This repo:** `ValidatedPattern/` at repository root (sibling of `demo/`)

**How to extend:**

1. Populate/maintain Multicloud GitOps under **`ValidatedPattern/`**. Demo-5 references that tree. Workload/OTEL manifests live under `demo/demo-5-distributed-agentic-workloads/manifests/` and are **referenced** from `ValidatedPattern/` Applications — avoid two divergent copies.
2. **Source of truth:** `ValidatedPattern/` = hub GitOps wiring (values, `clusterGroup: hub`, Argo apps). `demo/demo-5-distributed-agentic-workloads/` = this contract, site bootstrap scripts, fault harness, workload/OTEL manifests.
3. Keep `clusterGroupName: hub` for the OCP lab cluster.
4. Add ApplicationSets and/or ACM Policies + Placement (or Argo multi-destination Apps) for: hub OTEL+ClickHouse+Route, site collectors, site `todo` stack; later OpenShell Gateway and central AIOps.
5. **CNV VMs + MicroShift/K3s** are outside the stock “second OpenShift cluster” path: bootstrap with scripts, then register (RHACM **or** Argo kubeconfig).
6. **RHACM import:** try first. If MicroShift/K3s import fails, use **ArgoCD cluster secret + Application `destination.name`** and still set `site_id`. Do not pretend sites are ACM-managed if they are not. Do **not** block Phase 2 on ACM success.
7. Optional later: `edge` / `site` clusterGroup once multi-site placement is proven.

### Site networking

Document in README / `config/sites.yaml`:

| Path | Decision |
|------|----------|
| Hub ↔ site API | RHACM import and/or direct API via kubeconfig used by Argo |
| Site collector → hub OTLP | **OpenShift Route only** (TLS) on hub collector, e.g. `https://otel-intake.apps.<hub-domain>` |
| Todo image pulls | **OpenShift internal registry** via its Route; same `*.apps` reachability |
| Not allowed (OTLP / registry) | NodePort, hosts/IP hacks, long-term `oc port-forward`, or any non-Route path |
| Site VM size (start) | ≥4 vCPU / 16Gi RAM / 40Gi disk; adjust after first boot |
| Nested virt | CNV + host CPU virtualization enabled |

Site VMs **must** resolve and reach hub `*.apps` Route hostnames (OTLP intake **and** internal registry). If they cannot, fix DNS/network — do not add alternate OTLP or registry paths.

### Rules for this topology

1. **CNV VMs = site simulators (machines), not the workload unit.** Production sites are physical/appliance MicroShift/K3s; VMs are lab stand-ins.
2. **Run demo apps and OpenShell sandboxes inside guest clusters**, not only as pods on the OCP host (except hub Gateway / AIOps / pattern operators).
3. **Register every site** with `site_id=site-a|site-b|…` via RHACM import **or** Argo cluster registration.
4. Site count: first slice ≥1 **MicroShift** VM; full PoC ≥2 VMs with MicroShift **and** K3s.
5. **Central AIOps on the hub** (Phase 4+); sites export telemetry only for v1.
6. Deliver hub/site apps via **ArgoCD** (and ACM Placement when imported) from `ValidatedPattern/` + demo-5 manifests. Scripts may bootstrap CNV/OS and break-glass only.
7. Optional later: Kata / OpenShell MicroVM **inside** a site for stronger isolation — separate from site simulation.

### What not to do

- Do not treat CNV VMs as a substitute for lightweight Kubernetes sites.
- Do not expect stock Multicloud GitOps alone to provision CNV+MicroShift/K3s or to supply the AIOps workload.
- Do not run remediating AIOps agents on site VMs or inside tenant sandboxes for v1.
- Do not require a second full OpenShift managed cluster for v1 if CNV sites work.
- Do not pull OpenShell or AIOps remediation (including model wiring) into Phase 0–2.
- Do not use LangChain/AutoGen for the first AIOps harness — prefer tool-calling loops consistent with `research/agentic_aiops_architectures`.
- Do not ship todo without a verified OTEL path to hub ClickHouse (§5).
- Do not commit Vault secrets; use pattern home-directory `values-secret` practice.
- Do not float OpenShell on `main`/`dev` — pin a release tag at Phase 3.

---

## 3. OpenShell concepts (for implementers)

Relevant from **Phase 3** onward. Pin version per §11.

A **sandbox** is the local execution boundary (container/pod/microVM) containing:

- **Supervisor** (`openshell-sandbox`): starts first, outbound session to Gateway, policy proxy, credential injection, launches children
- **Workload child**: agentic or non-agentic app, unprivileged, constrained by Landlock / seccomp / egress / inference routing

AIOps must correlate telemetry with at least: `sandbox_id`, `policy_revision`, `site_id`, and workload identity.

---

## 4. Build phases (implement in order)

### Phase 0 — Hub via `ValidatedPattern/`

- Prerequisites: OpenShift with default StorageClass; CNV available or installable; `oc` login as `cluster-admin`.
- Install Multicloud GitOps from **`ValidatedPattern/`** (`pattern.sh` and/or Validated Patterns Operator + Pattern CR, `clusterGroupName: hub`).
- Secrets: home-directory values file per pattern docs — never commit.
- Verify ArgoCD sync; load Vault secrets if required.
- Confirm RHACM hub and OpenShift GitOps healthy.
- Add demo-5 hub apps: OTEL collector with **OTLP OpenShift Route**, ClickHouse in `demo5-observability`; empty `demo5-aiops` (+ `demo5-openshell` stub). Model / LLM wiring belongs in Phase 4+.
- Confirm CNV; define VM templates per §2 sizing.

### Phase 1 — Site VMs + lightweight K8s + registration

- Provision ≥1 **MicroShift** site VM (full PoC: ≥2 — add K3s).
- Install lightweight Kubernetes; ensure API + OTLP Route reachability to hub.
- Register site: **try RHACM import**; on failure use **ArgoCD cluster kubeconfig**. Always set `site_id`.
- Deploy **site OTEL Collector** → hub Route (inject `site_id`).
- Smoke: hub receives collector heartbeat/metrics — **todo not required yet**.

### Phase 2 — Todo + Postgres (OTEL-orchestrated)

- Deploy on **≥1 site** via GitOps. Second-site replica = full PoC (§5.2), not first-slice gating.
- **App (decided):** self-built **Python FastAPI** todo under `manifests/site-workloads/todo/src/` with **native OTEL** (OTLP/HTTP **traces, metrics, and logs**). Build/push from `src/Containerfile` to the **OpenShift internal registry** (Route); MicroShift pulls via pull secret + trusted CA. Do **not** use `quay.io/julin/todo:v1`.
- Postgres: `registry.redhat.io/rhel8/postgresql-10` (or current pin) + **postgres_exporter** — do not rebuild Postgres for telemetry.
- Manifests: `manifests/site-workloads/todo/` (kustomize). Site overlays must set `SITE_ID` and `OTEL_EXPORTER_OTLP_ENDPOINT` (site collector).
- Topology inspiration: `demo/demo-4-openshift-network-observability` (NetObserv-only assumptions removed).
- Secrets: lab `Secret` → pod **env** (`POSTGRES_*` / exporter keys) through Phase 2 exit; later replace the Secret source with **Vault** / pattern ESO — same env contract; do not invent a parallel secrets stack.
- Pipeline: todo/agent + exporter → **site collector** → **hub Route** → **hub collector** → **ClickHouse**.
- Resource attributes: `service.name`, `site_id`, `deployment.environment`, `k8s.namespace.name` (and `app`, `component=todo|postgres|loadgen`).
- **Load generator** required (idle apps invalid for MTTD demos).
- Verify script: **traces + metrics + logs** in ClickHouse by `site_id`; **fail closed** if any signal missing.
- **Fault injection vs GitOps:** pause/disable Argo sync (or use non-reverted mutations); document harness switch.

### Phase 3 — OpenShell on sites

- **Depends on:** Phase 2 green.
- Pin **`OPENSHELL_VERSION`** to a stable release tag (see §11); record in `config/openshell-version.env` + README.
- If Kubernetes compute driver is unusable, **stop and document** — do not fake OpenShell with plain pods.
- Deploy Gateway on hub; supervisors/sandboxes on sites via Argo/ACM.
- Emit platform signals into the same OTEL pipeline.

### Phase 4 — Central AIOps overlay

- **Depends on:** Phase 2. OpenShell harness faults need Phase 3.
- Implement harness/tooling structure first; the **model endpoint is a Phase 4 AIOps component** — stub until wired (§11).
- Central hub agents only:
  - **Observe:** query hub **ClickHouse** (the OTEL→AIOps handover).
  - **Act:** remediate via **MCP servers** and other gated tools (site kube APIs, OpenShell Gateway when Phase 3 exists) — todo/DB first; OpenShell when available.
- Prefer adapting `research/agentic_aiops_architectures` over a new framework.
- Coordinate fault injection with Argo (Phase 2 note).

### Phase 5 — Hardening

- Disconnect/reconcile tests; GitOps pause behavior under chaos.
- Optional site-local detect-only observers.
- Optional `edge` clusterGroup; EdgeLake; Kata/confidential.

---

## 5. Test workloads (inside site K8s)

Primary non-agentic app: **todo + Postgres with complete OTEL** — not OTEL Demo, not pattern `hello-world` / `config-demo`.

### 5.1 Primary workload — Todo + PostgreSQL

| Component | Requirement |
|-----------|-------------|
| **Todo** | Self-built Python app in `manifests/site-workloads/todo/src/`; native OTEL (**traces, metrics, logs**); `/health` + `/ready` + `/api` |
| **Image** | Build from `src/Containerfile`; push to **OpenShift internal registry** (Route); pin in kustomize `images:` |
| **PostgreSQL** | Per-site DB; creds via pod **env** from lab Secret (Phase 2) then Vault/ESO; **postgres_exporter** |
| **Loadgen** | Continuous traffic so RED metrics/traces/logs move (`todo-client` ns) |
| **Site collector** | OTLP in (+ scrape postgres_exporter) → add `site_id` → hub **Route** (traces, metrics, logs) |
| **Hub** | Collector → ClickHouse; AIOps reads this later |
| **Labels** | `app=todo`, `workload_class=non-agentic`, `site_id`, `component=todo\|postgres\|loadgen` |
| **Delivery** | ArgoCD Application (ACM Placement when imported) |

**Checklist:** Image built/pushed; GitOps deploys stack; FastAPI/SQLAlchemy spans in ClickHouse; todo RED + postgres exporter metrics; app logs in ClickHouse; filter by `site_id`; one documented fault has a clear signature **before** Phase 4.

### 5.2 Phase 3+ workloads

| Workload | Class | Notes |
|----------|-------|-------|
| Sandboxed agentic worker | Agentic in OpenShell | Not the AIOps agent |
| OpenShell Gateway/supervisor canaries | Platform | Fabric faults |
| MCP tool server | Support | Optional but recommended with agentic worker |
| Second-site todo replica | Non-agentic | Full PoC cross-site RCA |
| Sandboxed non-agentic Job | OpenShell lifecycle | Optional |

### 5.3 Defer / avoid for first slice and v1 defaults

- OTEL Demo as primary app  
- Pattern sample apps as AIOps targets  
- AIOps agents on site VMs  
- Model / LLM wiring before Phase 4 (it is an AIOps component, not a Phase 0–2 dependency)  
- NetObserv **instead of** OTEL  
- Unpinned OpenShell  

### 5.4 Manifest requirements

1. Complete OTEL path to hub ClickHouse (site collector → hub Route; **traces, metrics, logs**)  
2. Stable labels (`site_id`, `app`, `workload_class`)  
3. Ground-truth fault entries in YAML  
4. Nested-VM-friendly resources  
5. Sandbox policies: only required egress (collector, MCP, approved inference)

---

## 6. Fault catalog

### Platform / site

- Stop/start site VM; cut NIC / path to hub or Gateway  
- RHACM detach **or** Argo cluster-secret removal / unreachable destination  
- Scale Gateway to 0 / break Gateway Service (Phase 3+)  

### Runtime / security (OpenShell, Phase 3+)

- Kill supervisor; deny-all egress; bad inference backend; fill sandbox disk  

### Workload (todo + Postgres — first slice)

- Scale todo to 0; kill todo/Postgres pods  
- NetworkPolicy todo↔Postgres; bad JDBC URL  
- Toxiproxy-style pool/latency injection (demo-4 style)  

Score **workload** vs **fabric** MTTD/MTTR separately when both apply. GitOps must not silently self-heal injected faults.

---

## 7. AIOps overlay expectations

| Pattern | v1? | Notes |
|---------|-----|-------|
| **Central on hub** | **Yes** | Default |
| **On each site VM** | **No** | Detect-only observers only as Phase 5+ option |
| **LLM/model** | Phase 4+ | AIOps solution component — wire with engage/act; not first-slice scope |

**Observe handover:** site collectors → hub Route → hub collector → **ClickHouse**. AIOps consumes telemetry **only** from this backend for detect / RCA (not by scraping sites directly for the primary path).  
**Engage / Act:** Phase 4+; AIOps remediates through **MCP servers** and other gated tools (site APIs, OpenShell Gateway when present); model endpoint + Vault for tool secrets when needed. ClickHouse is not the act channel.  
**Metrics:** MTTD/MTTR semantics from `research/agentic_aiops_architectures`

---

## 8. Repository layout (target)

```text
# repo root
ValidatedPattern/         # Multicloud GitOps — hub Argo/RHACM/Vault SSOT
demo/demo-5-distributed-agentic-workloads/
  AGENTS.md               # this contract
  README.md               # quickstart: ValidatedPattern path, pins, Route hostname, secrets path
  docs/
    ARCHITECTURE.md
    FAULTS.md
  manifests/
    hub/                  # collector+Route, ClickHouse; later Gateway, AIOps stubs
    site-workloads/todo/  # self-built Python todo, Postgres, exporter, loadgen
    site-observability/   # site collector
    openshell/            # Phase 3+; version-pinned
  scripts/
    provision-site-vms.sh
    bootstrap-site-k8s.sh # distro install + RHACM import and/or Argo cluster secret
    deploy-todo-otel.sh   # verify traces+metrics+logs → ClickHouse; fail closed
    install-hub-pattern.sh
    inject-fault.sh       # includes Argo sync pause
  config/
    sites.yaml
    fault_ground_truth.yaml
    openshell-version.env # Phase 3+: OPENSHELL_VERSION=vX.Y.Z
```

---

## 9. Success criteria

### 9.1 First build slice (exit Phase 2)

1. Hub from **`ValidatedPattern/`**: ArgoCD + RHACM + Vault healthy.  
2. ≥1 CNV **MicroShift** site with `site_id` (RHACM **or** Argo registration).  
3. Hub OTLP **Route** works from site collector.  
4. Self-built todo image + Postgres + loadgen; **traces, metrics, and logs** in ClickHouse by `site_id`.  
5. Verify script fails closed if OTEL path is broken (any of traces / metrics / logs missing).  
6. ≥1 documented todo/DB fault shows a clear telemetry signature (no AIOps agent required).

### 9.2 Full PoC (Phases 3–4+)

1. ≥2 sites: MicroShift **and** K3s, registered and labeled.  
2. OpenShell pinned + Gateway + ≥1 real tenant sandbox.  
3. Central AIOps remediates ≥3 faults (≥1 todo/DB, ≥1 platform/OpenShell) with MTTD/MTTR — once the Phase 4 model + tool surface is wired.  
4. README documents topology, `ValidatedPattern/` usage, Route, registration mode, secrets path, OpenShell pin, GitOps pause for chaos.

---

## 10. Decision log (do not regress)

| Decision | Choice |
|----------|--------|
| OpenShell role | Managed runtime for tenant workloads (Phase 3+) |
| AIOps role | Overlay operating OpenShell + workloads (Phase 4+) |
| AIOps observe handover | Hub **ClickHouse** (OTEL streams in; AIOps reads) |
| AIOps act path | **MCP servers** + gated tools (site APIs, OpenShell Gateway); not via ClickHouse |
| Lab distribution | CNV VMs on one OCP; **MicroShift first**, then K3s (full PoC: both) |
| Hub foundation | Multicloud GitOps in **`ValidatedPattern/`** |
| GitOps | **ArgoCD** (OpenShift GitOps) |
| Fleet | **RHACM first**; **Argo kubeconfig destinations** if import blocked |
| Secrets | Lab Secret → pod **env** through Phase 2; later **Vault** / ESO (same env keys); never commit home-dir `values-secret` |
| Hub OTLP | **OpenShift Route only** (TLS) to hub collector |
| Todo app | **Self-built Python FastAPI** + native OTEL (`manifests/site-workloads/todo/`) |
| Prebuilt `quay.io/julin/todo` | **Do not use** |
| Postgres telemetry | **postgres_exporter** (do not rebuild Postgres image) |
| OTEL signals (Phase 2) | **Traces + metrics + logs** → hub ClickHouse (fail closed) |
| Todo image registry | **OpenShift internal registry** (Route); site pull secret + CA |
| LLM / AIOps model | Phase 4+ **AIOps component** (not first-slice scope) |
| OpenShell version | **Pin stable release** at Phase 3 (e.g. check releases; was `v0.0.92` as of 2026-07-27) |
| First build slice | Phase 0–2 only |
| AIOps placement | **Central hub** |
| Sample / OTEL Demo apps | Not primary AIOps targets |

---

## 11. Decided answers (formerly open questions)

| # | Topic | Decision |
|---|--------|----------|
| 1 | Pattern hosting | **`ValidatedPattern/`** in this repo |
| 2 | Hub OTLP | **OpenShift Route only** (TLS); no NodePort, hosts/IP, or port-forward fallbacks |
| 3 | Todo app | **Self-built Python FastAPI** with native OTEL; image from `src/Containerfile` |
| 4 | RHACM import | Try first; else **ArgoCD multi-destination / kubeconfig**; do not block Phase 2 |
| 5 | LLM / AIOps model | Phase 4+ AIOps component; stub harness until wired — not a Phase 0–2 dependency |
| 6 | OpenShell | **Pin** stable tag at Phase 3; no floating `main`/`dev` |
| 7 | First site distro | **MicroShift** for first slice; K3s added for full PoC (≥2 sites) |
| 8 | Phase 2 secrets | Lab Secret → pod **env** (`POSTGRES_*`, exporter keys); Vault/ESO later with same env contract |
| 9 | OTEL signals (Phase 2 exit) | **Traces + metrics + logs** all required in ClickHouse by `site_id`; verify fail-closed |
| 10 | Todo image registry | **OpenShift internal registry** exposed via Route; MicroShift pulls with CA + pull secret |
| 11 | AIOps observe / act | **Observe** = hub ClickHouse (OTEL in); **Act** = MCP servers + gated tools (site APIs / OpenShell Gateway) |

### Todo app (item 3) — implementation pointer

- Source: `manifests/site-workloads/todo/src/` (`app.py`, `telemetry.py`, `requirements.txt`, `Containerfile`)
- Manifests: `manifests/site-workloads/todo/base/` + `overlays/<site_id>/`; hub: `manifests/hub/`
- Site collector: `manifests/site-observability/otel-collector.yaml` (overlay sets `HUB_OTLP_ENDPOINT` / `SITE_ID`)
- Do **not** use `quay.io/julin/todo:v1` or a Java agent for this service
- Remaining gaps: live `oc login` + pattern install; pin todo image to OpenShift internal registry; replace `REPLACE_ME` Route host in overlay / `config/sites.yaml`; CNV VM + MicroShift bootstrap still manual stubs

