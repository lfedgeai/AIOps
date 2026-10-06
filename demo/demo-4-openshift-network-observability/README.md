# Network Observability on OpenShift — Demo Kit (IPI-on-AWS)

A self-contained kit for **Red Hat OpenShift Network Observability** demos — from a **scripted NetObserv + fault-injection lab** (`scripts/`) to a **full greenfield agentic AIOps stack** (`demo-4-greenfield-install/` + `demo-4-platform-kit/`). The basic path highlights per-tenant visibility, database microsegmentation, and auditable flow evidence. It also injects a real incident, captures flows with the `oc netobserv` CLI, and analyzes evidence with an LLM.

Tested on **OpenShift 4.21**.

The environment and application deployment and configuration demo video can be found at this [link](https://drive.google.com/file/d/1YrsLr0K4wLte1XAAa7ITZHX5Zz7370KY/view?usp=sharing)

The second demo video on fault injection, flow capture and analysis by frontier AI model can be found at this [link](https://drive.google.com/file/d/1IXMIjAY-xOEWHWdXe-_qS8g45qG6GpoX/view?usp=drive_link) 

---

## Choose your path

This directory contains **two related demos**:

| Path | Folder | Best for |
|------|--------|----------|
| **NetObserv basics** | [`scripts/`](scripts/) | Stand up NetObserv + todo app, inject a fault, capture flows, analyze evidence with an LLM |
| **Full agentic AIOps (greenfield)** | [`demo-4-greenfield-install/`](demo-4-greenfield-install/) + [`demo-4-platform-kit/`](demo-4-platform-kit/) | Phased install of NetObserv, OpenClaw/OpenShell, RHOAI/MLflow, Grafana, TrustyAI Guardrails, AAP, Slack, event-driven AIOps, SPIFFE, and RHCL — with end-to-end **Demo A** (fault → investigate → heal) |

The **basic quickstart below** uses `scripts/` only.

For the **full stack**, start here:

```bash
git clone https://github.com/lfedgeai/AIOps.git
cd AIOps/demo/demo-4-openshift-network-observability/demo-4-greenfield-install

cp config/site-secrets.example.yaml config/site-secrets.local.yaml
# oc login on your cluster bastion first — see docs/CLUSTER-LOGIN.md
./scripts/greenfield-install.sh config prompt   # AWS + LLM + Slack wizard
./scripts/greenfield-install.sh phases          # Phases 1–12
```

The platform kit (`../demo-4-platform-kit`) is resolved automatically as `DEMO_KIT_ROOT`. See [`demo-4-greenfield-install/README.md`](demo-4-greenfield-install/README.md) and [`INSTALL.md`](demo-4-greenfield-install/INSTALL.md) for the full runbook.

**Day-2 / repeat demos:** [`demo-4-greenfield-install/CLUSTER-WAKE.md`](demo-4-greenfield-install/CLUSTER-WAKE.md) · trial run: `"$DEMO_KIT_ROOT/scripts/netobserv-e2e-openclaw-test.sh" demo-a-fast`

---

## What's in here (NetObserv basics — `scripts/`)

| File | Purpose |
|------|---------|
| `install-netobserv-aws.sh` | Installs the Loki Operator + a dedicated S3-backed LokiStack, the Network Observability Operator, and a `FlowCollector`. Prompts for all credentials at runtime. |
| `deploy-netobserv-todo-app.sh` | Deploys a realistic 2-tier app (Quarkus `todo` + PostgreSQL) with a traffic generator and a DB microsegmentation policy, so every NetObserv view has meaningful data. |
| `netobserv-fault-injection.sh` | Induces a controllable JDBC **connection-pool exhaustion** (Toxiproxy DB latency + concurrency ramp). Subcommands: `inject` / `slow` / `heal` / `status` / `restore`. |
| `netobserv-capture-and-bundle.sh` | Runs an `oc netobserv` flow capture on the DB path, then extracts an **evidence bundle** (`evidence.json` + `evidence.md`) from the SQLite `flow.db`, app logs, and datasource config. |

---

## Prerequisites

- OpenShift **4.21** cluster, installed with **IPI on AWS**, using **OVN-Kubernetes** (default).
- Cluster credential mode is `mint`/`passthrough` (non-[STS](https://www.redhat.com/en/blog/what-is-aws-sts-and-how-does-red-hat-openshift-service-on-aws-rosa-use-sts) — this kit uses static S3 credentials).
- `cluster-admin` on the cluster.
- Local CLIs: [oc](https://mirror.openshift.com/pub/openshift-v4/clients/oc/latest/linux/), [aws](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), [jq](https://jqlang.org/download/).
- The **`oc netobserv`** plugin
- An AWS access key/secret with rights to create an S3 bucket (and, optionally, a scoped IAM user).

---

## Quickstart

**1. Clone the repository and go to the scripts directory**

```
git clone https://github.com/lfedgeai/AIOps.git
cd AIOps/demo/demo-4-openshift-network-observability/scripts
```

**2. Install Network Observability** — you will be prompted for cluster login, AWS keys, region and bucket name:

```bash
chmod +x install-netobserv-aws.sh
./install-netobserv-aws.sh 2>&1 | tee netobserv-install.log
```

When prompted, you can optionally enable a bucket-scoped IAM user (recommended for FSI) and the advanced eBPF features (**PacketDrop / DNSTracking / FlowRTT**).

**3. Deploy the sample app + traffic:**

```bash
chmod +x deploy-netobserv-todo-app.sh
./deploy-netobserv-todo-app.sh
```

**4. View flows:** in the console, **Observe → Network Traffic** (allow 3–5 min to populate). Then follow `netobserv-demo-walkthrough.md`.

Watch traffic being generated:

```bash
oc logs -n todo-client deploy/loadgen -f
```

---

## Configuration

Scripts can be overridden using environment variables

**`install-netobserv-aws.sh`**

| Var | Default | Notes |
|-----|---------|-------|
| `LOKI_SIZE` | `1x.extra-small` | `1x.pico` / `1x.extra-small` start lighter for small demo clusters. |
| `SAMPLING` | `50` | 1-in-N packet sampling. |
| `STORAGE_CLASS` | `gp3-csi` | Loki PVC storage class. |
| `KMS_KEY_ID` | *(unset)* | Set to a CMK ARN for SSE-KMS bucket encryption (FSI). |
| `CATALOG_SOURCE` | `redhat-operators` | Point at your mirror for disconnected clusters. |
| `LOKI_CHANNEL` / `NETOBSERV_CHANNEL` | `6.5` | Note that it is currently pinned to version 6.5. |

**`deploy-netobserv-todo-app.sh`**

| Var | Default | Notes |
|-----|---------|-------|
| `APP_NS` | `todo-demo` | App + database namespace. |
| `CLIENT_NS` | `todo-client` | Consumer namespace (set `= APP_NS` to co-locate). |
| `REGISTRY` | `registry.access.redhat.com` | Loadgen image registry; override for a mirror. |
| `ENABLE_DBPOLICY` | `true` | Microsegment PostgreSQL (allow only the app tier). |
| `PROBE_DB` | `true` | Loadgen attempts an unauthorized DB connection (drop demo). |
| `ENABLE_ROUTE_TRAFFIC` | `true` | Also drive north-south traffic via the Route. |

**`netobserv-fault-injection.sh`**

| Var | Default | Notes |
|-----|---------|-------|
| `LATENCY_MS` | `800` | DB latency injected by `slow` (also takes a positional arg). |
| `CONCURRENCY` | `80` | Parallel workers in the heavy load generator. |
| `ACQUISITION_TIMEOUT` | `2` | Seconds Agroal waits for a free connection before failing. |
| `TOXIPROXY_IMAGE` | `ghcr.io/shopify/toxiproxy:2.9.0` | Override for a disconnected mirror. |
| `TARGET_PATH` | `/api` | The todo endpoint that hits the DB. |

**`netobserv-capture-and-bundle.sh`**

| Var | Default | Notes |
|-----|---------|-------|
| `DURATION` | `180` | Capture seconds. Keep captures short (< 10 min). |
| `DB_PORT` | `5432` | Port to filter the flow capture on. |

---

## Fault Injection and AI Analysis Demo

On the RHEL 9 Bastion node, setup `oc netobserv` CLI
```bash
$ podman create --name netobserv-cli registry.redhat.io/network-observability/network-observability-cli-rhel9:1.12
$ podman cp netobserv-cli:/oc-netobserv .
$ sudo mv oc-netobserv /usr/local/bin/
$ oc netobserv version
```

Install `sqlite` on bastion
```
$ sudo dnf install sqlite sqlite-devel -y
```

Set up the incident rig (healthy baseline — Toxiproxy passthrough, load running)
```
$ ./netobserv-fault-injection.sh inject
```

Inject delay/fault, i.e. set latency to 800ms
```
$ ./netobserv-fault-injection.sh slow 800
```

Capture evidence while application is degraded
```
$ ./netobserv-capture-and-bundle.sh
```

Note that we can also run wire-level capture and feed the information to AI Models. The target port is 5432, where the postgresql is listening to.
```
$ oc netobserv packets --protocol=TCP --port=5432
```

Lastly, make use of AI models such as Qwen 3.6 to analyze the `evidence.json` and/or wireshark traces

---

## What gets deployed

```
todo-client (consumer)         todo-demo (application tenant)
  loadgen ──── east-west ─────> todo (Quarkus :8080) ──┐
     │        (cross-namespace)      ▲  Route (edge TLS)│
     └──── direct DB probe ──────────┼──> postgresql :5432
           (DENIED by NetworkPolicy) └── north-south via router

Fault injection and AI Analysis Demo inserts:  todo → toxiproxy → postgresql   (fault injection)
```

- **todo → postgresql:5432** — the real app-to-DB edge (topology + audit evidence).
- **loadgen → todo** — cross-namespace consumer (multi-tenancy view).
- **loadgen → postgresql:5432** — unauthorized, dropped by the DB policy (enforcement demo; visible as a dropped flow when PacketDrop is enabled).

---

## Teardown

```bash
./netobserv-fault-injection.sh restore     # remove Toxiproxy + load, restore original config
./deploy-netobserv-todo-app.sh delete
```

---

## Notes & gotchas

- **Non-STS by design.** This kit uses static S3 credentials for a demo. For a production/STS cluster, use the [Cloud Credential Operator](https://docs.redhat.com/en/documentation/openshift_container_platform/4.20/html/authentication_and_authorization/managing-cloud-provider-credentials) token flow instead.
- **PostgreSQL image** (`registry.redhat.io/rhel8/postgresql-10`) pulls via the cluster's global pull secret — present on a standard install. An `ImagePullBackOff` on `postgresql` means the pull secret, not the manifest.
- **Advanced eBPF features are optional.** The dropped-flow, DNS, and RTT demo moments require PacketDrop / DNSTracking / FlowRTT — enable them at install time.
- **Pool exhaustion is an application-layer condition.** NetObserv shows its *network fingerprint* (rising RTT to :5432, connections plateauing, drops) — strong corroborating evidence. The application logs provide the evidence.
- **The fault rig widens the DB policy.** `inject` temporarily permits `toxiproxy → postgresql` (since the application now reaches the DB through it); `restore` reverts it to todo application only.
- **Loki retention** is usually set to 7, 14 or 28 days. For longer compliance windows, export flows (Kafka / OpenTelemetry) to a long-term store, syslog server or SIEM such as Splunk.
- **Credentials never leave your terminal.** Both scripts prompt at runtime and never take secrets as arguments; the installer applies the S3 secret over stdin so the keys do not appear in the process list.
