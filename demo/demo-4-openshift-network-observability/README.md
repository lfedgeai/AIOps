# Network Observability on OpenShift — Demo Kit (IPI-on-AWS)

A self-contained kit to stand up **Red Hat OpenShift Network Observability** on a fresh IPI-on-AWS cluster and run a sample customer-facing demo. The demo highlights per-tenant visibility, database microsegmentation and auditable flow evidence.

Tested on **OpenShift 4.20**.

---

## What's in here

| File | Purpose |
|------|---------|
| `install-netobserv-aws.sh` | Installs the Loki Operator + a dedicated S3-backed LokiStack, the Network Observability Operator, and a `FlowCollector`. Prompts for all credentials at runtime. |
| `deploy-netobserv-todo-app.sh` | Deploys a realistic 2-tier app (Quarkus `todo` + PostgreSQL) with a traffic generator and a DB microsegmentation policy, so every NetObserv view has meaningful data. |

---

## Prerequisites

- OpenShift **4.20** cluster, installed with **IPI on AWS**, using **OVN-Kubernetes** (default).
- Cluster credential mode is `mint`/`passthrough` (**non-STS** — this kit uses static S3 credentials).
- `cluster-admin` on the cluster.
- Local CLIs: **`oc`**, **`aws`**, **`jq`**.
- An AWS access key/secret with rights to create an S3 bucket (and, optionally, a scoped IAM user).

---

## Quickstart

**0. Clone the repository and go to the Network Observability demo directory**

```
git clone https://github.com/lfedgeai/AIOps.git
cd AIOps/demo/demo-4-openshift-network-observability
```

**1. Install Network Observability** — you will be prompted for cluster login, AWS keys, region and bucket name:

```bash
cd scripts
chmod +x install-netobserv-aws.sh
./install-netobserv-aws.sh 2>&1 | tee netobserv-install.log
```

When prompted, you can optionally enable a bucket-scoped IAM user (recommended for FSI) and the advanced eBPF features (**PacketDrop / DNSTracking / FlowRTT**).

**2. Deploy the sample app + traffic:**

```bash
cd scripts
chmod +x deploy-netobserv-todo-app.sh
./deploy-netobserv-todo-app.sh
```

**3. View flows:** in the console, **Observe → Network Traffic** (allow 3–5 min to populate). Then follow `netobserv-demo-walkthrough.md`.

Watch traffic being generated:

```bash
oc logs -n todo-client deploy/loadgen -f
```

---

## Configuration

Both scripts read sensible defaults; override via environment variables.

**`install-netobserv-aws.sh`**

| Var | Default | Notes |
|-----|---------|-------|
| `LOKI_SIZE` | `1x.extra-small` | `1x.pico` / `1x.extra-small` start lighter for small demo clusters. |
| `SAMPLING` | `50` | 1-in-N packet sampling. |
| `STORAGE_CLASS` | `gp3-csi` | Loki PVC storage class. |
| `KMS_KEY_ID` | *(unset)* | Set to a CMK ARN for SSE-KMS bucket encryption (FSI). |
| `CATALOG_SOURCE` | `redhat-operators` | Point at your mirror for disconnected clusters. |
| `LOKI_CHANNEL` / `NETOBSERV_CHANNEL` | Note that it is currently pinned to version 6.5. |

**`deploy-netobserv-todo-app.sh`**

| Var | Default | Notes |
|-----|---------|-------|
| `APP_NS` | `todo-demo` | App + database namespace. |
| `CLIENT_NS` | `todo-client` | Consumer namespace (set `= APP_NS` to co-locate). |
| `REGISTRY` | `registry.access.redhat.com` | Loadgen image registry; override for a mirror. |
| `ENABLE_DBPOLICY` | `true` | Microsegment PostgreSQL (allow only the app tier). |
| `PROBE_DB` | `true` | Loadgen attempts an unauthorized DB connection (drop demo). |
| `ENABLE_ROUTE_TRAFFIC` | `true` | Also drive north-south traffic via the Route. |

---

## What gets deployed

```
todo-client (consumer)         todo-demo (application tenant)
  loadgen ──── east-west ─────> todo (Quarkus :8080) ──┐
     │        (cross-namespace)      ▲  Route (edge TLS)│
     └──── direct DB probe ──────────┼──> postgresql :5432
           (DENIED by NetworkPolicy) └── north-south via router
```

- **todo → postgresql:5432** — the real app-to-DB edge (topology + audit evidence).
- **loadgen → todo** — cross-namespace consumer (multi-tenancy view).
- **loadgen → postgresql:5432** — unauthorized, dropped by the DB policy (enforcement demo; visible as a dropped flow when PacketDrop is enabled).

---

## Teardown

```bash
# sample app only
./deploy-netobserv-todo-app.sh delete
```

---

## Notes & gotchas

- **Non-STS by design.** This kit uses static S3 credentials for a demo. For a production/STS cluster, use the CCO token flow (see the runbook) instead.
- **PostgreSQL image** (`registry.redhat.io/rhel8/postgresql-10`) pulls via the cluster's global pull secret — present on a standard install. An `ImagePullBackOff` on `postgresql` means the pull secret, not the manifest.
- **Advanced eBPF features are optional.** The dropped-flow, DNS, and RTT demo moments require PacketDrop / DNSTracking / FlowRTT — enable them at install time. The walkthrough notes which segments depend on them.
- **Loki retention** is tested/supported for up to ~30 days. For longer compliance windows, export flows (Kafka / OpenTelemetry) to a long-term store or SIEM.
- **Credentials never leave your terminal.** Both scripts prompt at runtime and never take secrets as arguments; the installer applies the S3 secret over stdin so keys don't appear in the process list.
