# Phase 4 — Grafana + OTel federation

**Time:** ~20–40 minutes (Grafana Operator wait)  
**Orchestrator:** `./scripts/greenfield-install.sh grafana`  
**Helper:** `./scripts/phase4-grafana.sh` (`plan` · `deploy` · `check` · `status` · `verify`)

Delivers the **Network AIOps — Gateway Diagnostics** dashboard: NetObserv flows/packets + OpenClaw agent economics (tokens, latency, estimated LLM cost).

---

## Before you start

| Requirement | Check |
|-------------|--------|
| Phase 2 complete | `./scripts/greenfield-install.sh phases` → `openshell ✓` |
| OpenClaw Running | `oc -n openclaw get deploy/openclaw` |
| NetObserv FlowCollector | Phase 1 — metrics for federation panels |
| StorageClass for Grafana PVC | `gp3-csi` or site `openshift.storage_class` |
| Cluster-admin | `oc whoami` |

```bash
./scripts/phase4-grafana.sh plan
```

---

## What gets installed

| Component | Namespace | Purpose |
|-----------|-----------|---------|
| Grafana Operator | `netobserv-demo` | Operator + CSV pin |
| `network-aiops` Grafana | `netobserv-demo` | Dashboard UI (Route) |
| OTel collector | `openclaw` | OpenClaw gateway metrics |
| `openclaw-otel-prometheus` | `openclaw` | Scrapes OTel + **federates NetObserv** |
| GrafanaDatasource `prometheus-openclaw-otel` | `netobserv-demo` | Single Prometheus backend |
| Dashboard `network-aiops-openclaw` | `netobserv-demo` | Gateway Diagnostics v16 |

Pins: `$DEMO_KIT_ROOT/manifests/grafana-network-aiops/` · Grafana Operator `v5.24.0`

---

## Automated install

```bash
cd ~/AIOps/demo/demo-4-greenfield-install
./scripts/greenfield-install.sh grafana
```

Or step-by-step:

```bash
./scripts/phase4-grafana.sh deploy
./scripts/phase4-grafana.sh check
./scripts/phase4-grafana.sh verify
```

**Grafana login:** `admin` / `netobserv-demo` (lab default)

```bash
oc -n netobserv-demo get route grafana-network-aiops -o jsonpath='https://{.spec.host}{"\n"}'
```

**Next phase:** `./scripts/greenfield-install.sh guardrails`

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Grafana Operator CSV stuck | Catalog / pull secret | `oc get csv,sub -n netobserv-demo` |
| Grafana pod 503 | SQLite / PVC wedged | `$DEMO_KIT_ROOT/scripts/install-grafana-network-aiops.sh recover` |
| Empty NetObserv panels | Federation down | `$DEMO_KIT_ROOT/scripts/sync-grafana-demo-metrics.sh sync` |
| `federate-netobserv` not up | Wrong scrape target | Config must use `prometheus-k8s.openshift-monitoring.svc:9091/federate` |
| Duplicate Grafana in `default` | Early `apply -k` without `-n` | `$DEMO_KIT_ROOT/scripts/install-grafana-network-aiops.sh cleanup-default` |
| OTel panels empty | No Control UI traffic yet | Send one chat message; wait ~30s for OTel flush |

Full presenter guide: `$DEMO_KIT_ROOT/docs/GRAFANA-PRESENTER-GUIDE.md`

---

## Quick reference

```bash
./scripts/phase4-grafana.sh plan
./scripts/phase4-grafana.sh status
./scripts/phase4-grafana.sh check
./scripts/phase4-grafana.sh verify
$DEMO_KIT_ROOT/scripts/sync-grafana-demo-metrics.sh status
```
