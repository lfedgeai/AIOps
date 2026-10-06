# Phase 3 — RHOAI + MLflow Traces

**Time:** ~30–50 minutes (operator install dominates)  
**Orchestrator:** `./scripts/greenfield-install.sh rhoai`  
**Helper:** `./scripts/phase3-rhoai.sh` (`plan` · `check` · `status` · `verify`)

RHOAI provides **platform MLflow + OpenShift AI dashboard** for agent audit (**Traces** and **Runs**). It does **not** replace your external LLM from Phase 2 — OpenClaw still calls LiteMaaS/vLLM for inference.

---

## Before you start

| Requirement | Check |
|-------------|--------|
| Phase 2 complete | `./scripts/greenfield-install.sh phases` → `openshell ✓` |
| OpenClaw Running | `oc -n openclaw get deploy/openclaw` |
| OperatorHub / redhat-operators | Cluster can pull `rhods-operator` |
| Cluster-admin | `oc whoami` |

```bash
./scripts/phase3-rhoai.sh plan
```

---

## What gets installed

| Component | Namespace | Purpose |
|-----------|-------------|---------|
| `rhods-operator` | `redhat-ods-operator` | OpenShift AI operator |
| `default-dsc` | cluster | Dashboard + MLflow enabled; KServe disabled |
| `mlflow` CR + deployment | `redhat-ods-applications` | RHOAI-managed MLflow |
| Experiment `openclaw-netobserv` | MLflow workspace `openclaw` | Agent tool audit |
| `openclaw-netobserv` SA/RBAC | `openclaw` | Token auth to RHOAI MLflow API |

Pins: `$DEMO_KIT_ROOT/manifests/platform-merge/` · channel `stable-3.4` (override `RHOAI_CHANNEL`).

---

## Step 1 — Install minimal RHOAI

```bash
export DEMO_KIT_ROOT="$HOME/AIOps/demo/demo-4-platform-kit"
cd "$HOME/AIOps/demo/demo-4-greenfield-install"

./scripts/greenfield-install.sh rhoai
# or manually:
"$DEMO_KIT_ROOT/scripts/install-rhoai-platform-minimal.sh" install
```

**Expect:** CSV `rhods-operator.*` → `Succeeded` · `default-dsc` → `Ready` · `deploy/mlflow` Available (can take **20–40 min** on a fresh cluster).

Monitor:

```bash
watch -n20 'oc get csv -n redhat-ods-operator; oc get dsc default-dsc -o jsonpath="{.status.phase}{\"\n\"}"; oc -n redhat-ods-applications get deploy/mlflow'
```

**Gate:** `./scripts/phase3-rhoai.sh check` (partial OK after install, before wire)

---

## Step 2 — Wire OpenClaw → RHOAI MLflow

```bash
MLFLOW_BACKEND=rhoai MLFLOW_REMOVE_STANDALONE=1 \
  "$DEMO_KIT_ROOT/scripts/wire-openclaw-mlflow.sh"
```

This script:

1. Resolves experiment `openclaw-netobserv` in RHOAI MLflow
2. Sets `MLFLOW_*` env on `deployment/openclaw`
3. Wires bridge, guard proxy, and `netobserv-mcp` for Traces
4. Removes legacy standalone MLflow in `openclaw` ns when `MLFLOW_REMOVE_STANDALONE=1`

**Gate:** `./scripts/phase3-rhoai.sh check` → OpenClaw `MLFLOW_TRACKING_URI` points at `redhat-ods-applications`

---

## Step 3 — Verify Phase 3

```bash
./scripts/phase3-rhoai.sh check
./scripts/phase3-rhoai.sh verify
./scripts/greenfield-install.sh phases    # rhoai → ✓
```

Presenter URLs:

```bash
oc -n redhat-ods-applications get route rhods-dashboard -o jsonpath='https://{.spec.host}{"\n"}'
oc get mlflow mlflow -n redhat-ods-applications -o jsonpath='{.status.url}{"\n"}'
```

**Next phase:** `./scripts/greenfield-install.sh grafana` — see [PHASE-4-GRAFANA.md](PHASE-4-GRAFANA.md)

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| CSV stuck `Installing` | OperatorHub / pull secret | `oc get csv,sub -n redhat-ods-operator` · fix catalog |
| `default-dsc` not Ready | DSCInitialization pending | `oc get dsci,dsc -o yaml` · wait or check operator logs |
| `deploy/mlflow` missing | MLflow CR not reconciled | `oc get mlflow -n redhat-ods-applications` · re-apply `05-mlflow-cr.yaml` |
| Experiment check fails | Wire not run or RBAC | `wire-openclaw-mlflow.sh` · `07-openclaw-mlflow-rbac.yaml` |
| Standalone MLflow still in `openclaw` | Old path | `MLFLOW_REMOVE_STANDALONE=1` on wire script |
| `mlflow-check` token error | SA missing | Re-run wire; `oc -n openclaw get sa openclaw-netobserv` |

Full design: `$DEMO_KIT_ROOT/docs/PLATFORM-MERGE-DESIGN.md`

---

## Quick reference

```bash
./scripts/phase3-rhoai.sh plan
./scripts/phase3-rhoai.sh status
./scripts/phase3-rhoai.sh check
./scripts/phase3-rhoai.sh verify
```
