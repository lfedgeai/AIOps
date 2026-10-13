# Greenfield start checklist — new cluster (Phases 0–3)

Run on the **new cluster bastion**. Complete **cluster login** and **site secrets** before Phase 1.

## Before you SSH to bastion (laptop)

- [ ] Both folders synced: `./scripts/sync-to-bastion.sh all` (set `BASTION_HOST`)

## On bastion — order matters

```bash
cd ~/AIOps/demo/demo-4-greenfield-install
export DEMO_KIT_ROOT=~/AIOps/demo/demo-4-platform-kit
export PATH="$HOME/.local/bin:$PATH"

# 1) OpenShift login (NOT in config prompt)
export OCP_API="https://api.<cluster-name>.<base-domain>:6443"
oc login "$OCP_API" -u kubeadmin -p '<password>'
./scripts/cluster-login.sh check

# 2) Site secrets (AWS + LLM for phases 1–3)
./scripts/greenfield-install.sh config prompt

# 3) Preflight (must exit 0)
./scripts/greenfield-install.sh preflight
```

See [CLUSTER-LOGIN.md](CLUSTER-LOGIN.md) for token/username options and troubleshooting.

Preflight installs **Helm 3** if missing (`install-helm3.sh`).

## Phase run order

```bash
./scripts/greenfield-install.sh netobserv       # Phase 1 (~20–40 min)
./scripts/greenfield-install.sh openshell       # Phase 2 (automated deploy + harden + probe)
./scripts/phase2-openshell.sh check && ./scripts/phase2-openshell.sh ui
./scripts/greenfield-install.sh rhoai           # Phase 3 (~30–50 min)
./scripts/phase3-rhoai.sh verify
./scripts/greenfield-install.sh phases
```

## Go / no-go

| Check | Required |
|-------|----------|
| `cluster-login.sh check` | **Yes** |
| `preflight` exit 0 | **Yes** |
| Site config `netobserv` + `openshell` valid | **Yes** |
| cluster-admin + OVN-Kubernetes | **Yes** |
| Slack tokens | No (until Phase 8) |
| Phase 2 automated | Yes (`phase2-openshell.sh deploy`) |
