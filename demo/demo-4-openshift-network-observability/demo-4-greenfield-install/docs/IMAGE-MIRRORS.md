# Container image mirrors — greenfield

Avoid ImagePullBackOff by knowing which images are **site-owned** (your Quay org) vs **upstream**, and running pull checks before each phase.

**Verify:** `$DEMO_KIT_ROOT/scripts/verify-image-pulls.sh`  
**Pins:** `$DEMO_KIT_ROOT/scripts/supply-chain-pins.env` + `config/env.local`

Set your org once in `config/env.local`:

```bash
export QUAY_ORG=your-org
export OPENCLAW_GATEWAY_IMAGE=quay.io/${QUAY_ORG}/openclaw-openshift:openclaw-v2026.6.11
export OPENCLAW_SANDBOX_IMAGE=quay.io/${QUAY_ORG}/openclaw-openshift:openclaw-v2026.6.11
export GITEA_IMAGE=quay.io/${QUAY_ORG}/gitea-openshift
export GITEA_TAG=gitea-openshift-v1
```

---

## P0 — must exist before Phases 1–2

| Image | Default ref | Owner | Risk | Action |
|-------|-------------|-------|------|--------|
| **OpenClaw** (gateway + sandbox) | `quay.io/${QUAY_ORG}/openclaw-openshift:openclaw-v2026.6.11` | **You** | **High** — upstream `ryan_nix` June digest deleted; newer tags break init | Mirror once: `./scripts/mirror-openclaw-image-to-quay.sh` |
| **NetObserv todo** | `quay.io/${QUAY_ORG}/todo:v1` | **You** | Low if already pushed | Push or mirror before Phase 1 |

Phase 2 **always** rewrites the lab checkout (`~/labs/openshell-on-openshift-lab`) away from `ryan_nix@sha256:a91dbc…` before deploy.

**Gitea mirror (one-time, Phase 6):** `./scripts/mirror-gitea-image-to-quay.sh`

---

## P1 — later phases (version-pinned upstream; usually OK)

| Image | Pin | Phase | Mirror to your org? |
|-------|-----|-------|---------------------|
| **Gitea** | `quay.io/${QUAY_ORG}/gitea-openshift:gitea-openshift-v1` | AAP / Phase 6 | Yes — from `ghcr.io/kwkoo/gitea-openshift:latest` |
| Kraken | `quay.io/krkn-chaos/krkn:v5.2.7` | Demo faults | Optional — official Quay |
| MLflow | `ghcr.io/mlflow/mlflow:v3.1.1` | MLflow wire | Unlikely needed |
| SPIFFE Helper | `registry.redhat.io/...` (digest in script) | Phase 10 | Use cluster pull secret |

---

## Pre-flight pull check

```bash
export DEMO_KIT_ROOT=~/AIOps/demo/demo-4-platform-kit
export QUAY_ORG=your-org
"$DEMO_KIT_ROOT/scripts/verify-image-pulls.sh"
```

---

## OpenClaw mirror workflow

1. On **any** cluster where the pinned digest still runs, on a bastion with `oc`:

```bash
cd demo-4-greenfield-install
export QUAY_ORG=your-org
export QUAY_TAG=openclaw-v2026.6.11
./scripts/mirror-openclaw-image-to-quay.sh
```

2. On your **greenfield** bastion, set `OPENCLAW_GATEWAY_IMAGE` / `OPENCLAW_SANDBOX_IMAGE` in `config/env.local` (see above).

3. Deploy Phase 2: `./scripts/greenfield-install.sh openshell`
