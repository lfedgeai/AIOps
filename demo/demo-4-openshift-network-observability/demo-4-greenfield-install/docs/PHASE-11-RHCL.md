# Phase 11 — RHCL OAuth for OpenClaw (optional)

**Time:** ~10–20 minutes  
**Orchestrator:** `./scripts/greenfield-install.sh rhcl`  
**Helper:** `./scripts/phase11-rhcl.sh` (`plan` · `operator` · `ingress` · `deploy` · `check` · `status` · `verify`)

Adds an **enterprise OAuth front door** for the OpenClaw Control UI via **Red Hat Connectivity Link (Kuadrant + Authorino)** and **OpenShift OAuth**.

**Optional:** Slack, MCP, and the legacy `openclaw` Route + gateway token continue to work without RHCL.

---

## Before you start

| Requirement | Check |
|-------------|--------|
| Phase 2 complete | `openclaw` deployment + service |
| cert-manager | `oc get pods -n cert-manager` |
| OperatorHub | **RHCL operator** (`rhcl-operator`) |
| Phase 3 recommended | RHOAI / MLflow on separate gateway (unchanged) |

```bash
./scripts/phase11-rhcl.sh plan
```

Skip entirely: `SKIP_RHCL=1`

---

## Architecture

```text
Browser → https://openclaw-rhcl.apps.<ingress>/
              │
              ▼
      openclaw-gateway (Kuadrant / Authorino)
              │ OpenShift OAuth (302)
              ▼
      OpenClaw Control UI (same agent, skills, MCP)

Legacy (unchanged): Route openclaw + gateway token — Slack / lab access
```

**Option A (default):** dedicated `openclaw-gateway` in `openclaw` namespace — OpenClaw is **not** attached to RHOAI `data-science-gateway`.

---

## Deploy

```bash
./scripts/phase11-rhcl.sh deploy
# or via orchestrator:
./scripts/greenfield-install.sh rhcl
```

This runs:

| Step | What |
|------|------|
| `install-rhcl-ingress.sh install` | RHCL operator subscription + Kuadrant CR |
| `install-rhcl-ingress.sh ingress` | Gateway, HTTPRoutes, OAuthClient, Authorino patch |

Step-by-step:

```bash
./scripts/phase11-rhcl.sh operator   # ~10 min (CSV + Kuadrant Ready)
./scripts/phase11-rhcl.sh ingress    # gateway + OAuth
```

Operator only (no public ingress): `SKIP_RHCL_INGRESS=1 ./scripts/phase11-rhcl.sh deploy`

---

## Verify

```bash
./scripts/phase11-rhcl.sh verify
./scripts/phase11-rhcl.sh status
```

**Manual test:** open **incognito** → `https://openclaw-rhcl.apps.<your-ingress>/` → OpenShift login → Control UI.

Legacy path still works:

```bash
oc -n openclaw get route openclaw -o jsonpath='https://{.spec.host}{"\n"}'
```

**Next phase:** `./scripts/greenfield-install.sh verify` (Phase 12)

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| RHCL CSV stuck | `oc get csv,installplan -n kuadrant-system` · wait or check OperatorHub |
| Kuadrant not Ready | `oc get kuadrant -n kuadrant-system -o yaml` · cert-manager pods |
| Gateway not Programmed | `oc describe gateway openclaw-gateway -n openclaw` |
| OAuth redirect loop | Re-run `./scripts/phase11-rhcl.sh ingress` (runs `patch-openclaw-oidc-openshift.sh`) |
| Control UI CORS | `ensure-openclaw-ui-origin.sh` (auto-run on ingress) |
| Logo 302 to OAuth | `HTTPRoute openclaw-ui-static` — re-run ingress |
| Skip RHCL on greenfield | `SKIP_RHCL=1` — use legacy Route + gateway token |

---

## Quick reference

```bash
./scripts/phase11-rhcl.sh plan
./scripts/phase11-rhcl.sh deploy
./scripts/phase11-rhcl.sh check
$KIT/install-rhcl-ingress.sh status
$KIT/netobserv-e2e-openclaw-test.sh ui-hints
```
