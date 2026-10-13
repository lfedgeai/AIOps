# Phase 6 — AAP + Gitea + ansible-mcp

**Time:** ~20–40 minutes (AAP operator wait) + **manual subscription license (Gateway UI)**  
**Orchestrator:** `./scripts/greenfield-install.sh aap` (install only — stops before wire)  
**Helper:** `./scripts/phase6-aap.sh` (`plan` · `deploy` · `wire` · `check` · `status` · `verify`)

Delivers **governed heal** — OpenClaw launches pre-approved Ansible job templates via **ansible-automation MCP**; playbooks live in **Gitea**.

**License policy:** Subscription manifest is applied **only in the AAP Gateway UI** by the operator. It is **not** stored in git, site config, or automated by greenfield scripts.

---

## Before you start

| Requirement | Check |
|-------------|--------|
| Phase 2 complete | `openshell ✓` · OpenClaw Running |
| Phases 3–5 recommended | MLflow audit + TrustyAI guardrails wired |
| AAP subscription | Export manifest `.zip` from [console.redhat.com](https://console.redhat.com) (keep off git) |
| Gitea image | `quay.io/${QUAY_ORG}/gitea-openshift:gitea-openshift-v1` (see `IMAGE-MIRRORS.md`) |
| Cluster-admin | `oc whoami` |

```bash
./scripts/phase6-aap.sh plan
```

---

## Step 1 — Install AAP (automated)

```bash
cd ~/AIOps/demo/demo-4-greenfield-install
./scripts/greenfield-install.sh aap
# or: ./scripts/phase6-aap.sh deploy
```

Installs AAP 2.7 operator + minimal `netobserv-aap` instance (controller + gateway; hub disabled).

---

## Step 2 — YOU apply license (manual)

1. Open Gateway URL from `./scripts/phase6-aap.sh status`
2. Login: `admin`
3. Password:

```bash
oc -n ansible-automation-platform get secret netobserv-aap-admin-password \
  -o jsonpath='{.data.password}' | base64 -d; echo
```

4. Upload subscription manifest in the Gateway UI (Settings / Subscriptions)
5. Confirm license shows **active**

---

## Step 3 — Wire (after license)

```bash
./scripts/phase6-aap.sh wire
./scripts/phase6-aap.sh check
./scripts/phase6-aap.sh verify
```

Wires Gitea, bootstraps job templates (`netobserv-heal-db-path`, `netobserv-restore-policy`), deploys `ansible-mcp`.

**Next phase:** `./scripts/greenfield-install.sh agent`

---

## What gets installed

| Component | Namespace | When |
|-----------|-----------|------|
| AAP operator + instance | `ansible-automation-platform` | `deploy` |
| Gitea | `gitea` | `wire` |
| `ansible-mcp` + launcher secret | `openclaw` | `wire` |
| Job templates | AAP | `wire` |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| AAP CSV stuck | `oc get csv,sub -n ansible-automation-platform` |
| Instance not Ready | `oc describe ansibleautomationplatform netobserv-aap -n ansible-automation-platform` |
| `wire` fails 401 | License not active in Gateway UI |
| Bootstrap fails | Re-run `./scripts/phase6-aap.sh wire` after license |
| Gitea ImagePullBackOff | `./scripts/mirror-gitea-image-to-quay.sh` |

Optional CLI license tool (bastion only, not greenfield): `$DEMO_KIT_ROOT/scripts/apply-aap-license.sh` — use only if you prefer API over UI; never commit the manifest.

---

## Quick reference

```bash
./scripts/phase6-aap.sh deploy    # install — stop here for license
./scripts/phase6-aap.sh wire      # continue after Gateway UI license
./scripts/phase6-aap.sh status
./scripts/phase6-aap.sh verify
```
