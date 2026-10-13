# Greenfield install — Demo A + full platform stack

Replicate the **lab stack** on a **new OpenShift 4.21+** cluster.  
**Run from this folder** (`demo-4-greenfield-install`) — not from the platform-kit folder.

| Path | Role |
|------|------|
| `demo-4-greenfield-install/` | Orchestrator + docs (this folder) |
| `demo-4-platform-kit/` | Platform kit — install scripts, skills, manifests (`DEMO_KIT_ROOT`) |

```bash
export DEMO_KIT_ROOT="${DEMO_KIT_ROOT:-$(cd ../demo-4-platform-kit && pwd)}"
cd demo-4-greenfield-install
./scripts/greenfield-install.sh plan
```

**Pins:** `$DEMO_KIT_ROOT/scripts/supply-chain-pins.env` · `$DEMO_KIT_ROOT/docs/SUPPLY-CHAIN-PINS.md`

---

## Time budget (first install)

| Phase | Script / action | Typical duration |
|-------|-----------------|------------------|
| 0 | Prerequisites + bastion tooling | 30–60 min (one-time) |
| 1 | NetObserv + Loki (AWS) | 20–40 min |
| 2 | OpenShell + OpenClaw + LLM | 45–90 min |
| 3 | RHOAI + MLflow wire | 30–50 min |
| 4 | Grafana + OTel federation | 15–25 min |
| 5 | TrustyAI guardrails | 15–30 min |
| 6 | AAP + Gitea + ansible-mcp | 20–40 min (+ license manual) |
| 7 | NetObserv agent kit seed | 10–20 min |
| 8 | Slack Socket Mode | 15 min manual + 5 min wire |
| 9 | Event-AIOps (Grafana alert → hooks) | 10 min |
| 10 | ZTWI / SPIFFE mTLS | 20–40 min |
| 11 | RHCL OAuth (optional) | 10–20 min |
| 12 | Verify + trial `demo-a-fast` | 15 min |

**Total:** roughly **4–8 hours** first time (RHOAI and operator installs dominate). Repeat on a second cluster is faster if secrets and Slack app are reused.

---

## Phase 0 — Prerequisites

### Bastion cluster login (required — not in site-secrets)

Log in to **this cluster** before `preflight` or Phase 1. Credentials come from your lab/install output (e.g. **kubeadmin** password), not from `config prompt`.

**Guide:** [docs/CLUSTER-LOGIN.md](docs/CLUSTER-LOGIN.md)

```bash
export OCP_API="https://api.<cluster-name>.<base-domain>:6443"
oc login "$OCP_API" -u kubeadmin -p '<password>'

./scripts/cluster-login.sh check
```

`config prompt` → AWS + LLM only. `cluster-login.sh` → OpenShift API session.

### Cluster

| Requirement | Notes |
|-------------|--------|
| OpenShift **4.21+**, cluster-admin | `oc whoami` |
| **cert-manager** (for RHCL) | `oc get pods -n cert-manager` |
| Default **StorageClass** | SPIRE PVC, Grafana PVC |
| **registry.redhat.io** pull | Cluster pull-secret or namespace secrets for ZTWI SPIFFE Helper |
| OperatorHub access | NetObserv, RHOAI, RHCL, AAP, Grafana Operator, ZTWI, TrustyAI (lemonade chart is git-sourced) |

### Bastion (or laptop with cluster access + Podman)

```bash
# CLIs
oc kubectl jq curl podman git python3 aws
./scripts/install-helm3.sh install   # Helm 3 for Phase 2 (also runs in prereq/preflight)

# Kraken fault injection (Demo A)
podman pull quay.io/krkn-chaos/krkn:v5.2.7

# Both folders side-by-side under AIOps/demo/
git clone <your-repo>/AIOps.git
cd AIOps/demo/demo-4-greenfield-install
cp config/env.example config/env.local   # edit; never commit
export DEMO_KIT_ROOT="$PWD/../demo-4-platform-kit"
chmod +x scripts/*.sh "$DEMO_KIT_ROOT/scripts/"*.sh
"$DEMO_KIT_ROOT/scripts/supply-chain-check.sh"
```

### Site secrets (required)

All sensitive values live in **`config/site-secrets.local.yaml`** (gitignored) or are collected by the wizard:

```bash
./scripts/greenfield-install.sh config init      # copy template
./scripts/greenfield-install.sh config prompt    # interactive (recommended)
./scripts/greenfield-install.sh config show      # masked summary
./scripts/greenfield-install.sh config validate  # before install
```

| YAML path | Used in phase |
|-----------|----------------|
| `aws.*` | 1 — NetObserv / Loki S3 |
| `llm.*` | 2 — OpenClaw + OpenShell policy |
| `slack.*` + `site.slack_channel_id` | 8–10 |
| `aap.license_file` | unused in greenfield — apply license in AAP Gateway UI (Phase 6) |

**Modes:** `GREENFIELD_CONFIG_MODE=hybrid` (default: load YAML, prompt for gaps) · `file` (YAML only) · `prompt` (wizard if missing)

Template: [config/site-secrets.example.yaml](config/site-secrets.example.yaml) or [.json](config/site-secrets.example.json)  
Requires **python3-pyyaml** for YAML on the bastion (`dnf install python3-pyyaml`).

### Manual credentials (summary)

| Item | YAML keys |
|------|-----------|
| AWS + S3 | `aws.region`, `aws.s3_bucket`, `aws.access_key_id`, `aws.secret_access_key` |
| LLM | `llm.base_url`, `llm.model_id`, `llm.api_key`, `llm.policy_host` |
| Slack | `slack.bot_token`, `slack.app_token`, `site.slack_channel_id` |
| AAP license | Gateway UI only (Phase 6) — not stored in git or site config |

Non-secret overrides: `config/env.local` (optional)

```bash
export OPENCLAW_NS=openclaw
export LAB_CFG="$HOME/labs/openshell-on-openshift-lab/manifests/openclaw/config.yaml"
```

---

## Phase 1 — NetObserv + sample app

```bash
./scripts/greenfield-install.sh netobserv
```

Or manually:

```bash
"$DEMO_KIT_ROOT/scripts/install-netobserv-aws.sh"
"$DEMO_KIT_ROOT/scripts/deploy-netobserv-todo-app.sh"
"$DEMO_KIT_ROOT/scripts/tune-loki-ingestion.sh" apply
```

**Verify:** Console → Observe → Network Traffic shows `todo-demo` flows (3–5 min).

---

## Phase 2 — OpenShell + OpenClaw + LLM

**~45–90 min · mostly manual.** Full walkthrough: [docs/PHASE-2-OPENSHELL.md](docs/PHASE-2-OPENSHELL.md)

```bash
./scripts/greenfield-install.sh openshell    # clone lab + checklist
./scripts/phase2-openshell.sh plan           # same checklist anytime
```

### Sub-steps (run `phase2-openshell.sh check` after each)

| Step | Action |
|------|--------|
| 1 | `clone-openshell-lab.sh` + `phase2-openshell.sh write-env` |
| 2 | Agent Sandbox controller |
| 3 | OpenShell Helm chart |
| 4 | `my-llm-key` + LLM reachability probe |
| 5 | `phase2-openshell.sh providers <profile>` + edit `config.yaml` / `managed-policy.yaml` |
| 6 | Gateway token + `kubectl apply -k manifests/openclaw` |
| 7 | `./scripts/phase2-openshell.sh harden` |
| 8 | `./scripts/phase2-openshell.sh check` + `probe` + `ui-hints` |

**LLM presets:** [docs/PHASE-2-LLM-PROVIDERS.md](docs/PHASE-2-LLM-PROVIDERS.md)

```bash
source ~/labs/openshell-env.sh
./scripts/phase2-openshell.sh check
./scripts/phase2-openshell.sh probe
./scripts/phase2-openshell.sh ui
"$DEMO_KIT_ROOT/scripts/netobserv-e2e-openclaw-test.sh" ui-hints
```

**Verify:** Control UI `/new` → first message → `openclaw-agent-*` sandbox Running.

---

## Phase 3 — RHOAI + MLflow Traces wire

**~30–50 min.** Full walkthrough: [docs/PHASE-3-RHOAI.md](docs/PHASE-3-RHOAI.md)

**Prereq:** Phase 2 (`openshell ✓`)

```bash
./scripts/greenfield-install.sh rhoai
./scripts/phase3-rhoai.sh check
./scripts/phase3-rhoai.sh verify
```

RHOAI is for **MLflow audit/Traces** — OpenClaw LLM stays on external endpoint from Phase 2.

**Verify:** `mlflow-check` · OpenShift AI dashboard · experiment `openclaw-netobserv`

---

## Phase 4 — Grafana + OTel (Network AIOps dashboard)

```bash
./scripts/greenfield-install.sh grafana
# or: ./scripts/phase4-grafana.sh deploy
```

```bash
"$DEMO_KIT_ROOT/scripts/deploy-platform-merge.sh" grafana
"$DEMO_KIT_ROOT/scripts/sync-grafana-demo-metrics.sh" sync
```

**Verify:** `./scripts/phase4-grafana.sh check` · Grafana route → **Network AIOps — Gateway Diagnostics** · `federate-netobserv` up.

---

## Phase 5 — TrustyAI guardrails (Layer 0)

Upstream: [lemonade-stand-assistant](https://github.com/rh-ai-quickstart/lemonade-stand-assistant) (`fms-orchestrator/chart`, Option A external MaaS).

```bash
./scripts/greenfield-install.sh guardrails
# or: ./scripts/phase5-guardrails.sh deploy
```

```bash
"$DEMO_KIT_ROOT/scripts/deploy-platform-merge.sh" guardrails
```

**Verify:** `./scripts/phase5-guardrails.sh verify` · `trustyai-guard-check`

---

## Phase 6 — AAP + Gitea + ansible-automation MCP

**Install** (automated — stops before wire):

```bash
./scripts/greenfield-install.sh aap
# or: ./scripts/phase6-aap.sh deploy
```

**License** (manual — Gateway UI only; do not commit manifest to git):

1. `./scripts/phase6-aap.sh status` — Gateway URL + admin password hint  
2. Upload subscription in AAP Gateway UI  

**Wire** (after license is active):

```bash
./scripts/phase6-aap.sh wire
```

**Verify:** `./scripts/phase6-aap.sh verify` · `aap-check`

Guide: [docs/PHASE-6-AAP.md](docs/PHASE-6-AAP.md)

---

## Phase 7 — NetObserv agent kit (skills + dual MCP)

```bash
./scripts/greenfield-install.sh agent
# or: ./scripts/phase7-agent.sh deploy
```

Deploys `netobserv-mcp`, `openshift-mcp`, heal/capture proxies, workspace skills, and enables `ansible-automation` MCP when Phase 6 is wired.

**Verify:**

```bash
./scripts/phase7-agent.sh check
./scripts/phase7-agent.sh verify
```

Control UI: `/new` → first message → `openclaw-agent-*` sandbox Running.

Guide: [docs/PHASE-7-AGENT.md](docs/PHASE-7-AGENT.md)

---

## Phase 8 — Slack Socket Mode

**Manual:** Create Slack app + Socket Mode (see guide).

```bash
./scripts/greenfield-install.sh config prompt --full   # slack tokens + channel ID (gitignored)
./scripts/greenfield-install.sh slack
# or:
./scripts/phase8-slack.sh secrets
./scripts/phase8-slack.sh wire
```

**Verify:** `./scripts/phase8-slack.sh verify` · test `@OpenClaw` in allowlisted channel

Guide: [docs/PHASE-8-SLACK.md](docs/PHASE-8-SLACK.md)

---

## Phase 9 — Event-driven AIOps

Requires Slack (Phase 8) + Grafana metrics sync (Phase 4).

```bash
./scripts/greenfield-install.sh event
```

Or step-by-step:

```bash
./scripts/phase9-event.sh metrics
./scripts/phase9-event.sh deploy
```

**Verify:** `./scripts/phase9-event.sh verify` · `event-aiops-check`  
*(Hooks smoke test may post one Slack thread during wiring — use `SKIP_SMOKE=1` to skip.)*

Guide: [docs/PHASE-9-EVENT.md](docs/PHASE-9-EVENT.md)

---

## Phase 10 — SPIFFE / ZTWI mTLS (Layer 3)

Requires event-AIOps (Phase 9). Upgrades Grafana bridge → OpenClaw hooks to **SPIFFE mTLS**.

```bash
./scripts/greenfield-install.sh spiffe
```

Or step-by-step:

```bash
./scripts/phase10-spiffe.sh ztwi    # ZTWI operator + SPIRE operands (~20–30 min first time)
./scripts/phase10-spiffe.sh wire    # mTLS bridge + openclaw-hooks-mtls
```

**Verify:** `./scripts/phase10-spiffe.sh verify` · `spiffe-check`  
**After reboot:** `./scripts/phase10-spiffe.sh repair`

Guide: [docs/PHASE-10-SPIFFE.md](docs/PHASE-10-SPIFFE.md)

---

## Phase 11 — RHCL OAuth for OpenClaw (optional)

Enterprise OAuth front door for Control UI. **Optional** — legacy Route + gateway token still works.

```bash
./scripts/greenfield-install.sh rhcl
```

Or step-by-step:

```bash
./scripts/phase11-rhcl.sh operator   # RHCL + Kuadrant (~10 min)
./scripts/phase11-rhcl.sh ingress    # gateway + OpenShift OAuth
```

**Verify:** `./scripts/phase11-rhcl.sh verify` · incognito login on `https://openclaw-rhcl.apps.<ingress>/`

Skip: `SKIP_RHCL=1` · Operator only: `SKIP_RHCL_INGRESS=1`

Guide: [docs/PHASE-11-RHCL.md](docs/PHASE-11-RHCL.md)

---

## Phase 12 — Full verify + Demo A trial

```bash
./scripts/greenfield-install.sh verify
```

```bash
"$DEMO_KIT_ROOT/scripts/demo-cluster-preflight.sh" check
"$DEMO_KIT_ROOT/scripts/netobserv-e2e-openclaw-test.sh" aap-check
"$DEMO_KIT_ROOT/scripts/netobserv-e2e-openclaw-test.sh" mlflow-check
"$DEMO_KIT_ROOT/scripts/netobserv-e2e-openclaw-test.sh" slack-check
"$DEMO_KIT_ROOT/scripts/netobserv-e2e-openclaw-test.sh" spiffe-check
"$DEMO_KIT_ROOT/scripts/netobserv-e2e-openclaw-test.sh" trustyai-guard-check

# Off-stage trial (~3 min)
"$DEMO_KIT_ROOT/scripts/netobserv-e2e-openclaw-test.sh" demo-a-fast
"$DEMO_KIT_ROOT/scripts/netobserv-krkn-fault.sh" restore
```

**Demo A trial:** `"$DEMO_KIT_ROOT/scripts/netobserv-e2e-openclaw-test.sh" demo-a-fast` · presenter prompts: `"$DEMO_KIT_ROOT/scripts/netobserv-e2e-openclaw-test.sh" ui-hints`

---

## Orchestrator quick reference

```bash
./scripts/greenfield-install.sh plan              # list phases
./scripts/greenfield-install.sh all               # run all phases (pauses for manual steps)
./scripts/greenfield-install.sh netobserv       # single phase
./scripts/greenfield-install.sh status            # aggregate status checks

# Skip phases already done on a partial cluster
SKIP_NETOBSERV=1 SKIP_OPENCLAW=1 ./scripts/greenfield-install.sh all
```

| Env | Default | Purpose |
|-----|---------|---------|
| `SLACK_CHANNEL_ID` | *(required for slack/event/spiffe)* | Slack + bridge + SPIFFE manifests |
| `SKIP_NETOBSERV` | `0` | Skip Phase 1 |
| `SKIP_OPENCLAW` | `0` | Skip Phase 2 prompts (OpenShell already up) |
| `SKIP_RHCL` | `0` | Skip Phase 11 |
| `SKIP_RHCL_INGRESS` | `0` | RHCL operator only, no OAuth ingress |
| `AUTO_CONTINUE` | `0` | `1` = no Enter prompts between phases |
| `DEMO_KIT_ROOT` | sibling `demo-4-platform-kit` | Platform kit scripts |
| `LAB_CFG` | `~/labs/.../config.yaml` | OpenClaw lab config path |

---

## After cluster reboot

See [CLUSTER-WAKE.md](CLUSTER-WAKE.md).

---

## Troubleshooting

| Symptom | Action |
|---------|--------|
| FlowCollector WebConsoleError | `$DEMO_KIT_ROOT/scripts/fix-netobserv-operator-rbac.sh apply` |
| Heal confirm gate fails | `$DEMO_KIT_ROOT/scripts/wire-openclaw-aap.sh bootstrap` |
| SPIFFE lost after seed | `./scripts/phase10-spiffe.sh wire` |
| Grafana NoData | `$DEMO_KIT_ROOT/scripts/sync-grafana-demo-metrics.sh sync` |

See demo kit README and `docs/*-PRESENTER-GUIDE.md`.
