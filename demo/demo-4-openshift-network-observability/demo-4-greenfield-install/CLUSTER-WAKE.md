# Cluster wake — greenfield site

Use when the cluster was shut down during install, or before a customer demo on a greenfield site.

---

## 1. Bastion + login

```bash
cd ~/AIOps/demo/demo-4-greenfield-install
source config/env.local    # optional: BASTION_*, SLACK_CHANNEL_ID
export DEMO_KIT_ROOT="${DEMO_KIT_ROOT:-$HOME/AIOps/demo/demo-4-platform-kit}"

oc login …
./scripts/cluster-login.sh check
```

`SLACK_CHANNEL_ID` is auto-loaded from `site-secrets.local.yaml` when scripts source `resolve-slack-channel.sh`.

---

## 2. Where did you leave off?

```bash
./scripts/greenfield-install.sh phases     # ✓/pending per phase
./scripts/greenfield-install.sh status     # component-level detail
```

Resume the **next** incomplete phase:

```bash
./scripts/greenfield-install.sh plan
./scripts/greenfield-install.sh <phase>    # e.g. rhoai, agent, spiffe
```

Or heal drift on a **finished** greenfield install:

```bash
"$DEMO_KIT_ROOT/scripts/demo-cluster-preflight.sh" heal
ENABLE_ANSIBLE_MCP=1 "$DEMO_KIT_ROOT/scripts/seed-openclaw-netobserv-skills.sh"   # if OpenClaw recycled
"$DEMO_KIT_ROOT/scripts/netobserv-e2e-openclaw-test.sh" aap-check
```

---

## 3. Demo A day-of (greenfield cluster ready)

```bash
"$DEMO_KIT_ROOT/scripts/demo-cluster-preflight.sh" check
"$DEMO_KIT_ROOT/scripts/netobserv-e2e-openclaw-test.sh" demo-a-fast
```

Full SPIFFE/event wake (optional — no synthetic Slack threads):

```bash
"$DEMO_KIT_ROOT/scripts/post-cluster-spiffe-resume.sh"
```

---

## 4. Scenario A heal reminder

| Step | What |
|------|------|
| Agent confirms heal | AAP job template **`netobserv-heal-db-path`** (latency / Kraken scenario) |
| Bastion cleanup | **`"$DEMO_KIT_ROOT/scripts/netobserv-krkn-fault.sh" restore`** — required after agent heal |
| Wrong template | Do **not** use `netobserv-restore-policy` for Scenario A (that is Scenario B / policy fault) |

Presenter flow: preflight check → `demo-a-fast` → Control UI investigate → confirm heal → bastion `restore`.
