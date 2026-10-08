# Platform Kit

Everything the greenfield installer deploys and everything you run on demo day. Install first with [`demo-4-greenfield-install/`](../demo-4-greenfield-install/README.md); come back here to run the scenarios and prove the controls.

For what the demo is and why, see the [top-level README](../README.md).

---

## What's inside

The kit is used in four stages. The installer drives the first two; you drive the last two.

| Stage | Where | What it is |
|---|---|---|
| **1 · Install** | `manifests/` · `scripts/install-*.sh` · `scripts/deploy-*.sh` | Operators and platform pieces: NetObserv and Loki, OpenShift AI, Grafana and OTel, TrustyAI, AAP and Gitea, ZTWI/SPIRE, RHCL, the todo app, and the upstream openshift MCP server |
| **2 · Wire** | `openclaw-skills/` · `scripts/wire-*.sh` · `scripts/seed-*.sh` | Agent skills, the netobserv and ansible MCP servers, and the wiring between agent, Slack, alerts, MLflow and the guard gateway |
| **3 · Run** | `scripts/netobserv-e2e-openclaw-test.sh` · `scripts/netobserv-*-fault.sh` · `ansible/playbooks/` | Fault injection, demo scenarios, and the two remediation playbooks AAP runs |
| **4 · Prove** | `scripts/openshell-sandbox-proof.sh` · the `*-check` subcommands | Scripts that demonstrate each control rather than describe it |

The agent's skills are in `openclaw-skills/`: `netobserv-investigate`, `netobserv-evidence`, `netobserv-live-flows`, `netobserv-heal` and `ansible-automation`.

The NetObserv install and todo-app scripts here are the installer's versions. The standalone lab in [`../demo-4-netobserv-basics/`](../demo-4-netobserv-basics/) has its own copies, which differ.

---

## Before a Demo

From the bastion, after the install:

```bash
export DEMO_KIT_ROOT="$PWD"                     # run from this folder
./scripts/demo-cluster-preflight.sh             # read-only check of agent, guard, event and Slack paths
./scripts/demo-cluster-preflight.sh heal        # fix common drift, then re-check
./scripts/pre-demo-sanity.sh
./scripts/netobserv-e2e-openclaw-test.sh status
```

Slack scripts read the channel from `SLACK_CHANNEL_ID`, or from `site.slack_channel_id` in the greenfield site config.

---

## Scenarios

| Scenario | What happens | Run |
|---|---|---|
| **A — Application latency** | Krkn injects latency on todo → PostgreSQL; the agent triages, captures flows, diagnoses, and heals after you confirm | `netobserv-e2e-openclaw-test.sh demo-a-fast` *(cold start: `demo-a`, ~20 min)* |
| **B — Microsegmentation** | A misconfigured NetworkPolicy breaks DB connectivity; the agent diagnoses and restores it after you confirm | `netobserv-e2e-openclaw-test.sh policy-all` |
| **Event-driven A** | The Grafana RTT alert opens the Slack thread and starts the investigation — no one asks | `prepare-event-demo.sh` |

The agent heals on a separate turn from the investigation; after a long investigation, start the heal with `/new`. `netobserv-e2e-openclaw-test.sh ui-hints` prints the Control UI prompts and gateway access commands.

**Reset** after any scenario:

```bash
./scripts/netobserv-e2e-openclaw-test.sh restore          # stops Krkn, removes load, restores policy, cleans captures
./scripts/netobserv-e2e-openclaw-test.sh policy-restore   # Scenario B only
```

---

## Proving the Controls

| Claim | Proof |
|---|---|
| The sandbox has no route to the API server | `openshell-sandbox-proof.sh prove-6443` |
| The tool surface is the MCP allowlist and nothing else | `openshell-sandbox-proof.sh prove-mcp` |
| The sandbox policy is a reviewable ConfigMap | `openshell-sandbox-proof.sh prove-policy` |
| All three in one run | `openshell-sandbox-proof.sh prove` |
| Destructive prompts are blocked before the model | `netobserv-e2e-openclaw-test.sh trustyai-guard-check` |
| Remediation runs through AAP | `netobserv-e2e-openclaw-test.sh aap-check` |
| Tool calls reach MLflow | `netobserv-e2e-openclaw-test.sh mlflow-check` |
| The event path runs on SPIFFE mTLS | `netobserv-e2e-openclaw-test.sh spiffe-check` ⚠️ |
| The alert → bridge → Slack path works | `netobserv-e2e-openclaw-test.sh event-aiops-check` ⚠️ |

⚠️ These two post a synthetic alert, which opens a real Slack thread. Don't run them during a live event-driven demo.

`openshell-sandbox-proof.sh hints` prints presenter narrative and negative-test prompts for the Control UI.

---

## Day-2

| When | Run |
|---|---|
| The cluster was stopped or idle for a long time | `post-cluster-resume.sh` (`status` for read-only) — then see [`CLUSTER-WAKE.md`](../demo-4-greenfield-install/CLUSTER-WAKE.md) |
| Restoring SPIFFE mTLS and the event path after a restart, without opening Slack threads | `post-cluster-spiffe-resume.sh` |
| Resuming specifically for an event-driven demo | `cold-start-event-demo.sh`, then `prepare-event-demo.sh` |
| Control UI sessions are stuck, or MCP reports "Session not found" | `clear-openclaw-sessions.sh` |
| Grafana network panels are empty | `sync-grafana-demo-metrics.sh` |

`post-cluster-resume.sh` runs the SPIFFE and event checks, so it opens Slack threads too. Right before a demo, use `post-cluster-spiffe-resume.sh` or `cold-start-event-demo.sh` instead.

---

## Supply Chain

Every image and chart is pinned in [`scripts/supply-chain-pins.env`](scripts/supply-chain-pins.env). `supply-chain-check.sh` verifies the pins; `verify-image-pulls.sh` confirms the images are pullable before you install.

One dependency lives outside this repository: Phase 2 clones the pinned OpenShell lab to `~/labs/openshell-on-openshift-lab` using `clone-openshell-lab.sh`.
