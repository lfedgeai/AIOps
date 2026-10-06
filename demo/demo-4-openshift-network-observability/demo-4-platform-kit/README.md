# Demo 4 — Platform kit

Install assets for the greenfield orchestrator (`demo-4-greenfield-install`).

**Contains:** install scripts, OpenShift manifests, OpenClaw skills/MCP, Ansible playbooks for AAP, and day-2 helpers (`demo-cluster-preflight.sh`, `netobserv-e2e-openclaw-test.sh`, …).

```text
demo-4-greenfield-install/    orchestrator + site secrets
demo-4-platform-kit/          this folder (DEMO_KIT_ROOT)
```

## Use

```bash
export DEMO_KIT_ROOT="$(cd ../demo-4-platform-kit && pwd)"
cd ../demo-4-greenfield-install
./scripts/greenfield-install.sh preflight
./scripts/greenfield-install.sh netobserv
```

**Slack:** scripts read `SLACK_CHANNEL_ID` from env or `site.slack_channel_id` in greenfield `site-secrets.local.yaml` (`scripts/resolve-slack-channel.sh`).

**External dependency:** Phase 2 clones the pinned OpenShell lab to `~/labs/openshell-on-openshift-lab` (`clone-openshell-lab.sh` + `supply-chain-pins.env`).

Full layout and sync: [../demo-4-greenfield-install/docs/PLATFORM-KIT.md](../demo-4-greenfield-install/docs/PLATFORM-KIT.md)
