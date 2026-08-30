# Phase 7 — NetObserv agent kit seed

**Time:** ~10–20 minutes  
**Orchestrator:** `./scripts/greenfield-install.sh agent`  
**Helper:** `./scripts/phase7-agent.sh` (`plan` · `deploy` · `check` · `status` · `verify`)

Seeds the **investigate → evidence → heal** agent workflow: NetObserv MCP, OpenShift MCP, heal/capture proxies, workspace skills, and `AGENTS.md`.

---

## Before you start

| Requirement | Check |
|-------------|--------|
| Phase 2 complete | `openshell ✓` · Agent Sandbox Running |
| Phase 6 recommended | `ansible-mcp` + `openclaw-aap-launcher` (governed heal) |
| Phase 3 recommended | RHOAI MLflow for MCP audit traces |
| OpenShell lab | `~/labs/openshell-on-openshift-lab` @ pinned commit |

```bash
./scripts/phase7-agent.sh plan
```

---

## What gets deployed

| Component | Namespace | Purpose |
|-----------|-----------|---------|
| `netobserv-mcp` | `openclaw` | Flow capture, evidence analysis tools |
| `openshift-mcp` | `openclaw` | Read-only cluster context (Helm) |
| `ansible-mcp` | `openclaw` | Launch AAP job templates (Phase 6) |
| `netobserv-heal-proxy` | `openclaw` | HTTP heal fallback |
| `netobserv-capture-proxy` | `openclaw` | In-cluster flow capture sidecar path |
| `netobserv-sandbox-flatten` | `openclaw` | OpenShell upload path safety net |
| Skills + `AGENTS.md` | OpenClaw workspace | `netobserv-investigate`, `evidence`, `heal` |

Script: `$DEMO_KIT_ROOT/scripts/seed-openclaw-netobserv-skills.sh`

---

## Automated install

```bash
cd ~/AIOps/demo/demo-4-greenfield-install
./scripts/greenfield-install.sh agent
```

Or:

```bash
./scripts/phase7-agent.sh deploy
./scripts/phase7-agent.sh check
./scripts/phase7-agent.sh verify
```

**Control UI proof (manual):**

1. Open OpenClaw Route → `/new`
2. Send first message — wait for `openclaw-agent-*` sandbox pod
3. Optional: `$DEMO_KIT_ROOT/scripts/openshell-sandbox-proof.sh wait`

**Next phase:** `./scripts/greenfield-install.sh slack`

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `mcp doctor` timeout | Wait for MCP rollouts; re-run `phase7-agent.sh deploy` |
| `ansible-automation` missing | Complete Phase 6 wire first |
| Sandbox ImagePullBackOff | `$DEMO_KIT_ROOT/scripts/fix-openshell-sandbox-image.sh` |
| Skills missing after recycle | Re-run `./scripts/phase7-agent.sh deploy` |
| No sandbox on `/new` alone | Expected — send a **message** to spawn agent pod |

---

## Quick reference

```bash
./scripts/phase7-agent.sh plan
./scripts/phase7-agent.sh status
./scripts/phase7-agent.sh check
$DEMO_KIT_ROOT/scripts/netobserv-e2e-openclaw-test.sh status
$DEMO_KIT_ROOT/scripts/openshell-sandbox-proof.sh wait
```
