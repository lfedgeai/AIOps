# Phase 10 — ZTWI / SPIFFE mTLS

**Time:** ~20–40 min first install (operator + SPIRE operands + wire)  
**Orchestrator:** `./scripts/greenfield-install.sh spiffe`  
**Helper:** `./scripts/phase10-spiffe.sh` (`plan` · `ztwi` · `wire` · `deploy` · `repair` · `check` · `status` · `verify`)

Replaces the shared **Bearer token** on the Grafana → OpenClaw event path with **SPIFFE workload identity + mTLS** (Red Hat **Zero Trust Workload Identity Manager**).

**Security story (Layer 3):** complements OpenShell sandbox (Layer 1) and scoped MCP (Layer 2).

---

## Before you start

| Requirement | Check |
|-------------|--------|
| Phase 9 complete | `event ✓` · hooks + `netobserv-grafana-bridge` |
| Site config | `site.slack_channel_id` (same as Phase 8/9) |
| OperatorHub | **Zero Trust Workload Identity Manager** available on cluster |
| Pull secret | Cluster can pull `registry.redhat.io` (SPIFFE Helper image) |

```bash
./scripts/phase10-spiffe.sh plan
./scripts/greenfield-install.sh phases   # event should be done
```

---

## Architecture

```text
Grafana alert webhook (HTTP, cluster-internal)
        │
        ▼
netobserv-grafana-bridge          SPIFFE client SVID (CSI + Helper)
        │
        ▼ mTLS (verified client cert)
openclaw-hooks-mtls proxy         SPIFFE server SVID; holds OPENCLAW_HOOKS_TOKEN
        │
        ▼ Bearer (localhost to OpenClaw svc)
OpenClaw /hooks/agent → Slack auto-investigate
```

**Before SPIFFE:** bridge pod held `OPENCLAW_HOOKS_TOKEN`.  
**After SPIFFE:** bridge proves identity via mTLS; Bearer stays on `openclaw-hooks-mtls` only.

Full guide: `$DEMO_KIT_ROOT/docs/SPIFFE-WORKLOAD-IDENTITY-GUIDE.md`

---

## Deploy

```bash
./scripts/phase10-spiffe.sh deploy
# or via orchestrator:
./scripts/greenfield-install.sh spiffe
```

This runs:

| Step | Script | What |
|------|--------|------|
| 1 | `install-ztwi-spire.sh all` | ZTWI operator + SPIRE Server/Agent/CSI CRs |
| 2 | `wire-openclaw-spiffe.sh all` | ClusterSPIFFEID + `openclaw-hooks-mtls` + bridge upgrade |

Step-by-step:

```bash
./scripts/phase10-spiffe.sh ztwi    # operator + operands (~20–30 min first time)
./scripts/phase10-spiffe.sh wire    # mTLS workloads
```

**Verify:**

```bash
./scripts/phase10-spiffe.sh verify
```

**Note:** `spiffe-check` and wire smoke tests post a synthetic webhook — you may see one Slack investigate thread. Skip before customer demos if unwanted.

---

## After cluster reboot

SPIFFE mTLS does **not** survive a clean restart. Re-run:

```bash
./scripts/phase10-spiffe.sh repair
./scripts/phase10-spiffe.sh wire    # if bridge reverted to Bearer
./scripts/phase10-spiffe.sh verify
```

Or: `$DEMO_KIT_ROOT/scripts/post-cluster-spiffe-resume.sh`

---

## Rollback (optional)

Revert to Bearer-token bridge (non-SPIFFE event path):

```bash
./scripts/phase10-spiffe.sh rollback
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| ZTWI not in OperatorHub | Confirm OCP version / entitlement; install manually from console |
| `ZeroTrustWorkloadIdentityManager` not Ready | `./scripts/phase10-spiffe.sh repair` · check `oc get pods -n zero-trust-workload-identity-manager` |
| SPIFFE Helper `ImagePullBackOff` | `$DEMO_KIT_ROOT/scripts/fix-spiffe-pull-secret.sh` · cluster `registry.redhat.io` pull secret |
| Bridge not 2/2 ready | `repair` · check spiffe-helper sidecar logs |
| `openclaw-hooks-token missing` | Complete Phase 9 first |
| Event path broken after SPIFFE | `phase10-spiffe.sh wire` · `spiffe-check` |
| Lost after agent seed | Re-run `phase10-spiffe.sh wire` (seed does not remove SPIFFE) |

---

## Quick reference

```bash
./scripts/phase10-spiffe.sh plan
./scripts/phase10-spiffe.sh ztwi
./scripts/phase10-spiffe.sh wire
./scripts/phase10-spiffe.sh check
./scripts/phase10-spiffe.sh repair
$KIT/netobserv-e2e-openclaw-test.sh spiffe-check
```

**Next phase:** `./scripts/greenfield-install.sh verify` (Phase 12)
