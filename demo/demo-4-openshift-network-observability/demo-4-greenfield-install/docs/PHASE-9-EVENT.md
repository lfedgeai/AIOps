# Phase 9 — Event-driven AIOps

**Time:** ~10 minutes  
**Orchestrator:** `./scripts/greenfield-install.sh event`  
**Helper:** `./scripts/phase9-event.sh` (`plan` · `metrics` · `hooks` · `alerts` · `wire` · `deploy` · `check` · `status` · `verify`)

When NetObserv metrics show elevated **todo-demo RTT**, Grafana fires an alert that triggers OpenClaw to **start an investigation in Slack** — no human `@OpenClaw` required.

**Investigate is automatic. Heal stays human-confirmed.**

---

## Before you start

| Requirement | Check |
|-------------|--------|
| Phase 8 complete | `slack ✓` · Socket Mode connected |
| Phase 4 complete | Grafana route + `openclaw-otel-prometheus` |
| Site config | `site.slack_channel_id` (same channel as Phase 8) |

```bash
./scripts/phase9-event.sh plan
./scripts/greenfield-install.sh phases   # slack + grafana should be done
```

---

## Architecture

```text
Kraken / real fault
        │
        ▼
NetObserv metrics (federated → openclaw-otel-prometheus)
        │
        ▼
Grafana alert: todo-demo avg RTT > 350ms for 2m
        │
        ▼
netobserv-grafana-bridge (normalizes payload)
        │
        ▼ POST /hooks/agent + Bearer token
OpenClaw isolated investigate turn → Slack channel
```

Full guide: `$DEMO_KIT_ROOT/docs/EVENT-DRIVEN-AIOPS-GUIDE.md`

---

## Deploy

```bash
./scripts/phase9-event.sh deploy
# or via orchestrator:
./scripts/greenfield-install.sh event
```

This runs:

| Step | What |
|------|------|
| `sync-grafana-demo-metrics.sh sync` | Federate NetObserv → demo Prometheus (single Grafana datasource) |
| `wire-openclaw-hooks.sh` | Secret `openclaw-hooks-token`, hooks config, `netobserv-grafana-bridge` |
| `wire-grafana-openclaw-alerts.sh` | Grafana contact point + RTT alert rule |

**Note:** The hooks smoke test posts a synthetic webhook — you may see one Slack thread during wiring. Skip with `SKIP_SMOKE=1 ./scripts/phase9-event.sh hooks`.

---

## Verify

```bash
./scripts/phase9-event.sh verify
```

**Live demo** (after wiring):

```bash
SLACK_CHANNEL_ID=$SLACK_CHANNEL_ID $DEMO_KIT_ROOT/scripts/prepare-event-demo.sh
# or legacy inject:
$KIT/netobserv-e2e-openclaw-test.sh demo-a-fast
```

Wait ~2–3 minutes after RTT rises → Slack auto-investigate thread.

**Next phase:** `./scripts/greenfield-install.sh spiffe`

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `openclaw-slack-tokens missing` | Complete Phase 8 first |
| `Grafana route missing` | `./scripts/phase4-grafana.sh deploy` |
| `openclaw-hooks-token missing` | `./scripts/phase9-event.sh hooks` |
| Bridge smoke test fails | Wait for OpenClaw rollout; re-run hooks with `SKIP_SMOKE=0` |
| Alert stays NoData | `./scripts/phase9-event.sh metrics` · confirm todo-demo traffic |
| No auto-investigate in Slack | Check Grafana alert state · `event-aiops-check` · channel ID on bridge |
| SPIFFE bridge 2/2 not ready | Phase 10 — `install-ztwi-spire.sh repair` |

---

## Quick reference

```bash
./scripts/phase9-event.sh plan
./scripts/phase9-event.sh metrics
./scripts/phase9-event.sh hooks
./scripts/phase9-event.sh alerts
./scripts/phase9-event.sh check
$KIT/netobserv-e2e-openclaw-test.sh event-aiops-check
$KIT/prepare-event-demo.sh
```
