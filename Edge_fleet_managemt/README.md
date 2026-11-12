# Edge Fleet Management — FlightCtl + agentic-AIOps runtime harness

Independent integration of the **agentic-AIOps runtime harness** into the
**FlightCtl** edge-fleet-management UI. Everything here is self-contained and
deliberately separate from the rest of this repository.

## Layout

```
Edge_fleet_managemt/
├── flightctl/        # FlightCtl backend (vendored clone, pinned below)
├── flightctl-ui/     # FlightCtl web UI (vendored clone, pinned below)
│   └── apps/standalone/src/app/components/AIOpsRuntime/AIOpsRuntimePage.tsx
│                      # ← our addition: /aiops page embedding the harness
└── runtime_harness/  # INDEPENDENT COPY of the agentic-AIOps runtime harness
                       #   (the copy at the repository root is untouched and
                       #    remains the development original)
```

## Pinned upstream commits (vendored, nested .git removed)

| Repo | Commit | Date |
|---|---|---|
| `github.com/flightctl/flightctl` | `f99870e` — EDM-6476: Add enrollment CSR revocation API (#3817) | cloned 2026-11-12 (shallow) |
| `github.com/flightctl/flightctl-ui` | `c088274` — EDM-4784: Restore confirmation when user leaves dirty unsaved form (#853) | cloned 2026-11-12 (shallow) |

To refresh a pin: delete the directory, `git clone --depth 1` again, remove
`.git`, and re-apply the integration diff (the changes are limited to
`apps/standalone/src/app/{routes.tsx,components/AIOpsRuntime/,index.tsx}`,
`apps/standalone/webpack.config.ts`, and `CONFIGURATION.md` — see the
commit that added this directory for the exact set).

## The integration

The FlightCtl UI (standalone app) gains an **"AIOps Runtime"** nav entry →
`/aiops` page that embeds the runtime harness UI in an iframe. The harness
remains a fully separate service — its own process, FastAPI/UI, ClickHouse,
model wiring, and remediation modes (HIL / Governed / Direct /
Image-rebuild via bootc + FlightCtl rollout, which is the bridge between
the two systems: the harness's `image_rebuild` mode patches FlightCtl
fleet specs).

The harness URL is set at build time:

```bash
cd flightctl-ui
npm ci
AIOPS_RUNTIME_URL=http://localhost:8085 npm run build --workspace apps/standalone
# dev: AIOPS_RUNTIME_URL defaults to http://localhost:8085
```

## Running

```bash
# 1. Runtime harness (this copy, own install — independent of the repo-root copy)
cd runtime_harness
pip install -e ".[mcp,k8s]"
export GRANITE_API_BASE=... GRANITE_API_KEY=... GRANITE_MODEL=...
export GLM_API_BASE=...     GLM_API_KEY=...     GLM_MODEL=...
aiops-runtime serve --host 0.0.0.0 --port 8085

# 2. FlightCtl UI (dev server proxies to a FlightCtl backend)
cd ../flightctl-ui
npm ci && npm run start --workspace apps/standalone

# 3. FlightCtl backend (see flightctl/README.md — kind deploy or podman/systemd)
cd ../flightctl
make build   # or the quick-start path from its README
```

Verified: `tsc --noEmit` clean, ESLint clean, production webpack build
succeeds with the `/aiops` route and the harness-URL define in the bundle.
