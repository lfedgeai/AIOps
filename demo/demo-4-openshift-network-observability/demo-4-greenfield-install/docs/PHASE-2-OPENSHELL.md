# Phase 2 — OpenShell + OpenClaw + private LLM

**Time:** ~45–90 minutes  
**Automation:** `./scripts/greenfield-install.sh openshell` (deploy → harden → check → probe)  
**Helper:** `./scripts/phase2-openshell.sh` (`deploy` · `check` · `harden` · `probe` · `providers` · `ui`)

## Automated install (recommended)

```bash
cd ~/AIOps/demo/demo-4-greenfield-install
source config/env.local
./scripts/greenfield-install.sh openshell
```

This runs: lab checkout → site-secrets apply → Agent Sandbox → OpenShell Helm → OpenClaw deploy → kit hardening → `check` → LLM `probe`.

Skip the LLM probe: `SKIP_PHASE2_PROBE=1 ./scripts/greenfield-install.sh openshell`

Deploy only (no harden/probe): `./scripts/phase2-openshell.sh deploy`

---

## Manual steps (reference)

The sections below document what `deploy` automates. Use them for troubleshooting or partial re-runs.

---

## Before you start

| Requirement | Check |
|-------------|-------|
| Phase 1 complete (NetObserv + todo) | `./scripts/greenfield-install.sh phases` → `netobserv ✓` |
| `oc` / `kubectl` cluster-admin | `oc whoami` |
| Helm 3 on bastion | `helm version` — auto: `./scripts/install-helm3.sh install` |
| Private OpenAI-compatible LLM endpoint | URL, model id, API key |
| Outbound HTTPS from cluster to LLM host | bastion `curl` + in-cluster probe (step 4) |

**Security:** OpenShell eval mode and OpenClaw `dangerouslyDisableDeviceAuth` are **lab-only**. See upstream lab security notice.

Print the checklist anytime:

```bash
./scripts/phase2-openshell.sh plan
```

---

## LLM worksheet (fill in before editing config)

| Setting | Your value | Example |
|---------|------------|---------|
| `LLM_BASE_URL` | | `https://litemaas.example.com/v1` |
| `LLM_MODEL` | | `Qwen3.6-35B-A3B` |
| Primary model id | `openai/<model>` | `openai/Qwen3.6-35B-A3B` |
| `contextWindow` | | `131072` (match backend) |
| `maxTokens` | | `12288` |
| LLM hostname (no scheme) | | `litemaas.example.com` |
| Secret name | `my-llm-key` / key `api-key` | fixed by lab |

---

## Step 1 — Pins + lab checkout

The upstream lab ([openshell-on-openshift-lab](https://github.com/redhat-et/openshell-on-openshift-lab)) pins OpenClaw **2026.6.11** at digest `sha256:a91dbc…`. That digest was **removed from Quay** — do **not** chase `latest`, `hummingbird-*`, or `2026.08.*` tags (they ship Node 22.23.1 + unsafe SQLite and stripped init tools).

**One-time — mirror the working image to your Quay org** (from any cluster where OpenClaw is still running the pinned digest):

```bash
cd ~/AIOps/demo/demo-4-greenfield-install
export QUAY_ORG=your-org
export QUAY_TAG=openclaw-v2026.6.11
./scripts/mirror-openclaw-image-to-quay.sh
```

Then on your **greenfield** bastion, set in `config/env.local`:

```bash
export OPENCLAW_GATEWAY_IMAGE=quay.io/your-org/openclaw-openshift:openclaw-v2026.6.11
export OPENCLAW_SANDBOX_IMAGE=quay.io/your-org/openclaw-openshift:openclaw-v2026.6.11
```

**Lab checkout + pins** (both bastions):

```bash
export DEMO_KIT_ROOT="$HOME/AIOps/demo/demo-4-platform-kit"
cd "$HOME/AIOps/demo/demo-4-greenfield-install"

"$DEMO_KIT_ROOT/scripts/clone-openshell-lab.sh"
"$DEMO_KIT_ROOT/scripts/supply-chain-check.sh"
./scripts/phase2-openshell.sh write-env
source ~/labs/openshell-env.sh
```

Pinned versions live in `$DEMO_KIT_ROOT/scripts/supply-chain-pins.env` (chart **0.0.82**, lab commit **4a325ae…**, Agent Sandbox **v0.5.1**). Do **not** `git clone` the lab without the pin script.

**Gate:** `"$DEMO_KIT_ROOT/scripts/clone-openshell-lab.sh" status`

---

## Step 2 — Agent Sandbox controller

```bash
source ~/labs/openshell-env.sh

kubectl apply -f "${AGENT_SANDBOX_MANIFEST}"
kubectl -n agent-sandbox-system rollout status deployment/agent-sandbox-controller --timeout=300s
```

**Gate:** `./scripts/phase2-openshell.sh check` (Agent Sandbox line ✓)

---

## Step 3 — OpenShell Helm chart

```bash
source ~/labs/openshell-env.sh
cd ~/labs/openshell-on-openshift-lab

# First install on cluster: follow lab README Level 0 (namespace, SCC, JWT secret) if needed.

helm upgrade --install openshell \
  oci://ghcr.io/nvidia/openshell/helm-chart \
  --version "${OPENSHELL_VERSION}" \
  --namespace "${OPENSHELL_NAMESPACE}" \
  --create-namespace \
  --values manifests/openshell/values.yaml

kubectl -n "${OPENSHELL_NAMESPACE}" rollout status statefulset/openshell --timeout=600s
```

**Optional — CLI smoke test** (second terminal):

```bash
kubectl -n openshell port-forward svc/openshell 8080:8080
openshell gateway add http://127.0.0.1:8080 --local --name openshift-lab
openshell gateway select openshift-lab
openshell status
```

**Gate:** `./scripts/phase2-openshell.sh check` (OpenShell StatefulSet ✓)

---

## Step 4 — LLM endpoint + Secret

### 4a — Prove reachability

```bash
export LLM_BASE_URL="https://YOUR-LLM-HOST/v1"
export LLM_MODEL="YourModelId"
read -s LLM_API_KEY && echo

# From bastion
curl -sS -H "Authorization: Bearer ${LLM_API_KEY}" \
  "${LLM_BASE_URL}/models" | head

# From cluster (no key in shell history after unset)
oc run llm-probe --rm -i --restart=Never \
  --image=registry.access.redhat.com/ubi9/ubi-minimal:latest -- \
  curl -sS -o /dev/null -w 'http=%{http_code}\n' --max-time 20 \
  -H "Authorization: Bearer ${LLM_API_KEY}" \
  "${LLM_BASE_URL}/models"
```

### 4b — Create `my-llm-key`

```bash
export OPENCLAW_NAMESPACE="${OPENCLAW_NAMESPACE:-openclaw}"
kubectl create namespace "${OPENCLAW_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

printf '%s' "${LLM_API_KEY}" | kubectl -n "${OPENCLAW_NAMESPACE}" \
  create secret generic my-llm-key \
  --from-file=api-key=/dev/stdin \
  --dry-run=client -o yaml | kubectl apply -f -

unset LLM_API_KEY
kubectl -n "${OPENCLAW_NAMESPACE}" get secret my-llm-key -o name
```

OpenClaw mounts this as `OPENAI_API_KEY` (OpenAI SDK name; backend can be any compatible gateway).

**Gate:** `./scripts/phase2-openshell.sh check` (my-llm-key ✓)

---

## Step 5 — OpenClaw config + sandbox network policy

**Provider presets:** [docs/PHASE-2-LLM-PROVIDERS.md](docs/PHASE-2-LLM-PROVIDERS.md)

```bash
./scripts/phase2-openshell.sh providers          # list: litemaas, vllm, maas-16k, openai
./scripts/phase2-openshell.sh providers litemaas # copy-paste settings for your site
```

Edit **`~/labs/openshell-on-openshift-lab/manifests/openclaw/config.yaml`** and **`policies/managed-policy.yaml`** (unless using public OpenAI).

Minimum changes:

1. **`models.providers.openai.baseUrl`** → your `LLM_BASE_URL` (include `/v1` if required)
2. **`agents.defaults.model.primary`** → `openai/<LLM_MODEL>`
3. **`contextWindow` / `maxTokens`** → match backend limits (undershoot wastes headroom; overshoot → HTTP 400)
4. Model entry under `models.providers.openai.models[]` with matching `id`

Example provider block (adjust host/model):

```json
"models": {
  "providers": {
    "openai": {
      "baseUrl": "https://YOUR-LLM-HOST/v1",
      "apiKey": "env:OPENAI_API_KEY",
      "api": "openai-completions",
      "models": [{
        "id": "YourModelId",
        "name": "Your model",
        "contextWindow": 131072,
        "maxTokens": 12288
      }]
    }
  }
}
```

### Network policy (required for private LLM)

Stock `manifests/openclaw/policies/managed-policy.yaml` only allows `api.openai.com:443`. Add your host under `network_policies.openai_api.endpoints` — **each entry needs host, port, protocol, access, enforcement** (duplicate YAML keys break parsing):

```yaml
network_policies:
  openai_api:
    endpoints:
      - host: YOUR-LLM-HOST.example.com
        port: 443
        protocol: rest
        access: full
        enforcement: enforce
    binaries:
      - { path: /usr/bin/node }
      - { path: /usr/local/bin/node }
      - { path: "/app/**" }
```

Without this, the sandbox may start but LLM calls from the agent workload are **denied**.

**Gate:** `./scripts/phase2-openshell.sh check` (config.yaml line)

---

## Step 6 — Deploy OpenClaw

```bash
source ~/labs/openshell-env.sh
cd ~/labs/openshell-on-openshift-lab

# Control UI bearer token (not the LLM key)
openssl rand -hex 32 | kubectl -n "${OPENCLAW_NAMESPACE}" \
  create secret generic openclaw-gateway-token \
  --from-file=token=/dev/stdin \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl kustomize manifests/openclaw | \
  sed "s/openshell\.openshell\.svc/openshell.${OPENSHELL_NAMESPACE}.svc/" | \
  kubectl -n "${OPENCLAW_NAMESPACE}" apply -f -

kubectl -n "${OPENCLAW_NAMESPACE}" set env deployment/openclaw \
  OPENSHELL_GATEWAY_URL="http://openshell.${OPENSHELL_NAMESPACE}.svc.cluster.local:8080"

kubectl -n "${OPENCLAW_NAMESPACE}" rollout restart deployment/openclaw
kubectl -n "${OPENCLAW_NAMESPACE}" rollout status deployment/openclaw --timeout=600s

OPENCLAW_URL="$(kubectl -n "${OPENCLAW_NAMESPACE}" get route openclaw -o jsonpath='https://{.spec.host}')"
kubectl -n "${OPENCLAW_NAMESPACE}" set env deployment/openclaw OPENCLAW_PUBLIC_URL="${OPENCLAW_URL}"
kubectl -n "${OPENCLAW_NAMESPACE}" rollout status deployment/openclaw --timeout=600s
```

> Always apply with **`-n openclaw`**. Never `kubectl apply -k` without namespace — resources land in `default`.

In-pod validation (does not print API key):

```bash
./scripts/phase2-openshell.sh probe
```

Or manually:

```bash
OPENCLAW_POD="$(kubectl -n "${OPENCLAW_NAMESPACE}" get pod \
  -l app.kubernetes.io/name=openclaw --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}')"

kubectl -n "${OPENCLAW_NAMESPACE}" exec "${OPENCLAW_POD}" -- sh -lc '
  node /app/openclaw.mjs config validate
  openshell status --gateway-endpoint "${OPENSHELL_GATEWAY_URL}"
  node /app/openclaw.mjs models status --probe \
    --probe-provider openai --probe-max-tokens 16 \
    --probe-timeout 60000 --json
'
```

Expect: config valid · OpenShell **Connected** · model probe `status: ok`.

**Gate:** `./scripts/phase2-openshell.sh check`

---

## Step 7 — Kit hardening (required)

```bash
cd "$HOME/AIOps/demo/demo-4-greenfield-install"
./scripts/phase2-openshell.sh harden
```

This runs:

- `patch-openclaw-seed-idempotent.sh` — power-cycle-safe init, Control UI logo, `@openclaw/slack` prep
- `ensure-openclaw-ui-origin.sh` — Route origin in `allowedOrigins` (fixes “Browser origin not allowed”)

---

## Step 8 — Verify Phase 2

```bash
./scripts/phase2-openshell.sh check
./scripts/phase2-openshell.sh probe
./scripts/phase2-openshell.sh ui
"$DEMO_KIT_ROOT/scripts/netobserv-e2e-openclaw-test.sh" ui-hints
./scripts/greenfield-install.sh phases    # openshell → ✓
```

**Control UI warm-up**

1. Open Route URL · paste **gateway token** (not LLM key)
2. `/new` in chat
3. Send first message — OpenShell spawns `openclaw-agent-*` in **15–60s** (`/new` alone does not create a sandbox)

```bash
"$DEMO_KIT_ROOT/scripts/openshell-sandbox-proof.sh" wait
```

**Next phase:** `./scripts/greenfield-install.sh rhoai`

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Browser origin not allowed` | Route HTTPS origin missing from `allowedOrigins` | `./scripts/phase2-openshell.sh harden` |
| Model probe fails / HTTP 400 | Wrong `contextWindow`/`maxTokens` or `baseUrl` | Edit `config.yaml`, re-apply kustomize, rollout restart |
| Sandbox starts, no LLM reply | Network policy missing LLM host | Edit `managed-policy.yaml`, re-apply |
| `openclaw-agent-*` ImagePullBackOff | Stale sandbox image ref | `"$DEMO_KIT_ROOT/scripts/fix-openshell-sandbox-image.sh"` |
| Init stuck / SQLite WAL error on `seed-openclaw` | June digest gone; newer Quay tags use unsafe SQLite | Mirror June image to your Quay (`mirror-openclaw-image-to-quay.sh` on reference bastion); set `OPENCLAW_*_IMAGE` in `env.local` |
| `curl: command not found` in `seed-openclaw` | Wrong image tag (not June 2026.6.11) | Use mirrored June image; seed harden v7 has node fetch fallback |
| `tar: command not found` in `seed-openclaw` | July+ images strip OS tools | Use mirrored June image; seed harden v7 has node tar.gz fallback |
| Init:CrashLoopBackOff on OpenClaw | ConfigMap missing managed policy | `oc -n openclaw apply -k ~/labs/.../manifests/openclaw` (not raw configmap) |
| Duplicate resources in `default` | `apply -k` without `-n` | Delete stray resources; always `-n openclaw` |
| Empty assistant content (16k MaaS) | Qwen thinking on small context | `./scripts/phase2-openshell.sh providers maas-16k` |
| Wrong baseUrl in probe vs disk config | ConfigMap not re-applied | Re-apply kustomize + rollout restart |

**Provider-specific fixes:** [PHASE-2-LLM-PROVIDERS.md](PHASE-2-LLM-PROVIDERS.md)

Full demo-kit narrative: `$DEMO_KIT_ROOT/README.md` **Phase 3 — Agentic AI**.

---

## Quick reference

```bash
source ~/labs/openshell-env.sh
./scripts/phase2-openshell.sh plan
./scripts/phase2-openshell.sh providers litemaas
./scripts/phase2-openshell.sh check
./scripts/phase2-openshell.sh probe
./scripts/phase2-openshell.sh harden
./scripts/phase2-openshell.sh ui
```
