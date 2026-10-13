# Phase 2 — LLM provider presets

Use with [PHASE-2-OPENSHELL.md](PHASE-2-OPENSHELL.md). Pick **one** profile, fill in secrets/hosts, then run `./scripts/phase2-openshell.sh probe`.

```bash
./scripts/phase2-openshell.sh providers              # list profiles
./scripts/phase2-openshell.sh providers litemaas   # settings for a profile
```

---

## Profile comparison

| Profile | Typical use | `contextWindow` | `maxTokens` | Qwen thinking | Notes |
|---------|-------------|-----------------|-------------|---------------|-------|
| **litemaas** | RHOAI / LiteMaaS 128k | `131072` | `12288` | `true` | Recommended default for this guide |
| **vllm** | Self-hosted vLLM | match server | match server | optional | Set `LLM_BASE_URL` + model id |
| **maas-16k** | Legacy workshop MaaS | `16384` | `6144` | **`false`** | Thinking burns output budget |
| **openai** | Public OpenAI API | model default | model default | n/a | Stock lab default; policy allows `api.openai.com` |

---

## litemaas (recommended)

| Setting | Value |
|---------|--------|
| `LLM_BASE_URL` | `https://litemaas.example.com/v1` |
| Model id | `Qwen3.6-35B-A3B` |
| Primary | `openai/Qwen3.6-35B-A3B` |
| Policy host | `litemaas.example.com:443` |

`config.yaml` essentials:

```json
{
  "agents": {
    "defaults": {
      "model": { "primary": "openai/Qwen3.6-35B-A3B" },
      "models": {
        "openai/Qwen3.6-35B-A3B": {
          "alias": "Qwen3.6-35B-A3B",
          "params": {
            "extra_body": {
              "chat_template_kwargs": { "enable_thinking": true }
            }
          }
        }
      },
      "compaction": {
        "mode": "default",
        "reserveTokens": 4096,
        "reserveTokensFloor": 4096,
        "keepRecentTokens": 2048,
        "maxHistoryShare": 0.5,
        "recentTurnsPreserve": 2
      }
    }
  },
  "models": {
    "providers": {
      "openai": {
        "baseUrl": "https://litemaas.example.com/v1",
        "apiKey": "env:OPENAI_API_KEY",
        "api": "openai-completions",
        "models": [{
          "id": "Qwen3.6-35B-A3B",
          "name": "Qwen3.6 35B A3B (LiteMaaS)",
          "contextWindow": 131072,
          "maxTokens": 12288
        }]
      }
    }
  }
}
```

**After Phase 5 (guardrails):** probe may show requests via `netobserv-llm-guard-proxy` — that is expected.

---

## vllm (generic OpenAI-compatible)

Replace placeholders:

```bash
export LLM_BASE_URL="https://YOUR-VLLM-HOST/v1"
export LLM_MODEL="your-model-id"
export LLM_POLICY_HOST="YOUR-VLLM-HOST"   # no scheme, for managed-policy.yaml
```

| Check | Command |
|-------|---------|
| Models list | `curl -sS -H "Authorization: Bearer $KEY" "${LLM_BASE_URL}/models"` |
| Chat smoke | See PHASE-2-OPENSHELL.md step 4a |
| In-cluster | `oc run llm-probe …` (same doc) |

`config.yaml` pattern:

```json
"agents": {
  "defaults": {
    "model": { "primary": "openai/YOUR-MODEL-ID" }
  }
},
"models": {
  "providers": {
    "openai": {
      "baseUrl": "https://YOUR-VLLM-HOST/v1",
      "apiKey": "env:OPENAI_API_KEY",
      "api": "openai-completions",
      "models": [{
        "id": "YOUR-MODEL-ID",
        "name": "Your vLLM model",
        "contextWindow": 32768,
        "maxTokens": 8192
      }]
    }
  }
}
```

**Policy** (`managed-policy.yaml`):

```yaml
- host: YOUR-VLLM-HOST
  port: 443
  protocol: rest
  access: full
  enforcement: enforce
```

**Common vLLM failures**

| HTTP / symptom | Fix |
|----------------|-----|
| 400 `maximum context length` | Lower `contextWindow` / message size |
| 404 on `/v1/chat/completions` | Try `baseUrl` with/without `/v1` |
| Connection reset from sandbox | Add policy host; verify egress from `openshell` ns |
| Empty `content`, `stopReason=length` | Lower `maxTokens` or disable thinking |

---

## maas-16k (legacy workshop)

Workshop endpoint example: `https://maas-rhdp.apps.maas.redhatworkshops.io/v1`

| Setting | Value |
|---------|--------|
| Model id | `qwen3-14b` |
| Primary | `openai/qwen3-14b` |
| `contextWindow` | `16384` |
| `maxTokens` | `6144` |
| Thinking | **`enable_thinking: false`** |

Investigate turns often hit `stopReason=length` on 16k — acceptable for short demos only. Prefer **litemaas** or **vllm** for full Demo A.

Policy host: `maas-rhdp.apps.maas.redhatworkshops.io` (or your workshop hostname).

---

## openai (public API)

Stock lab `config.yaml` works for `api.openai.com`. Create `my-llm-key` with your OpenAI key.

- No managed-policy edit required (stock allows `api.openai.com:443`).
- Not recommended for customer demos (data egress, cost, rate limits).

---

## Apply config changes (all profiles)

```bash
source ~/labs/openshell-env.sh
cd ~/labs/openshell-on-openshift-lab

# Edit config.yaml + policies/managed-policy.yaml (except public OpenAI)

kubectl kustomize manifests/openclaw | \
  sed "s/openshell\.openshell\.svc/openshell.${OPENSHELL_NAMESPACE}.svc/" | \
  kubectl -n "${OPENCLAW_NAMESPACE}" apply -f -

kubectl -n "${OPENCLAW_NAMESPACE}" rollout restart deployment/openclaw
kubectl -n "${OPENCLAW_NAMESPACE}" rollout status deployment/openclaw --timeout=600s

cd "$HOME/AIOps/demo/demo-4-greenfield-install"
./scripts/phase2-openshell.sh probe
```

**Live config drift:** if probe shows the wrong `baseUrl` but disk `config.yaml` is correct, re-apply kustomize (ConfigMap not updated) and restart the pod.

---

## Verify

```bash
./scripts/phase2-openshell.sh probe
```

Expect: `config valid` · OpenShell **Connected** · `"status": "ok"` in probe JSON.
