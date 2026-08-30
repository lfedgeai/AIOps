# Phase 5 — TrustyAI Guardrails

**Time:** ~30–60 minutes (DSC TrustyAI + detector model cache)  
**Orchestrator:** `./scripts/greenfield-install.sh guardrails`  
**Helper:** `./scripts/phase5-guardrails.sh` (`plan` · `deploy` · `check` · `status` · `verify`)

**Layer 0** — routes OpenClaw LLM traffic through TrustyAI Guardrails before your external LiteMaaS/vLLM backend.

---

## Upstream reference (lemonade-stand-assistant)

TrustyAI is installed from the **pinned** [rh-ai-quickstart/lemonade-stand-assistant](https://github.com/rh-ai-quickstart/lemonade-stand-assistant) repo.

| Upstream | NetObserv greenfield |
|----------|----------------------|
| Repo | `LEMONADE_REPO` in `supply-chain-pins.env` |
| Branch / commit | `nemo-guardrails` @ `b342f224…` |
| Chart path | `fms-orchestrator/chart` (not `nemo-guardrails/chart`) |
| Helm release | `netobserv-trustyai-guardrails` in `netobserv-guardrails` |
| LLM | **Option A — external MaaS** (`model.endpoint` + `model.port` + `model.api_key` from `my-llm-key`) |
| Workshop UI | **Removed** (`SKIP_LEMONADE_STAND=1`) — OpenClaw uses `netobserv-llm-guard-proxy` |
| Gateway preset | `netobserv-sre` in `manifests/trustyai-guardrails/01-gateway-config.yaml` |
| NLP config patch | `02-orchestrator-builtin-detector.yaml` (applied **after** Helm — built-in regex detector + external TLS) |

Pin: `scripts/supply-chain-pins.env` · Install: `scripts/install-trustyai-guardrails.sh`

---

## Before you start

| Requirement | Check |
|-------------|--------|
| Phase 2 complete | `openshell ✓` · `my-llm-key` present |
| Phase 3 recommended | `default-dsc` from RHOAI (TrustyAI patch extends it) |
| LLM in site config | `./scripts/greenfield-install.sh config validate openshell` |
| Cluster memory | Prompt-injection detector requests **16Gi** (upstream default) |

```bash
./scripts/phase5-guardrails.sh plan
```

---

## Automated install

```bash
cd ~/AIOps/demo/demo-4-greenfield-install
./scripts/greenfield-install.sh guardrails
```

What `install-trustyai-guardrails.sh` does (aligned with upstream README):

1. Enable TrustyAI + KServe in `default-dsc`
2. Wait for `guardrailsorchestrators.trustyai.opendatahub.io` CRD
3. Apply NetObserv gateway + guard-proxy manifests
4. `helm upgrade --install` **`fms-orchestrator/chart`** with external MaaS settings
5. Wait for MinIO HuggingFace model cache (detector weights)
6. Patch `fms-orchestr8-config-nlp` (built-in-detector + LiteMaaS TLS)
7. Enable gateway on `GuardrailsOrchestrator` CR
8. Prune `lemonade-stand` workshop UI (not used by OpenClaw)

Then `wire-openclaw-trustyai-guardrails.sh` points OpenClaw `baseUrl` at the guard proxy.

Verify:

```bash
./scripts/phase5-guardrails.sh check
./scripts/phase5-guardrails.sh verify
```

**Next phase:** `./scripts/greenfield-install.sh aap`

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Helm: ConfigMap ownership error | `fms-orchestr8-config-nlp` applied before Helm | Fixed in install script — Helm first, then patch |
| Detector `Init:CrashLoopBackOff` | Predictors started before MinIO finished HF download | Wait for `minio-storage-guardrail-detectors` Ready; delete predictor pods |
| `GuardrailsOrchestrator` not Ready | InferenceServices still loading | `oc get isvc -n netobserv-guardrails` — allow 10–15 min |
| OpenClaw still on LiteMaaS URL | Wire not run | `wire-openclaw-trustyai-guardrails.sh all` |
| `/chat/completions` HTTP ≠ 200 | Guard proxy cold start | Wait ~90s; check `netobserv-llm-guard-proxy` logs |
| OOM on detectors | Cluster too small | `LIGHT_DETECTORS=1 install-trustyai-guardrails.sh install` |

Upstream validation: [lemonade-stand-assistant README — Validating the deployment](https://github.com/rh-ai-quickstart/lemonade-stand-assistant#validating-the-deployment)

Full NetObserv guide: `$DEMO_KIT_ROOT/docs/TRUSTYAI-GUARDRAILS-GUIDE.md`

---

## Quick reference

```bash
./scripts/phase5-guardrails.sh plan
./scripts/phase5-guardrails.sh status
./scripts/phase5-guardrails.sh check
$DEMO_KIT_ROOT/scripts/install-trustyai-guardrails.sh status
$DEMO_KIT_ROOT/scripts/netobserv-e2e-openclaw-test.sh trustyai-guard-check
```
