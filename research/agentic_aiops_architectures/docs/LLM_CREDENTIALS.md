# LLM API Credentials

Each hosted agent requires an API key for its LiteLLM endpoint. Local agents (Ollama) need no key.

## Quick Setup

1. Copy `config/.env.example` to `config/.env`
2. Add your API keys
3. Run `./scripts/run_harness.sh` — it sources `config/.env` automatically

```bash
cp config/.env.example config/.env
# Edit config/.env with your keys
./scripts/run_harness.sh --flag cartFailure --variant on
```

## Agent API Keys

| Agent | Env var | Model | Default endpoint |
|-------|---------|-------|-----------------|
| `deepseek_agent` | `DEEPSEEK_API_KEY` | deepseek-r1-distill-qwen-14b | litellm-prod.apps.maas.redhatworkshops.io |
| `qwen3_agent` | `QWEN3_API_KEY` | qwen3-14b | litellm-prod.apps.maas.redhatworkshops.io |
| `llama_scout_agent` | `LLAMA_SCOUT_API_KEY` | llama-scout-17b | litellm-prod.apps.maas.redhatworkshops.io |
| `ollama_qwen2` | `OPENAI_API_KEY` (default: `ollama`) | qwen2.5 | localhost:11434 |

## MLflow Logging

When `MLFLOW_RUN_ID` is set (the harness does this automatically), agents upload:

| Artifact | Contents |
|----------|----------|
| `agent_prompts.txt` | Human-readable conversation transcript |
| `agent_tool_calls.txt` | Each tool call with arguments and results |
| `agent_llm_rounds.txt` | Round-by-round LLM reasoning summary |
| `agent_llm_prompts.json` | Machine-readable prompts + conversation |
| `agent_llm_tool_calls.json` | Machine-readable tool call log |
| `agent_llm_rounds.json` | Machine-readable per-round data |

## Security

- **Do not** commit `config/.env` — it is gitignored
- For CI/CD, inject keys as environment variables or secrets
