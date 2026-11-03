# Baseline engine notes (Claude, Ollama)

Consulted: 2026-10-05. These are the facts that `llm/` uses, with their sources. Check the sources again before you change request or response handling.

## Claude via the Anthropic API (`llm/claude.py`, `ClaudeAPIEngine`)

Source: the Claude API reference bundled with Claude Code 2.1.289 (Python SDK README, structured outputs, model table), and Anthropic's pricing page https://platform.claude.com/docs/en/about-claude/pricing.

- SDK: `anthropic==1.11.0` (official). `anthropic.Anthropic()` reads `ANTHROPIC_API_KEY`. Version 1.x is built on `httpx2`.
- Models: `claude-sonnet-5-5` (Claude Sonnet 5.5) and `claude-opus-5-5` (Claude Opus 5.5). Use the exact IDs, with no date suffix.
- Structured output: `output_config={"format": {"type": "json_schema", "schema": ...}}` on `messages.create()`. The first text block is then valid JSON for the schema. The old `output_format` parameter is deprecated.
- Effort: `output_config.effort` (`low`, `medium`, `high`, `xhigh`, `max`). Opus 5.5 defaults to `medium` and Sonnet 5.5 to `high`, so the engine always sends `LLM_EFFORT` (default `medium`) explicitly.
- Thinking: adaptive by default. It cannot be disabled on Opus 5.5 (`{"type": "disabled"}` is a 400). The engine does not send `thinking`. `max_tokens` is 16000 so thinking has room before the JSON answer.
- Sampling: `temperature`, `top_p` and `top_k` are rejected on these models, so Claude runs are not deterministic.
- Refusals: HTTP 200 with `stop_reason: "refusal"` and `stop_details.category`. The engine records `invalid` with `refusal: <category>`. Anthropic recommends enabling server-side refusal fallbacks (`fallbacks`), which re-run the request on another model. We do not enable them on purpose, because the benchmark has to know which model made each decision.
- Errors: the SDK's typed exceptions. `RateLimitError` (429), `InternalServerError` (5xx) and `APIConnectionError` are retried up to 3 times with backoff by the engine (SDK retries are set to 0 so attempts can be counted). Any other `APIStatusError` (for example 401) is an immediate `error`.
- Usage: `usage.input_tokens`, `usage.output_tokens`, `usage.cache_creation_input_tokens`, `usage.cache_read_input_tokens`. The engine reports input tokens as the sum of the three input fields. Thinking tokens are billed as output tokens (the extended-thinking page says `usage.output_tokens_details.thinking_tokens` reports "how many of the billed output tokens were internal reasoning").
- Prices (2026-10-05, standard first-party API, USD per million tokens): Sonnet 5.5 $2 input / $10 output; Opus 5.5 $4 input / $20 output. The code does not build them in. Set `CLAUDE_SONNET_PRICE_*` / `CLAUDE_OPUS_PRICE_*` to estimate cost. `.env.example` lists these values as commented examples.

## Claude via the CLI fallback (`ClaudeCLIEngine`)

Used only when `ANTHROPIC_API_KEY` is not set. `run.py` prints a warning to stderr first, and every record says `transport: claude-cli`.

Checked against `claude --help` for Claude Code 2.1.289:

| Flag | Why |
|---|---|
| `-p` | non-interactive, prompt read from stdin |
| `--model claude-sonnet-5-5` / `claude-opus-5-5` | same model IDs as the API |
| `--output-format json` | one JSON result with `result`, `structured_output`, `usage`, `total_cost_usd`, `duration_ms`, `duration_api_ms`, `modelUsage` |
| `--json-schema <schema>` | the same schema as the API engine |
| `--system-prompt <text>` | replaces Claude Code's default system prompt with the same line the API gets |
| `--tools ""` | no tools |
| `--effort <LLM_EFFORT>` | same effort as the API engine |
| `--setting-sources ""` | loads no user, project or local settings, so no hooks or plugins from them |
| `--strict-mcp-config` (no `--mcp-config`) | no MCP servers |
| `--no-session-persistence` | nothing saved |

The command runs in a fresh empty temporary directory, so no `CLAUDE.md` is found. `CLAUDECODE` and `CLAUDE_CODE_*` variables from a parent Claude Code session are removed from the child's environment.

`--bare` would be the cleanest mode, but it requires `ANTHROPIC_API_KEY`, which is exactly what the fallback lacks.

The CLI still wraps the model in the Claude Code harness. Its token counts include the harness's own context, and `total_cost_usd` is recorded as `reported:claude-cli`. On a subscription plan, that figure is notional.

Isolation check: see "Live check" below.

## Ollama (`llm/ollama.py`)

Ollama 0.22.1, local HTTP API:

- `POST {OLLAMA_URL}/api/chat` with `model`, `messages` (system + user), `format` (a JSON schema, which constrains the output), `stream: false`, `options: {"temperature": 0, "seed": <run seed>}`.
- Response: `message.content` (the JSON answer), `prompt_eval_count` (input tokens), `eval_count` (output tokens), `total_duration` (nanoseconds), `model`.
- qwen3 thinks before answering by default. Its reasoning comes back in `message.thinking` and is not included in `eval_count`, so `output_tokens` covers only the final JSON answer, while latency includes the thinking. Seen in the 2026-10-05 batch: 35-39 output tokens next to 1,800-12,700 characters of thinking.
- Local timings depend on the machine. The machine used for the 2026-10-05 batch is listed in [results-2026-10-05-tier1-baselines.md](results-2026-10-05-tier1-baselines.md#test-machine-for-the-local-qwen38b-run).
- An unknown model is an HTTP 404 and counts as `error`. A server that cannot be reached is an `error` naming `OLLAMA_URL`.
- Cost is `n/a` (local).

## Live check (2026-10-05)

One run each, `--scenario degraded --seed 1234`, both HTTP 200 / exit 0 and parsed:

- `ollama:qwen3:8b` (Ollama 0.22.1): 2,197 input and 37 output tokens, 32.6 s (mostly loading the model on first use).
- `claude-sonnet` through the CLI fallback (Claude Code 2.1.289, no `ANTHROPIC_API_KEY`): `structured_output` present, `modelUsage` shows only `claude-sonnet-5-5`. 2,322 input tokens in total (2 uncached + 2,320 cache write), 128 output, 0 thinking tokens, 3.2 s round trip.

Isolation: the CLI's 2,322 input tokens are close to the size of our own prompt and schema (Ollama's tokenizer counts 2,197 for the same request), and only one model appears in `modelUsage`. Claude Code's default system prompt, tool definitions and this repo's `CLAUDE.md` would add many thousands of tokens, so none of them reached the model. `--setting-sources ""` is accepted by this CLI version. `num_turns` is 2: the CLI delivers structured output through an extra internal turn.

Cost: the CLI reported $0.010564. That matches Anthropic's list price for a 1-hour cache write of the whole prompt (2,320 × $4/MTok, twice the base input price) plus 128 × $10/MTok output. The CLI caches the prompt and the API engine does not. As a result, the cost that the CLI reports is about 1.8× what the same call would cost through the API. Compare cost only between runs with the same transport.

## Tier 3 and Tier 4 Claude baseline (`tier3/claude_engine.py`)

Added 2026-10-18. The same transports and settings as above (model IDs, `LLM_EFFORT`, `max_tokens`, retries, the CLI isolation flags and environment), with Tier 3's question set instead of Tier 1's:

- The questions and option descriptions are `tier3/questions.py`, the same objects the Jev adapter sends. `incident_detected` is Jev's yes/no (`noul`) question. For Claude its options are the strings `true` / `false` (Jev answers a probability, scored at 0.5 or more).
- The observation is wrapped as `{"observations": ...}`, the key the Jev adapter uses.
- The JSON schema allows exactly the four answer sets. The parser is strict: a missing or out-of-set value is `invalid`. Claude reports no confidence, so `confidence` is null.
- Cost: `total_cost_usd` from the CLI (`reported:claude-cli`), else an estimate from `CLAUDE_OPUS_PRICE_*`, else null.
- Smoke check, 2026-10-18, CLI transport (no `ANTHROPIC_API_KEY`), one stored K000 observation: exit 0, valid answer, 6,732 input / 394 output tokens, about 7 s.
