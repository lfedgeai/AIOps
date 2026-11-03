## Context

Tier 1 sends one generated observation per run to Jev (`jev/client.py`) and scores the answer against a hand-written expected outcome. `run.py` already takes the engine as an object with a `decide(observations) -> Decision` method (`main(argv, client=...)`), and tests use `FakeClient` through that seam. The `Decision` dataclass, the `QUESTIONS` wording and the status constants all live in `jev/client.py`.

Jev returns per-option probabilities and a confidence. General-purpose LLMs do not. Jev publishes no price. Claude has published prices, but the project rule is that cost is either reported by the engine or estimated from user-set prices.

Environment checked on 2026-10-05: `claude` CLI 2.1.289 (has `--json-schema`, `--system-prompt`, `--tools`, `--setting-sources`, `--strict-mcp-config`, `--effort`, `--output-format json`, `--no-session-persistence`; `--bare` requires `ANTHROPIC_API_KEY` so it cannot serve the no-key fallback). Ollama 0.22.1 is running locally with `qwen3:8b` among others. `ANTHROPIC_API_KEY` is not set.

## Goals / Non-Goals

**Goals:**
- Compare Jev with Claude Sonnet 5.5, Claude Opus 5.5 and a local Ollama model on exactly the same Tier 1 observations, questions and option sets.
- Keep every existing rule: hand-written expected outcomes, strict parsing, no invented cost, no leaks, offline tests.
- Make the transport (API vs `claude -p`) impossible to miss, both on screen and in the results.
- A comparison report that reads several result files and lines engines up per decision and per scenario.

**Non-Goals:**
- Tier 2. The engines are written so a later change can wrap them as `AIOpsEngine`s.
- Prompt engineering to make the LLMs score better. The baseline asks the same questions Jev gets, with a short neutral system prompt.
- OpenAI or other hosted providers.
- Self-reported confidence from LLMs. It is not comparable with Jev's probabilities, so `confidence` stays null.

## Decisions

### D1. One `llm/` package, sharing Jev's question set
`llm/prompt.py` builds the prompt and a JSON schema from `jev.client.QUESTIONS`. Each question becomes an enum-constrained property: `severity`, `action`, `probable_cause` from their `criteria` keys, and `human_review` as `"yes"`/`"no"`. The `noul` question keeps its wording; the two criteria texts become the descriptions of `yes` and `no`. The prompt contains the question instructions and option descriptions verbatim, then the observation JSON (`{"server_observations": obs}`, the same `state` shape Jev receives). The system prompt is one neutral line: "You assess server observations and answer each question with exactly one of the allowed options."
*Why:* importing `QUESTIONS` means the wording cannot drift between engines. *Alternative:* a separate LLM-tuned prompt. Rejected because the comparison would then measure prompt differences, not engines.

### D2. Engines return the existing `Decision`
`llm/engines.py` (or one module per engine) defines `ClaudeAPIEngine`, `ClaudeCLIEngine` and `OllamaEngine`, each with `decide(observations) -> Decision`, reusing `Decision` and the status constants from `jev/client.py`. Strict parsing goes through one shared function: the reply must be a JSON object with all four fields, each in its allowed set, or the decision is `invalid` with the reason. `confidence`, `field_confidence`, `probabilities` and `human_review_probability` stay empty or null. `raw_response` keeps the model output (text and metadata, never credentials).
*Why:* the runner, evaluator and report already understand `Decision`; nothing downstream changes. *Alternative:* a new decision type plus an adapter. More code for no gain.

### D3. Claude over the official SDK; fallback to `claude -p`
- With `ANTHROPIC_API_KEY` set: `anthropic.Anthropic().messages.create(model=..., max_tokens=1024, system=..., messages=[...], output_config={"format": {"type": "json_schema", "schema": ...}, "effort": LLM_EFFORT})`. Thinking is left at the model default (adaptive; it cannot be turned off on Opus 5.5). Temperature is not sent (rejected on these models). Model IDs: `claude-sonnet-5-5`, `claude-opus-5-5`. Usage comes from `response.usage` (`input_tokens`, `output_tokens`, cache fields kept raw). `response.model` becomes `model_version`. The SDK's own retries (default 2) handle 429/5xx.
- Without it: run `claude -p` through an injectable runner (`subprocess.run` by default) with `--model <id> --output-format json --json-schema <schema> --system-prompt <system> --tools "" --effort <LLM_EFFORT> --no-session-persistence --strict-mcp-config --setting-sources ""`, the prompt on stdin, and `cwd` set to a fresh empty temp directory so no `CLAUDE.md` is discovered. The JSON result supplies the structured output, `usage`, `total_cost_usd` and `duration_ms`.
- The engine factory decides the transport once per batch and prints a banner to stderr when it falls back: `*** ANTHROPIC_API_KEY not set: using the claude CLI (claude -p) for <model>. Results are labelled transport=claude-cli and are not directly comparable with API runs. ***`. The same text goes into the summary meta and the report header.
- Refusals (`stop_reason == "refusal"`) are `invalid` with message `refusal: <category>`: the model replied but gave no decision, so it counts as incorrect, like any out-of-set reply. Server-side refusal fallbacks are deliberately **not** enabled. They would let a different model answer, and the benchmark has to know which model made each decision.
*Why the SDK:* the official SDK is the supported path for Python; it gives typed errors, retries and usage. *Alternative:* raw `requests` like Jev. Rejected, because the SDK exists and Jev's raw HTTP is a Jev-specific choice.

### D4. Ollama over its local HTTP API
`POST {OLLAMA_URL}/api/chat` (default `http://localhost:11434`) with `model`, `messages` (system + user), `format: <json schema>`, `stream: false`, `options: {"temperature": 0, "seed": <run seed>}`. Tokens: `prompt_eval_count` and `eval_count`. Server time: `total_duration` (ns → ms). Cost: always `n/a` (local). The model name comes from `--engine ollama:<model>`, so there is no default model. A connection failure or unknown model is `error`. The HTTP call for Ollama lives only in this engine.
*Why:* temperature 0 and a fixed seed make local runs reproducible; hosted Claude cannot offer that. *Alternative:* the `ollama` Python package. Rejected because `requests` is already a dependency.

### D5. Engine selection and identity
`run.py --engine {jev, claude-sonnet, claude-opus, ollama:<model>}` (default `jev`). `run-all.sh` passes extra args through already, so `./run-all.sh 70 --engine claude-sonnet` works; its "key check" step only demands `JEV_API_KEY` when the engine is `jev`. Each record gets `engine: {name, transport, model_requested, model_version}`; meta gets the same plus the fallback notice. Result files become `<stamp>-<runs>runs[-nologs]-<engine>.jsonl`, with Jev files keeping today's name so existing tooling and docs stay valid. Records without `engine` are read as Jev.

### D6. Cost
Claude API: estimated from per-engine `CLAUDE_SONNET_PRICE_*` / `CLAUDE_OPUS_PRICE_*` (prices differ per model) when the user sets them, else `n/a` (`cost_source: estimated`). No price is built in, matching the Jev rule, even though Anthropic publishes prices. The README points to Anthropic's pricing page. Claude CLI: `total_cost_usd` is recorded as `cost_source: reported:claude-cli`, with a report note that on a subscription plan this is a notional figure, not a charge. Ollama: `n/a`.
*Alternative:* hard-code Anthropic's list prices. Rejected because it breaks "never invent cost" and goes stale.

### D7. Comparison report
`report.py --compare f1.jsonl f2.jsonl ...` loads each file, groups by engine name, and renders: overall accuracy, per-decision accuracy, probable cause, per-scenario accuracy, telemetry (tokens, latency p50/p95, cost or n/a), invalid/error counts, and the transport per engine. It checks that every file covers the same `(scenario, seed)` pairs and with/without logs. If not, it prints a warning at the top, not an error. All figures come from `evaluation/evaluator.aggregate`, so nothing is computed differently per engine.

### D8. Leak guarantee extends to every engine
The existing `test_request_does_not_contain_scenario_or_expected` pattern is repeated for the LLM prompt: for every scenario, the prompt and schema must not contain the scenario name or the expected values in a way that singles them out. Option names naturally appear because they are the allowed answers, exactly as in Jev's request. For the CLI fallback, running in an empty temp dir with `--setting-sources ""`, `--strict-mcp-config` and no tools keeps the repo's `CLAUDE.md`, hooks and MCP servers (which mention scenario names) out of the model's context.

## Risks / Trade-offs

- [CLI fallback is not a bare model: Claude Code adds its own harness, and user plugins or hooks could inject context] → Run with an empty cwd, `--setting-sources ""`, `--strict-mcp-config`, `--tools ""` and a replacing `--system-prompt`. Task 5.4 verifies isolation with a canary run that compares the CLI's reported input tokens with the API prompt size (or, without a key, with the token count of our own prompt) and fails loudly if the gap suggests injected context. Results are always labelled `transport=claude-cli`.
- [No determinism on Claude: temperature is not settable, and thinking is adaptive] → Repeat runs (70 = 10 per scenario) and report the spread per scenario. Record `LLM_EFFORT` in meta.
- [Effort changes results: Opus 5.5 defaults to `medium`, Sonnet 5.5 to `high`] → Send effort explicitly for both. `LLM_EFFORT` defaults to `medium` for both and is recorded per run.
- [Structured output makes invalid replies rare for Claude, so "invalid" counts are not comparable with Jev's] → Report invalid counts per engine anyway. The constraint is part of the engine as used, and Jev's typed questions are its own form of constrained output.
- [Small local models may ignore the schema] → Ollama's `format` enforces JSON. Anything out of set is `invalid` and counted, which is a valid result.
- [Spending real money by accident] → The engine is chosen explicitly. Batches print the number of calls before starting, as `run-all.sh` already does.
- [`claude -p` flag drift between CLI versions] → Record the CLI version (`claude --version`) in meta. A non-zero exit or unparseable JSON is `error` with stderr's first 200 characters.

## Migration Plan

Additive. `jev` remains the default engine, and existing result files and reports keep working (a missing `engine` means Jev). Rollback is reverting the change. No data migration.

## Open Questions

- Should the comparison also run with `--no-logs` for each engine? Supported by the design (same flag), so it is a decision about which batches to run, not a code question.
- Tier 2 wrapping is deferred to a follow-up change.
