## 1. Setup

- [x] 1.1 Add `anthropic` to `requirements.txt`, pinned to the installed tested version, and check that it installs on Python 3.10
- [x] 1.2 Add `ANTHROPIC_API_KEY`, `LLM_EFFORT`, `OLLAMA_URL`, `CLAUDE_SONNET_PRICE_*` and `CLAUDE_OPUS_PRICE_*` (input and output per million tokens) to `.env.example` and the README variables table

## 2. Shared prompt and parsing

- [x] 2.1 `llm/prompt.py`: build the system prompt, user prompt and JSON schema from `jev.client.QUESTIONS` (enums per field, `human_review` as yes/no)
- [x] 2.2 Shared strict parser: JSON text or dict → `Decision` (`ok` / `invalid` with reason), never coercing
- [x] 2.3 Tests: schema enums equal the Tier 1 option sets, prompt leak test over every scenario, parser accepts valid answers and rejects out-of-set, missing-field and non-JSON replies

## 3. Claude engines

- [x] 3.1 `ClaudeAPIEngine` on the official SDK: `output_config` with format and effort, usage and model captured, refusal → `invalid`, SDK errors → `error`, injectable client for tests
- [x] 3.2 `ClaudeCLIEngine`: `claude -p` with the isolation flags from design D3 in a fresh temp dir, prompt on stdin, JSON result parsed (structured output, usage, `total_cost_usd`, duration), non-zero exit or bad JSON → `error`, injectable runner for tests
- [x] 3.3 Engine factory: choose API or CLI from `ANTHROPIC_API_KEY`, print the stderr fallback banner once, raise a configuration error naming both options when neither is available
- [x] 3.4 Cost: reported (CLI) or estimated from the per-engine `CLAUDE_*_PRICE_*`, else `n/a`
- [x] 3.5 Tests with a fake SDK client and a fake subprocess runner: happy path, refusal, invalid JSON, CLI failure, banner printed only on fallback, no key in any saved field

## 4. Ollama engine

- [x] 4.1 `OllamaEngine`: `/api/chat` with schema `format`, temperature 0, run seed, tokens and server time from the response, connection error → `error` naming `OLLAMA_URL`
- [x] 4.2 Tests with canned HTTP responses: happy path, out-of-set answer, unreachable server, seed and temperature in the request

## 5. Runner and report

- [x] 5.1 `run.py --engine`: parse and validate, build the engine, pass the run seed where the engine uses it, add the `engine` block to records and meta (effort, transport, CLI version), and name result files with the engine for non-Jev engines
- [x] 5.2 `run-all.sh`: require `JEV_API_KEY` only for the `jev` engine and show the engine in the step headers
- [x] 5.3 `report.py --compare`: per-engine tables (overall, per decision, cause, per scenario, invalid/error, telemetry, transport), mismatch warning; single-file mode unchanged and old files read as Jev
- [x] 5.4 Isolation check for the CLI fallback: one canary run comparing the reported input tokens with our prompt size, documented in `docs/llm-baseline-notes.md`
- [x] 5.5 Tests: default engine unchanged (file names, records), unknown engine rejected, compare report on fake records, mismatch warning, old record without `engine`

## 6. Docs and rules

- [x] 6.1 `docs/llm-baseline-notes.md`: API facts used (SDK, model IDs, structured outputs, effort, refusal handling, no fallbacks) with sources and date; CLI flags and version; Ollama API fields
- [x] 6.2 README: a "Compare with an LLM" section (commands, transport notice, cost note, determinism caveat)
- [x] 6.3 `CLAUDE.md`: allow `llm/` as the home for baseline engine HTTP/SDK code, extend the "sees only the observations" rule to all engines, add the commands to the table
- [x] 6.4 `./run-all.sh --check` passes offline with no keys; full `pytest` green

## 7. First comparison (live, needs approval)

- [x] 7.1 One run per engine (`--scenario degraded --seed 1234`) to check each transport end to end
- [x] 7.2 70-run batches with the same `--seed` for Jev, `claude-sonnet`, `claude-opus` and `ollama:qwen3:8b`, then `report.py --compare`
