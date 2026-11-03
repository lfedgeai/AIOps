## Why

Tier 1 measures how often Jev makes the right operational decision (48.6% over 70 runs on 2026-10-05), but a number alone does not say whether that is good. A general-purpose LLM given the same observation and the same questions is the obvious baseline: it shows whether Jev's typed-decision model adds anything over asking a capable chat model.

## What Changes

- Add an `--engine` option to `run.py` and `run-all.sh`. `jev` stays the default, so existing commands behave exactly as before.
- New engines: `claude-sonnet` (`claude-sonnet-5-5`), `claude-opus` (`claude-opus-5-5`) and `ollama:<model>` (a local model, for example `ollama:qwen3:8b`).
- Claude transport: the Anthropic API when `ANTHROPIC_API_KEY` is set. Without it, fall back to the `claude -p` CLI, announce that loudly on stderr at the start of the batch, and record the transport in every run record and the summary.
- Every engine receives the same observation, the same question wording and the same option sets that Jev gets (built from the existing `QUESTIONS`). Answers are parsed strictly: an out-of-set answer is `invalid`, a failed call is `error`, exactly as for Jev.
- Run records and summaries carry an `engine` block (name, transport, model requested, model that answered). Result file names include the engine.
- New comparison report: `report.py --compare a.jsonl b.jsonl ...` puts engines side by side (overall, per decision, per scenario, telemetry), and warns when the runs did not use the same seeds.
- Scenarios, expected outcomes, scoring and the Jev adapter are unchanged. Tier 2 is out of scope (a later change can wrap these engines as `AIOpsEngine`s).

## Capabilities

### New Capabilities
- `llm-baseline-engines`: Claude (API or CLI fallback) and Ollama engines that answer the Tier 1 questions from the same observation as Jev, with strict parsing, telemetry and honest cost handling.
- `engine-comparison`: engine selection on the command line, engine identity in records and summaries, and the side-by-side comparison report.

### Modified Capabilities
<!-- Existing capabilities live under openspec/changes/archive/2026-11-02-jev-ops-experiment-v1/specs (not yet archived to openspec/specs). Their requirements still hold for the Jev engine; the new specs add to them rather than change them. -->

## Impact

- New package `llm/` (prompt builder and three engines). `run.py` (engine selection, file naming, meta), `report.py` (compare mode), `run-all.sh` (pass `--engine` through).
- `jev/client.py` is unchanged. The shared `QUESTIONS`, `Decision` and status constants are imported from it, so the wording cannot drift between engines.
- New pinned dependency: `anthropic` (official SDK) in `requirements.txt`. Ollama is called over its local HTTP API with `requests`; `claude` CLI is optional and only used for the fallback.
- New environment variables in `.env.example` and the README: `ANTHROPIC_API_KEY`, `LLM_EFFORT`, `OLLAMA_URL`, `CLAUDE_SONNET_PRICE_*` and `CLAUDE_OPUS_PRICE_*` (input and output per million tokens).
- `CLAUDE.md`: the "only `jev/client.py` touches HTTP/API" rule gains `llm/` as the home for baseline engines; the "Jev sees only the observations" rule extends to every engine.
- Tests stay offline: fake SDK client, fake subprocess runner and canned Ollama responses. No test calls Anthropic, Ollama or `claude`.
