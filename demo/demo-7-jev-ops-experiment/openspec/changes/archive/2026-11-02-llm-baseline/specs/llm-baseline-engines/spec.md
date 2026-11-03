## ADDED Requirements

### Requirement: Same input as Jev
Every baseline engine SHALL receive the same observation that Jev receives for a run, and SHALL be asked the same four questions with the same instructions and the same allowed options, built from the shared Tier 1 question set. The prompt MUST NOT contain the scenario name, the expected outcome, or any hint toward it.

#### Scenario: Question wording shared
- **WHEN** the Tier 1 question set changes
- **THEN** the baseline prompt and schema change with it, with no separate copy to update

#### Scenario: No leak in the prompt
- **WHEN** a prompt is built for any scenario
- **THEN** it contains neither the scenario name nor any field of the expected outcome as a hint

#### Scenario: Logs withheld consistently
- **WHEN** a batch runs with `--no-logs`
- **THEN** the baseline engine receives the observation without logs, exactly as Jev would

### Requirement: Strict parsing
A baseline engine SHALL accept an answer only when all four fields are present and each value is in its allowed set. Anything else, including a refusal, SHALL be `invalid` with the reason recorded. A failed call SHALL be `error`. No value is coerced or guessed.

#### Scenario: Valid answer
- **WHEN** the model returns all four fields with allowed values
- **THEN** the decision is `ok` with those values

#### Scenario: Out-of-set answer
- **WHEN** the model returns a value outside an allowed set, or omits a field
- **THEN** the decision is `invalid` and counts as incorrect

#### Scenario: Refusal
- **WHEN** the Claude API returns `stop_reason` `refusal`
- **THEN** the decision is `invalid` with message `refusal: <category>`, and no other model is asked

#### Scenario: Transport failure
- **WHEN** the call fails (network, HTTP error, CLI non-zero exit, unknown Ollama model)
- **THEN** the decision is `error` and is excluded from accuracy

### Requirement: No invented confidence
Baseline engines SHALL leave confidence, per-field confidence, probabilities and human-review probability empty or null.

#### Scenario: LLM decision
- **WHEN** a baseline engine returns a decision
- **THEN** its confidence is null

### Requirement: Claude transport and loud fallback
The Claude engines SHALL call the Anthropic API through the official SDK when `ANTHROPIC_API_KEY` is set. When it is not set, they SHALL call the `claude -p` CLI instead, print a clearly marked notice to stderr before the first run, and record `transport` (`anthropic-api` or `claude-cli`) in every run record and in the batch summary.

#### Scenario: Key present
- **WHEN** `ANTHROPIC_API_KEY` is set and a Claude engine is chosen
- **THEN** every run uses the API and records `transport: anthropic-api`

#### Scenario: Key absent
- **WHEN** `ANTHROPIC_API_KEY` is not set and a Claude engine is chosen
- **THEN** a fallback notice is printed to stderr before any call, every run records `transport: claude-cli`, and the summary meta carries the notice

#### Scenario: Neither available
- **WHEN** no key is set and the `claude` command is not found
- **THEN** the run stops before any call with a configuration error that names both options

### Requirement: Isolated CLI fallback
The CLI fallback SHALL run with a replacing system prompt, no tools, no setting sources, no MCP servers, no session persistence, and a fresh empty working directory, so that no repository file, hook or memory reaches the model.

#### Scenario: Working directory
- **WHEN** the CLI fallback runs
- **THEN** its working directory is an empty temporary directory, not the repository

### Requirement: Explicit model and effort
The Claude engines SHALL use the model IDs `claude-sonnet-5-5` and `claude-opus-5-5` and SHALL send the effort level from `LLM_EFFORT` (default `medium`) explicitly on both transports. Effort SHALL be recorded in the run meta.

#### Scenario: Effort recorded
- **WHEN** a Claude batch finishes
- **THEN** the summary meta states the effort level used

### Requirement: Local Ollama engine
The Ollama engine SHALL call the local Ollama HTTP API with the model named in `--engine ollama:<model>`, constrain the output with the same JSON schema, use temperature 0 and the run seed, and report tokens and server time from the response. There is no default model.

#### Scenario: Reproducible local run
- **WHEN** the same Ollama model runs the same scenario and seed twice
- **THEN** both requests carry the same seed and temperature 0

#### Scenario: Ollama not running
- **WHEN** the Ollama server cannot be reached
- **THEN** the decision is `error` with a message naming `OLLAMA_URL`

### Requirement: Honest cost
Baseline engines SHALL use a cost the engine reports (the CLI's `total_cost_usd`, marked as reported by the CLI) or an estimate from user-set per-engine prices (`CLAUDE_SONNET_PRICE_*`, `CLAUDE_OPUS_PRICE_*`), and otherwise report `n/a`. No price is built into the code. Ollama cost is always `n/a`.

#### Scenario: No prices set
- **WHEN** the Claude API is used and no LLM prices are set
- **THEN** cost is `n/a`

### Requirement: Credentials never stored
No decision, record, summary or report SHALL contain `ANTHROPIC_API_KEY` or any other credential.

#### Scenario: Record contents
- **WHEN** a Claude API run is saved
- **THEN** the saved record does not contain the API key
