## ADDED Requirements

### Requirement: Engine selection
`run.py` SHALL accept `--engine` with the values `jev` (default), `claude-sonnet`, `claude-opus` and `ollama:<model>`. With the default, behaviour and output file names SHALL be unchanged from before this change. `JEV_API_KEY` SHALL be required only for the `jev` engine.

#### Scenario: Default engine
- **WHEN** `run.py` runs without `--engine`
- **THEN** it uses Jev and writes the same file names as before

#### Scenario: Baseline without a Jev key
- **WHEN** `run.py --engine ollama:qwen3:8b` runs and `JEV_API_KEY` is not set
- **THEN** the batch runs and no Jev key is asked for

#### Scenario: Unknown engine
- **WHEN** `--engine` names an unknown engine, or `ollama:` without a model
- **THEN** argument parsing fails with the list of valid engines

### Requirement: Engine identity in results
Every run record and batch summary SHALL include the engine name, transport, model requested and model that answered. Result file names for non-Jev engines SHALL include the engine name. A record without engine information SHALL be read as a Jev record.

#### Scenario: Old result file
- **WHEN** a result file written before this change is loaded
- **THEN** its records are treated as engine `jev`

### Requirement: Same scoring for every engine
All engines SHALL be scored by the same comparison and aggregation functions against the same hand-written expected outcomes. Probable cause SHALL stay outside PASS/FAIL for every engine.

#### Scenario: Shared scoring
- **WHEN** a Claude run and a Jev run return the same decision for the same scenario
- **THEN** both get the same per-field and overall result

### Requirement: Comparison report
`report.py --compare` SHALL take two or more result files and render, per engine, overall accuracy, accuracy per decision, probable-cause accuracy, accuracy per scenario, invalid and error counts, transport, and telemetry (tokens, latency, cost or `n/a`). Every figure SHALL be computed from the run records.

#### Scenario: Side by side
- **WHEN** a Jev file and a Claude file are compared
- **THEN** the report shows both engines' figures in the same tables

#### Scenario: Mismatched inputs
- **WHEN** the compared files do not cover the same scenario and seed pairs, or mix runs with and without logs
- **THEN** the report starts with a warning naming the difference

#### Scenario: CLI transport flagged
- **WHEN** any compared file was produced through the `claude -p` fallback
- **THEN** the report says so next to that engine's figures
