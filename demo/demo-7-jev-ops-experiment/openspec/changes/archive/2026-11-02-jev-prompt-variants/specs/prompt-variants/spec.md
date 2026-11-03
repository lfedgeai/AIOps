## ADDED Requirements

### Requirement: Variant files
A prompt variant SHALL be one YAML file at `prompts/<tier>/<name>.yaml` with `name`, `version` (a positive integer), `tier` (`tier1`, `tier2` or `tier3`), `rationale`, `dev_records` (the records it was written against) and `questions`. For each question, `questions` SHALL hold `instructions` and either `criteria` (text for every option) or a per-option `criteria_templates`. Tier 4 SHALL use the `tier3` variants.

#### Scenario: Valid variant loads
- **WHEN** `prompts/tier3/strict-incident.yaml` gives text for every Tier 3 question and option
- **THEN** the loader returns a question dict with the Tier 3 question IDs, types and option keys

#### Scenario: Name does not match the file
- **WHEN** the file `strict-incident.yaml` says `name: other`
- **THEN** loading fails with an error naming the file

### Requirement: Variants change text only
A variant SHALL NOT add, remove or rename questions, change a question's type, or add, remove or rename options. Question IDs, types and option keys SHALL come from the tier's code and vocabularies. The loader SHALL reject a variant that has a missing or extra question, a missing or extra option, or a `type` key. The parsers and the expected outcomes SHALL NOT change.

#### Scenario: Extra option
- **WHEN** a Tier 3 variant gives criteria for an action `drain_node` that is not in `ACTIONS`
- **THEN** loading fails and names `drain_node`

#### Scenario: Missing option text
- **WHEN** a Tier 1 variant has no description for the severity option `critical`
- **THEN** loading fails and names `severity` and `critical`

#### Scenario: Type override
- **WHEN** a variant sets `type: score` on a question
- **THEN** loading fails

### Requirement: Baseline reproduces today's questions
Each tier SHALL have a `baseline` variant, version 1. Built from it, the question dict SHALL equal the question set used before this change, byte for byte as canonical JSON. The tier's `QUESTIONS` constant SHALL be built from `baseline`.

#### Scenario: Equality check
- **WHEN** the tests build `baseline` for each tier
- **THEN** each dict equals the frozen pre-change question set exactly

### Requirement: Selecting a variant
`run.py`, `python -m tier2`, `python -m tier3` and `python -m tier4` SHALL accept `--prompt <name>`, defaulting to `baseline`. An unknown name SHALL stop the command before any run starts, and the error SHALL list the available variants for that tier.

#### Scenario: Unknown variant
- **WHEN** `./run-k8s-experiment K001 --prompt nope` runs
- **THEN** it exits with an error listing the Tier 3 variants, and no fault is injected and no engine is called

### Requirement: One variant for every engine
Within one command, every engine SHALL be asked the selected variant's questions: Jev, the Claude and Ollama Tier 1 baselines, and the Tier 3/4 Claude baseline. No engine module SHALL import a fixed question constant for building its request or prompt.

#### Scenario: Claude baseline with a variant
- **WHEN** `run.py --engine claude-opus --prompt strict-incident` runs
- **THEN** the Claude prompt contains the strict-incident instructions and option descriptions, and the record names `strict-incident`

### Requirement: Variant recorded in every record
Every run record, Tier 1 JSONL line, summary and replay record SHALL hold `prompt: {name, version, sha256}`, where `sha256` is the hash of the canonical JSON of the questions sent. A record without `prompt` SHALL be read as `baseline` version 1, and reports SHALL say that they did this. The hash SHALL NOT include the observation.

#### Scenario: Old record
- **WHEN** the report reads a Tier 3 record from 2026-10-18 without `prompt`
- **THEN** it groups the record under `baseline` v1 and notes "no prompt field: read as baseline"

#### Scenario: Edited without a version bump
- **WHEN** two records say `strict-incident` v1 but their hashes differ
- **THEN** the comparison report fails and names both hashes

### Requirement: Wording and policy-stated variants
A variant file MAY set `style` to `wording` (the default) or `policy`. Any other value SHALL fail to load. `style` SHALL be recorded in `prompt` on every record. A record without it SHALL be read as `wording`. The comparison reports SHALL label policy-stated columns. A policy-stated variant SHALL state its rule in general operational terms only, under the same leak rules as every variant.

#### Scenario: Policy variant in a comparison
- **WHEN** replay records of a `style: policy` variant are compared
- **THEN** its column label ends with "policy stated"

#### Scenario: Unknown style
- **WHEN** a variant sets `style: clever`
- **THEN** loading fails and lists `wording` and `policy`

### Requirement: Variant text passes the leak rules
No variant text SHALL contain a scenario name or ID, a fault type name, `jev-fault`, `flagd`, `feature_flag`, `ground_truth`, `expected`, `scenario` or `experiment`. Every existing request and prompt leak test SHALL run once per variant file.

#### Scenario: Variant names a fault
- **WHEN** a variant's text contains `scale_zero`
- **THEN** the leak test fails and names the variant and the term

#### Scenario: Whole-path leak test per variant
- **WHEN** the Tier 3 runner leak test runs
- **THEN** it runs every scenario under every Tier 3 variant and finds no ground truth in any request
