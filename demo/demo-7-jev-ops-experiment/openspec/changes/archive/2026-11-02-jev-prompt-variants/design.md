## Context

Jev is not prompted with free text. Each request is `state` (the observation) plus `questions`, a map of question IDs to `{type, instructions, criteria}`. The `criteria` give a description for each option (`docs/jev-api-notes.md`). Every engine is asked one fixed set of questions in each tier:

| Tier | Question set | Engines that use it |
|---|---|---|
| 1 | `jev/client.py` `QUESTIONS` (severity, action, probable_cause, human_review) | Jev; `llm/prompt.py` builds the Claude and Ollama prompts from it |
| 2 | `tier2/jev_adapter.py` `QUESTIONS` | Jev |
| 3/4 | `tier3/questions.py` `QUESTIONS` | Jev (`tier3/jev_adapter.py`) and the Claude Opus baseline (`tier3/claude_engine.py`) |

The parsers take answers only from fixed vocabularies (`tier2/vocab.py`, `tier3/vocab.py`, the Tier 1 option sets). The expected outcomes are written by hand against those vocabularies. Tier 2 and Tier 3 store exactly what the engine received: `request` in a Tier 2 record and `polls[].request` in a Tier 3/4 record, after scrubbing and with no ground truth.

Constraints from `CLAUDE.md` that this change must keep: expected outcomes are hand-written and never depend on the engine; the engine sees only observations; every engine gets the same questions; out-of-set replies are `invalid`; nothing is invented; Jev HTTP code stays in `jev/client.py`; Tier 2 and Tier 3 adapters are their tier's only `jev.client` importers; tests run offline.

## Goals / Non-Goals

**Goals:**
- Switch how Jev is asked by name (`--prompt strict-incident`) in every tier, without editing code.
- Know from every run record exactly which wording produced it.
- Compare variants on identical observations and print one table: `baseline 27/30 (90%)`, `strict-incident 9/30 (30%)`.
- Guard against tuning a variant to the scenarios it is scored on.
- Leave every earlier result valid as `baseline`.

**Non-Goals:**
- Changing vocabularies, option keys, question types, question IDs, parsers or expected outcomes. A variant that needs new options is a change to the experiment, done separately.
- Splitting questions across requests, for example "incident first, then diagnosis only on yes". Filling the unasked answers would mean inventing them. A later change can take this up with its own scoring rule.
- Changing the observation (what goes into `state`), the incident threshold (`INCIDENT_THRESHOLD = 0.5`) or the polling. Each is its own experiment.
- Automatic prompt search or an LLM writing variants. Variants are hand-written, like expected outcomes.

## Decisions

### D1. Variants are YAML data, not code

One file per variant: `prompts/<tier>/<name>.yaml`.

```yaml
name: strict-incident
version: 1
tier: tier3
rationale: >
  Tiers 2-4 show false alarms on healthy runs. Ask for a clear,
  user-facing change against the baseline window before calling an incident.
questions:
  incident_detected:
    instructions: "Compared with the baseline window, is there a clear incident ..."
    criteria:
      "true": "..."
      "false": "..."
  affected_component:
    instructions: "..."
    criteria: {none: "Nothing is affected."}
    criteria_templates:                      # one text per named group of options
      services: "The problem originates in {option}."
      nodes: "The problem originates in node {option}."
  diagnosis: {instructions: "...", criteria: {none: "...", ...}}
  recommended_action: {instructions: "...", criteria: {...}}
```

`prompts/loader.py` reads a variant and builds the question dict for the tier. The tier's code still supplies the question IDs, types and option keys, from the vocabularies. The YAML supplies only text. The loader raises an error when the file is missing a question, has an unknown question, lacks text for an option, has an option outside the vocabulary, or tries to set `type`. A per-option template is allowed because Tier 3's component list is long and generated from `tier3/vocab.py` (nodes get their own template).

*Alternative considered:* Python modules per variant. Rejected because editing data is easier and safer, and the loader can validate data against the vocabulary at load time.

### D2. `baseline` reproduces today's questions exactly

`baseline.yaml` holds today's text for each tier. The `QUESTIONS` constants become `load("baseline")` for their tier. A test compares the built dict with a frozen copy of the current dict (JSON with sorted keys, compared byte for byte). Records without a `prompt` field count as `baseline` v1, and the report says so.

### D3. Identity: name, version, hash

Every run record and every replay record gets:

```json
"prompt": {"name": "strict-incident", "version": 1, "sha256": "<hash of the built questions, canonical JSON>"}
```

The hash covers the questions actually sent, so editing a file without bumping `version` shows up. The comparison report refuses to combine records that have the same name and version but different hashes. Tier 1 JSONL lines and summaries carry the same field.

### D4. One variant for every engine in a run

`--prompt` selects the variant once per process. The Jev adapters pass the built questions to `JevClient.build_request(questions=...)`, which already exists. `llm/prompt.py` and `tier3/claude_engine.py` take the question dict as an argument instead of importing the constant. So, within a variant, Jev and the baselines are asked the same thing, and the existing rule holds per variant. The engine `info` and the record show the variant.

### D5. Replay for Tiers 2, 3 and 4

Live Tier 2/3 runs never see the same observation twice, so two live batches mix variant effects with run-to-run noise. Replay removes that noise:

- `python -m tier3 replay <records-dir> --prompt <variant> [--engine jev|claude-opus]` loads each stored record (Tier 3 or Tier 4) and sends each stored `polls[i].request` to the engine with the variant's questions, in order. It follows the same rule as `poll` mode: the first detecting poll is scored, else the last stored poll. Then it scores the four verdicts with `tier3/evaluate.py` against the record's stored `ground_truth`.
- `python -m tier2 replay` does the same for the single stored `request`.
- Replay reads `request` only to send it, and reads `ground_truth` only to score. The two code paths are separate. A test runs every fixture record through replay and checks that no request contains anything outside the original `request` plus the variant's questions.
- Replay writes new records under `results/<tier>/replay/<run-id>/` with `"mode": "replay"`, `source` (original record path and its `run_id`), and `prompt`. It never overwrites the originals.

**Censoring.** In live `poll` mode, polling stops at the first detecting poll for every Tier 3 fault run and for every Tier 4 run, no-incident runs included (Tier 4's `stop_at_first_detection`). Such a record stores only the polls up to that detection, often one. A source is *truncated* when its first detecting poll is its last stored poll. If a variant reports no incident on any stored poll of a truncated source, we cannot know what it would have said later. The run is then `censored`. A censored fault run counts as not detected on the stored polls and is never reported as a confirmed miss. A censored no-incident run counts as no false positive on the stored polls and is never reported as a confirmed true negative. The comparison shows the censored count next to recall and precision. Tier 3 no-incident runs and the K000 control poll their whole window, so false positives on the Tier 3 dev set are measured in full. On the Tier 4 confirm set, a variant that stays quiet where the baseline raised a false alarm is censored on that run. (2026-11-01, during task 4: the first draft said all no-incident runs store every poll; that holds for Tier 3 only.)

**What replay cannot measure.** MTTD as wall-clock time, and anything about remediation. `mttd_seconds` is `null` in replay records. Replay records instead hold `detect_poll` and that poll's original `seconds_since_inject`, labelled `replay_detect_seconds_since_inject`. These are never mixed with live MTTD.

*Alternative considered:* only live runs, interleaving variants within one batch. That costs cluster hours per variant and still compares different observations. Live runs stay possible (`--prompt` on `run`/`batch`), for confirming a winner.

### D6. Comparison report

- Tier 1: `python report.py --compare A.jsonl B.jsonl` already compares files. Its column label becomes `engine / prompt`, for example `jev / strict-incident`. A new `--by-prompt` option groups records by prompt instead of by file.
- Tiers 2/3/4: `python -m tier3 compare <dir> [<dir> ...]` (also `python -m tier2 compare`) groups records by `(engine, prompt name, version)`. It prints one column per group and one row per figure: detect, RCA, diagnosis, remediation correct, FP runs, FP polls, detection precision, recall and F1, censored runs, invalid replies, engine calls. Each cell is `k/n (p%)`. The percentage always comes with its count; there are no bare percentages.
- Groups are compared only on the same source records. If two groups did not replay the same set of `source` run IDs, the report prints a warning and the run IDs that differ, and it shows only figures over the common set.

### D7. Dev and confirm sets

A variant is written while looking at the dev set and judged on the confirm set:

| Tier | Dev | Confirm |
|---|---|---|
| 1 | seed 1000 (existing batches) | seed 2000 (new batches, same 70-run design) |
| 2 | `results/tier2/20261012-190447-369b/` | `results/tier2/20261012-195728-7983/` |
| 3/4 | Tier 3 batch `10.3-combined` | Tier 4 Jev batch `6.2-combined` (same faults, observed separately) |

The comparison report takes `--dev` and `--confirm` directories and prints both, confirm first. A variant's file states which dev records it was written against. Tier 2's split is small (6 and 18 runs). The report says so.

### D8. Starting variants

These are hand-written per tier, each with a rationale in its file:

- `baseline`: today's text.
- `strict-incident`: the incident (Tiers 2/3) and human-review (Tier 1) questions require a clear change against the baseline window that affects users or availability. Small fluctuations are described as normal. Aimed at the false alarms in `docs/results-2026-11-01-precision-recall-f1.md`.
- `evidence-options`: option descriptions say what would be seen in the observation if that option were right (as written in task 6.2: the component, diagnosis, severity and cause options; the action and human-review texts stay the baseline's, so a change in action scores cannot come from reworded actions). For example, `scale_up`: "the service has fewer available instances than its baseline". The text stays general. It names no scenario, fault or experiment term (D9).

More variants are added the same way. Each new one needs a rationale and a version.

### D10. Wording variants and policy-stated variants

Decided by the user on 2026-11-01 (task 6.1, `docs/prompt-variants-dev-notes.md`). Some baseline errors come from the question leaving out the rule the expected answers follow, for example "a failure that has already recovered is not an incident now". A variant MAY state such a rule. It must be in general operational terms that a real team could give its tool. It must name no scenario, fault type or service, and it must hold no rule tied to one case (D9). Each variant file declares `style`:

- `wording` (default): only makes the existing questions and options clearer.
- `policy`: also states the operating policy behind the expected answers.

`style` is part of the recorded identity (`prompt.style`). The comparison reports mark policy-stated columns ("policy stated"), and results pages report the two kinds separately. A gain from a policy-stated variant shows that Jev can follow a stated policy. It does not show that Jev reads the situation better.

### D9. Leak and hint rules for variant text

Variant text is sent to the engine, so the existing leak rules apply to it. A test loads every variant file and fails on scenario names and IDs (Tier 1 names, `F00x`, `K00x`), the ten fault type names, `jev-fault`, `flagd`, `feature_flag`, `ground_truth`, `expected`, `scenario` and `experiment`. The existing whole-path leak tests (`test_request_does_not_contain_scenario_or_expected`, the Tier 2/3/4 runner leak tests, `test_prompt_does_not_leak_scenario_or_expected`) are parametrised over every variant. A variant may describe what an option looks like in general terms. It may not contain rules tied to a scenario, such as "if cart has 0 replicas, answer scale_up". That cannot be fully tested, so it is a review rule in `CLAUDE.md`.

## Risks / Trade-offs

- [Variants tuned to the scored scenarios] → dev/confirm split (D7); the headline is the confirm figure; variants are hand-written with a rationale; the scenario-specific hint ban (D9).
- [Replay censoring makes recall look worse for stricter variants] → censored count shown beside recall; the false-positive comparison is complete; a live confirm batch for any winner.
- [Small samples (Tier 2 has 4 healthy runs; Tier 3/4 22 negatives)] → counts with denominators everywhere; the existing Wilson intervals on Tier 1; no claims from a one-run difference.
- [Refactoring `QUESTIONS` changes what is sent] → byte-equality test for `baseline` (D2); hash in every record (D3).
- [Jev drift between dev runs (October) and replay (now)] → replay `baseline` again alongside each variant, and compare the variant with the replayed baseline, not the stored one. The report shows both baselines when they differ.
- [Cost of replay calls] → about 70 calls per Tier 1 batch and about 351 or 280 per Tier 3/4 replay per variant. Cost is shown only from reported usage or user-set prices, as now.

## Migration Plan

1. Add the loader and `baseline` files. Switch the constants to them. The equality tests pass, so nothing sent changes.
2. Add `--prompt` and the `prompt` record field. Old records read as `baseline`.
3. Add replay and compare. Add the two candidate variants per tier.
4. Run: replay `baseline` and the candidates on the dev sets, then on the confirm sets. Write a results page.

Rollback: `--prompt baseline` (the default) is identical to today. Removing the change leaves old records readable.

## Open Questions

- Does Jev's reply change with `criteria` wording alone, or mostly with `instructions`? The first replay will show it; it may decide which variants are worth adding next.
- Should the Tier 3/4 Claude baseline be replayed too? It costs `claude -p` or API calls. The default plan replays Jev only and leaves Claude as an option (`--engine claude-opus`).
- Tier 1 confirm seed 2000 needs fresh Jev and baseline-engine runs, so Tier 1's stored seed-1000 runs are dev only. Is 70 runs per variant on seed 2000 enough, or should the confirm batch be 140?
