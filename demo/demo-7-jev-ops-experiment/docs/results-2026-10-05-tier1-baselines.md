# Results: Jev vs. three LLMs on Tier 1 (2026-10-05)

This page summarizes the first full comparison run. It covers Tier 1 only (synthetic data). When it was written, Tier 2 (the live OpenTelemetry Demo) had not been run yet. It ran on 2026-10-12 ([results-2026-10-12-tier2-jev.md](results-2026-10-12-tier2-jev.md)).

## What we tested

We generated 70 made-up server situations and asked four engines the same questions about each one:

- Severity: how bad is it? (normal, degraded, high, critical)
- Action: what should happen next? (observe, investigate, restart, escalate)
- Human review: should a person look at this? (yes, no)
- Probable cause: what is the likely cause? This is scored separately and is not part of pass/fail.

There are 7 kinds of situation (scenarios), 10 runs of each. Each engine saw exactly the same data (same seed, 1000) and never saw the scenario name or the expected answer. We wrote the expected answers by hand before the run.

A run passes only if severity, action and human review are all right.

| Engine | What it is | How we called it |
|---|---|---|
| Jev | TypeSafe's ops decision API (model `jev-1.13.0`) | Jev API |
| Claude Sonnet 5.5 | Anthropic general-purpose LLM | `claude -p` command-line tool |
| Claude Opus 5.5 | Anthropic's larger LLM | `claude -p` command-line tool |
| qwen3:8b | small open model, run on a laptop | Ollama, locally |

## Results

| | Jev | Sonnet | Opus | qwen3:8b |
|---|---|---|---|---|
| Runs passed | 37/70 (53%) | 29/70 (41%) | 34/70 (49%) | 30/70 (43%) |
| Likely range (95%) | 41–64% | 31–53% | 37–60% | 32–55% |
| Severity right | 90% | 61% | 93% | 57% |
| Action right | 80% | 80% | 89% | 89% |
| Human review right | 67% | 59% | 64% | 69% |
| Probable cause right | 60% | 56% | 66% | 54% |
| Slowest normal answer (p95) | 0.5 s | 5.0 s | 9.9 s | 35.7 s |
| Cost per decision | unknown | ~$0.009* | ~$0.022* | free (local) |

\* Reported by the `claude -p` tool. On a subscription plan this is a notional figure, not a charge. Jev publishes no per-token price, so its cost is unknown.

Every engine answered all 70 runs. There were no invalid answers and no failed calls.

### Passes by scenario (out of 10)

| Scenario | Jev | Sonnet | Opus | qwen3:8b |
|---|---|---|---|---|
| healthy | 9 | 10 | 10 | 10 |
| degraded | 0 | 3 | 0 | 0 |
| critical | 10 | 10 | 10 | 6 |
| ambiguous | 8 | 0 | 0 | 9 |
| contradictory | 0 | 0 | 0 | 5 |
| hung_worker | 0 | 6 | 10 | 0 |
| log_only_errors | 10 | 0 | 4 | 0 |

## What this means

1. No engine is clearly better. All four get roughly half the runs fully right, and their likely ranges overlap. 70 runs are not enough to rank them.
2. Severity and action are mostly right. Human review is what fails. If we ignore human review, Opus is right on 59 of 70 runs and Jev on 53. For Opus, 25 failed runs were wrong only on human review. For Jev, 16 runs were wrong only on human review.
3. The engines lean in different directions on human review. Jev and qwen ask for a person too often (Jev: 21 unneeded "yes"). Sonnet skips the person when one is needed (27 missed "yes"). Opus errs both ways.
4. Clear cases are easy. The cases in between are hard. Healthy and critical situations are close to 100% for everyone. In the in-between scenarios, each engine has its own blind spots. Jev is the only one that reliably catches errors that show up only in the logs, but it never gets the hung worker right. Opus is the reverse.
5. Jev is about 10 times faster than the next engine.

## Read with care

- The data is synthetic. Real telemetry may behave differently. Tier 2 will test that.
- We wrote the expected answers ourselves. In the "degraded" scenario, Opus and most of Jev's runs answer "high, investigate, yes", while we expect "no" human review. When several engines agree against our answer, that rule deserves a second look. Changing it would change the experiment, so it needs a decision first.
- 10 runs per scenario are not 10 independent cases. They are variations of the same situation. Read scenario scores as a direction, not a precise rate.
- Claude ran through the `claude -p` tool, not the Anthropic API. Its times include the tool's start-up time.
- qwen3 "thinks" before it answers (Ollama's default for this model). It wrote up to about 12,700 characters of reasoning per answer. Ollama does not count these in the output tokens, so qwen's token count (37 per answer) is far too low, and the thinking explains its slow times.
- Only Jev reports how confident it is, so confidence was not compared.

## Test machine (for the local qwen3:8b run)

Jev and Claude ran on their providers' servers, so only their network round trip depends on this machine. qwen3:8b ran entirely on it, so its speed depends on this hardware.

| | |
|---|---|
| Laptop CPU | AMD Ryzen AI 7 350 (8 cores, 16 threads) with Radeon 860M |
| Memory | 30 GiB |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU, 8 GiB, driver 595.84 |
| OS | Pop!_OS 24.04 LTS, Linux 7.1.5 |
| Ollama | 0.22.1 |
| Model | qwen3:8b, 8.2B parameters, Q4_K_M quantization, context 16,384 |
| GPU use | all 37 layers on the GPU (from the Ollama log) |
| Settings | temperature 0, seed = run seed, JSON schema output |

All four batches started at the same time. The Jev and Claude batches finished within about 7 minutes. qwen ran for 25 minutes and had the machine to itself for most of that. Other unrelated work was running on the laptop during the run.

## Source and how to repeat

Figures come from `python report.py --compare` and `python report.py --metrics` on these run files, published with both reports in [sample-run-2026-10-05/](sample-run-2026-10-05/):

- `20261005-224157-70runs.jsonl` (Jev)
- `20261005-224548-70runs-claude-sonnet.jsonl`
- `20261005-224835-70runs-claude-opus.jsonl`
- `20261005-230609-70runs-ollama-qwen3-8b.jsonl`

The human-review breakdown in points 2 and 3 was counted from the `match` and `decision` fields of the same files.

To repeat: `python run.py --runs 70 --seed 1000 --engine <engine>` for `jev`, `claude-sonnet`, `claude-opus` and `ollama:qwen3:8b`. The same seed gives the same input data. The answers can differ between runs, because the hosted models are not fully deterministic.
