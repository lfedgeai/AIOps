# Summary: can Jev make bounded operational decisions from observations? (2026-11-02)

This page sums up the whole experiment, from 2026-09-30 to 2026-11-02. It is for [lfedgeai/AIOps#41](https://github.com/lfedgeai/AIOps/issues/41) and for anyone who wants the findings without reading every results page. It reports no new runs. Each figure is copied from the linked results page, and that page names the run records the figure came from.

## What was tested

Jev is the AI engine that this experiment tests. Jev gets an observation of a system: a snapshot of its metrics, logs, traces and cluster state. It answers fixed multiple-choice questions: is there an incident, which component, what kind of fault, and what action to take. We wrote the expected answers by hand, and Jev never sees them. The experiment has four tiers, and each tier is closer to a real system than the one before:

| Tier | Data | What is scored |
|---|---|---|
| 1 | Synthetic server state, 7 scenarios × 10 runs | severity + action + human review (PASS/FAIL); probable cause separately |
| 2 | OpenTelemetry Demo on Docker Compose, faults set by feature flag | detection, localization, diagnosis, action, each on its own |
| 3 | The demo on a local kind cluster, the LF Edge group's 10 Kubernetes faults plus a healthy control | the same four, plus time to detect (MTTD) |
| 4 | Tier 3, plus the harness carrying out Jev's action once from a fixed allowlist | the same, plus recovery and MTTR |

Some terms in the table:

- Detection: saying that there is an incident.
- Localization: naming the component that failed. Some rows below use the LF Edge group's name for it, RCA (root cause analysis).
- Diagnosis: naming the kind of fault.
- Healthy control: a run with no fault, where the right answer is "no incident".
- kind: a tool that runs a small Kubernetes cluster on one machine.
- Harness: the test program that injects each fault, asks Jev and records the answers.
- Allowlist: the short, fixed list of actions that the harness is allowed to carry out.
- MTTR (time to recovery): here, verified MTTR is the time from the fault until the service checks pass and keep passing to the end of the watch, after the one action. It counts only runs that the action fixed, and the tables give the median.

The baselines (other engines that answer the same questions, for comparison) are Claude Haiku, Sonnet and Opus 5.5 (through the Anthropic API), and qwen3:8b through Ollama on Tier 1.

## Short answer

Jev is good at noticing that something is wrong. It is fast and cheap. It is weaker at saying where the problem is, and that matters most once the action is carried out.

- Detection is strong. Jev detected every fault run at Tiers 2, 3 and 4 (20/20, 18/18, 18/18). It answers in about 0.4-1 s.
- Localization is the weak spot. At Tiers 3/4 Jev names the right component on 8-10 of 18 fault runs. Opus gets 18/18 on the same observations. Jev usually names the caller that logs the errors (`frontend`) instead of the service that failed (`cart`, `valkey-cart`) or the node (K009, 0/3 in every batch).
- That gap decides recovery. At Tier 4, Jev's actions fixed 3 of 18 faults, and Opus's actions fixed 11 of 18 (live, 2026-11-02). Both chose a correct type of action every time. The difference is which component they acted on.
- False alarms come from two scenarios. A false alarm (a false positive) is an incident reported when there is none. A single pod restart that has already recovered (K002) is called an incident on every run, and a full volume (K010) on some runs. The healthy control is clean at Tiers 3/4 (0/10). Opus also raises these two, and one more (K006).
- Nothing we tried fixed the weak spots. We tried other prompt wording, other incident thresholds and an observation layout that states each row's facts in plain words. None of them moved the false alarms. The facts layout gained 2 localization runs (10 → 12 of 18), which is small.
- On the Tier 1 decision, no engine is clearly ahead. Jev, Haiku, Sonnet and Opus all pass 43-53% of runs, and their 95% confidence intervals (the range that likely holds each engine's true pass rate) overlap.

## By tier

In this table, precision is the share of reported incidents that were real. F1 is one score that combines precision with recall (the share of real faults that were detected).

| | Jev | Best Claude baseline | Page |
|---|---|---|---|
| **Tier 1** PASS (seed 1000 / 2000, API baselines) | 36/70, 33/70 | Sonnet 34/70, 37/70 | [llm-api-baselines](results-2026-11-01-llm-api-baselines.md) |
| Tier 1 without logs | 31/70 (37/70 with logs) | — | [tier1-jev](results-2026-10-18-tier1-jev.md) |
| **Tier 2** detection / localization / diagnosis / action (20 fault runs) | 20 / 15 / 13 / 19 | — | [tier2-jev](results-2026-10-12-tier2-jev.md) |
| Tier 2 healthy control | incident reported on 4/4 | — | same |
| **Tier 3** detect / RCA / diagnosis (18 fault runs) | 18 / 8 / 10 | — | [tier3-jev](results-2026-10-18-tier3-jev.md) |
| Tier 3 false-positive runs (22 no-incident) | 4/22 (precision 18/22) | — | same |
| **Tier 4 live, 2026-11-02** detect / RCA / diagnosis | 18 / 9 / 9 | Opus 18 / 18 / 18 | [tier4-jev-vs-opus-api](results-2026-11-02-tier4-jev-vs-opus-api.md) |
| Tier 4 false-positive runs, detection F1 | 4/22, 0.90 | Opus 7/22, 0.84 | same |
| Tier 4 recovered by action | 3/18 | Opus 11/18 | same |
| Tier 4 MTTD median | 16.6 s | Opus 18.7 s | same |
| Tier 4 verified MTTR median | 106.2 s (n=3) | Opus 122.7 s (n=11) | same |

Precision, recall and F1 for every tier: [precision-recall-f1](results-2026-11-01-precision-recall-f1.md).

## Where Jev goes wrong, and what we tried

A prompt variant is a different wording of the same questions. The held-out set is a set of runs that we did not use to write the variants or the layout.

| Problem | Where | What we tried | Result |
|---|---|---|---|
| Names the caller, not the failed service | K001, K007, K008 (cart / valkey-cart scaled to zero) | 2 prompt variants, `evidence-options` and `strict-incident` ([prompt-variants](results-2026-11-01-prompt-variants.md)) | No variant clearly beats the baseline on the held-out set |
| | | `facts` observation layout ([observation-facts](results-2026-11-02-observation-facts.md)) | +2 runs on held-out (K001, K008); K007 unchanged |
| Never names the tainted node | K009 | the same | 0/3 throughout |
| Calls a recovered pod restart an incident | K002 3/3, K010 some runs | incident-probability cutoff sweep ([incident-threshold](results-2026-11-02-incident-threshold.md)) | False alarms (p 0.75-0.94) sit inside the range of true detections (p 0.68-0.95); no cutoff separates them |
| | | `strict-incident` prompt, `facts` layout | No change beyond noise |
| Leans to the first option listed | action question | reversed option order ([option-order](results-2026-11-02-option-order.md)) | 33 of 34 changed action answers moved to an earlier option (5% of answers); no headline score moved |
| Does not hold back when signals disagree | Tier 1 `contradictory` | — | 0/10: it picks one side (`observe` or `restart`) instead of `investigate` |

## Cost and speed

Jev answers about 5 to 7 times faster than Opus through the Anthropic API. In the live Tier 4 batch, Jev's median answer time was 384 ms (p95 499 ms), and Opus's was 2,157 ms (p95 4,999 ms) ([tier4-jev-vs-opus-api](results-2026-11-02-tier4-jev-vs-opus-api.md)). On Tier 1, the median round trips were 352 ms for Jev and 2,540 ms for Opus ([llm-api-baselines](results-2026-11-01-llm-api-baselines.md)). The local model, qwen3:8b, is much slower (p95 35.7 s on Tier 1, [tier1-baselines](results-2026-10-05-tier1-baselines.md)).

In the same Tier 4 batch, Jev used 1.27M input and 114k output tokens over 295 calls, and Opus used 1.47M input and 17k output tokens over 250 calls. Jev's cost was about $0.05, estimated from TypeSafe's published price (`JEV_PRICE_*`, [jev-api-notes](jev-api-notes.md)). At Anthropic's published price for Opus 5.5 ($4 input, $20 output per million tokens, set as `CLAUDE_OPUS_PRICE_*`), the same batch cost about $6.21 for Opus, about 117 times Jev's cost. On Tier 1, 70 runs cost about $0.69 for Opus and $0.0043 for Jev (seed 1000). The records themselves keep Opus's cost as `n/a`, because the API reports no cost. These figures are estimates from the recorded token counts.

## Limits

- The samples are small: 18 fault runs per Tier 3/4 batch, 3 per scenario. A change of 2-3 runs is within noise.
- The experiment uses one application (the OpenTelemetry Demo), one cluster layout, and ten fault types.
- Tier 4 MTTR is single-shot: one answer, carried out once, with no feedback or retry. It is not operational MTTR.
- The expected answers are our own judgement. We wrote them down with a rationale for each scenario. Where Jev disagrees on a borderline case (Tier 1 `ambiguous`, `contradictory`), a person could reasonably side with Jev.
- Jev changed during the experiment (`jev-1.13.0` → `jev-1.14.0`). Each page says which version answered.

## What would help next

These are not planned work. They are where the evidence points:

1. Localization across a call edge (one service calling another). Jev's errors share one pattern: the failing service sits one hop behind the service that shows the errors. That is a model question for TypeSafe. Prompt and layout changes did not fix it here.
2. "Has it already recovered?" K002 needs the engine to see that the failure is over. The observation window mixes the time before and after the recovery. We designed a layout with separate recent and earlier windows (`time-split`) but did not build it, because the `facts` results did not meet its gate (the result it had to reach first).
3. More runs per scenario before reading any 2-3 run difference.

## Reproduce

Everything runs from this repository. The offline tests need no key (`./run-all.sh --check`). Live runs need `JEV_API_KEY`, and Tiers 3/4 need the kind testbed (`./setup-k8s.sh`). The commands are in the [README](../README.md). The specs are in `openspec/specs/`, and the history of each change is in `openspec/changes/archive/`.
