# Prompt variants: what the baseline wording gets wrong (dev sets only)

Task 6.1 of `openspec/changes/archive/2026-11-02-jev-prompt-variants`, 2026-11-01. This note is the input for the candidate variants (task 6.2). We write the variants from this note and from the dev records it cites, and from nothing else.

## What was read

Only the dev sets of design D7:

| Tier | Dev records | Runs |
|---|---|---|
| 1 | `results/20261018-134025-70runs.jsonl` (Jev, logs, seed 1000) | 70 |
| 2 | `results/tier2/20261012-190447-369b/` (run 1) | 6 |
| 3 | `results/tier3/10.3-combined/` (Tier 3 batch: fault runs and control) | 40 |

We opened none of the confirm-set run records for this note: Tier 1 seed 2000, Tier 2 `20261012-195728-7983`, Tier 4 `6.2-combined`. We had already seen some confirm-set *totals* earlier the same day, in `docs/results-2026-11-01-precision-recall-f1.md` and while checking the comparison report against the published scorecards: Tier 4 Jev 5/22 false-positive runs, and Tier 2 run 2-4 3/3 false positives on the healthy control. No per-run detail from those sets went into this note.

## Findings

### 1. "Right now" is read as "anywhere in the window" (Tier 3)

All 4 Tier 3 false-positive runs are self-healing faults that the scenarios define as "no incident now":

- K002 (cart pod killed, replaced within 15 s), all three runs: incident on polls 1-12 at p = 0.79-0.94, then p ≈ 0.06-0.09 from poll 13.
- K010-2 (one OOM kill of cart, restarted in seconds): p ≈ 0.87-0.91 on polls 1-11, then 0.27-0.29.

The drop comes at the same point in every run, about when the transient errors leave the observation window. Jev applies "is something failing or clearly degraded compared with the baseline" to the whole window, not to the present. The question says "right now", but the `true` criterion ("something is failing or clearly degraded compared with the baseline") has no time in it.

Healthy control K000: p(incident) 0.05-0.20 on all 150 polls. Jev is not jumpy on a quiet cluster.

### 2. Latency noise with no errors is called an incident (Tier 2)

The healthy control F000 got p = 0.79, diagnosis `latency_degradation`. Its observation shows zero errors on every service, but p95 latency up against the baseline window by 1.1 s (frontend-proxy), 1.7 s (frontend) and 4.6 s (checkout, 9.6 s → 14.3 s). Those are normal swings in this demo's p95 between windows. The only real latency fault in the dev set, F005 (frontend-proxy latency), got p = 0.80, about the same. A wording that discounts latency-only changes may lose F005. That is the trade-off to measure, not to tune away.

### 3. The symptom is named instead of the origin (Tiers 2 and 3)

The localization question asks for "the origin of the problem". Jev often names the caller that reports the errors, not the component that failed:

| Run | Observation shows | Expected | Jev named (top probabilities) |
|---|---|---|---|
| K001, K001-2 | cart 0/1 available | cart | frontend 0.56, cart 0.33-0.37 |
| K008 ×3 | cart 0/1 available | cart | frontend 0.64-0.67, cart 0.29-0.34 |
| K007-2, K007-3 | valkey-cart 0/1 available | valkey-cart | frontend-proxy 0.46-0.79 |
| K009 ×3 | 12-13 unhealthy or new pods (node evicted) | jev-worker2 | frontend, recommendation, frontend-proxy |
| F006 | product-catalog slow | product-catalog | frontend-proxy |

The diagnosis follows the wrong component: when Jev names frontend it says `application_failure`, where the expected answer is `service_unavailable`. In all 6 runs where the cart Deployment had no available instances (K001, K008), the observation said so explicitly, and Jev still named frontend in 5. The runs where the failing component's own signal is plain (K003 OOM BackOff, K005 readiness `Unhealthy`) are right 6/6.

### 4. Human review is asked for routine, contained cases (Tier 1)

Human review was wrong in 24 of 70 runs, almost all in one direction:

- `degraded` 10/10 and `hung_worker` 10/10 answered `yes`. The expected answer is `no`: "investigate, a human does not need to be paged yet" and "restart is the standard, low-risk fix ... contained".
- `contradictory` 3/10 answered `no` where `yes` is expected. `healthy` 1/10 answered `yes`.

The question gives no rule for when review is needed. Its criteria are "A human should review it" and "No human review is needed".

### 5. A cause is named when the evidence does not support one (Tier 1)

`ambiguous` and `contradictory` expect `unknown`. Jev named a specific cause in 10/10 and 10/10 of those runs (resource_exhaustion, dependency_failure, configuration, application_bug). In `degraded`, 6/10 said `dependency_failure` where `resource_exhaustion` is expected.

## What the baseline gets right (a variant must not break it)

- Tier 3: K003 (OOM) and K005 (readiness) 6/6 on all four dimensions. Every fault run detected (18/18). K000 quiet on 150/150 polls.
- Tier 2: F001, F003 and F005 right on all four dimensions. F004 right except diagnosis.
- Tier 1: `log_only_errors` 10/10 on every field. `critical` right on the decision 10/10. Severity 63/70 overall.

## What this means for the two candidate variants

- `strict-incident` (incident and human-review wording only; option descriptions unchanged):
  - Tiers 2/3: ask about the state at the end of the observation window. Say that a failure that has already recovered is not an incident now (finding 1). Say that a latency change without errors counts only when it is clearly beyond normal variation (finding 2; watch F005).
  - Tier 1: say when human review is needed, in general terms (finding 4). See the open question below.
- `evidence-options` (option descriptions only; questions' instructions unchanged):
  - Component options: describe "origin" as the component that is itself failing, not one that only reports errors from something it calls (finding 3). Apply the same idea to nodes, via the template.
  - Diagnosis and cause options: describe what each looks like in an observation, and make `unknown` the answer when signals are weak or conflict (finding 5).
- Each variant keeps to general wording. It names no scenario, fault type or service-specific rule (design D9).
- Censoring (design D5): on the Tier 4 confirm set, `strict-incident` can only show a removed false alarm as "censored", because those runs stopped at the first false alarm. A Tier 3 live confirmation batch (task 7.4) would measure it in full.

## Open question for review (task 6.3)

Findings 1 and 4 can be fixed by writing the rule the ground truth uses into the question: "a recovered failure is not an incident now"; "a contained case with a routine next step needs no human". That is fair if the variant states an operating policy a real team would give its tool. It is unfair if it is phrased to fit our scenarios. Two ways to keep it honest:

1. Write the policy in general operational terms only, and report such variants as "policy stated" in the results, separate from variants that only clarify wording.
2. Leave policy out entirely, and only make the existing words unambiguous ("now" = at the end of the window).

**Decided (2026-11-01, by the user): option 1.** Policy text is allowed in general operational terms. Such a variant sets `style: policy`, which is recorded with every run and labelled "policy stated" in the comparison reports (design D10).
