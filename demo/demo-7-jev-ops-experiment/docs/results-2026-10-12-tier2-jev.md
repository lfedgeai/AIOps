# Results: Jev on Tier 2, the live OpenTelemetry Demo (2026-10-12)

This page summarizes the first live Tier 2 runs. Tier 2 injects a known fault into a running copy of the OpenTelemetry Demo, a small online shop made of about 20 services. Then it asks Jev what is wrong and scores the answer against a hidden expected answer. Only Jev ran on Tier 2. The Tier 1 summary is in [results-2026-10-05-tier1-baselines.md](results-2026-10-05-tier1-baselines.md).

## What ran

There are six scenarios. Each scenario ran four times, for 24 runs in total. All 24 runs finished with status `ok`. No run was `contaminated` (the system did not come back to normal after the fault was removed), and no Jev call failed.

| | |
|---|---|
| Jev model | `jev-1.13.0` |
| Demo | OpenTelemetry Demo 3.1.0, Docker Compose, one laptop (30 GiB memory, Docker 29.8.2) |
| Load | Locust, 10 simulated users |
| Code | commit `3472c93` |
| Run 1 | `results/tier2/20261012-190447-369b/`, one pass of all six scenarios |
| Runs 2 to 4 | `results/tier2/20261012-195728-7983/`, `./run-all-experiments --repeats 3` |

## How a run works

The harness is the code in `tier2/` that runs the experiment. A flag is a switch in flagd, the feature-flag service of the demo. Each fault is one demo flag that the demo authors built for fault tests. The harness does these steps for each run:

1. It makes sure that the demo is healthy, the load is at 10 users, and no flag is on.
2. It waits 180 seconds for the load to settle. The 180 seconds before the fault become the baseline (the normal state for comparison).
3. It turns the fault flag on. The healthy control, F000, turns no flag on.
4. It polls the metrics every 15 seconds for up to 150 seconds and records when the fault becomes visible.
5. At 240 seconds after injection, it collects the last 180 seconds of telemetry from Prometheus, Jaeger and OpenSearch.
6. It builds an observation from that telemetry and sends it to Jev once.
7. It scores Jev's answer against the expected answer.
8. It turns the flag off in a `finally` block, so this step always runs. It then waits until the services are back at the baseline. If they do not come back within 300 seconds, the run is `contaminated` and the batch stops.

The harness only records Jev's answer. It never acts on it. A run takes about 9 minutes, and a healthy-control run takes about 7 minutes.

## The scenarios

We wrote the expected answers by hand before the runs, from measurements on the demo. The reasons for each answer are in `tier2/scenarios/*.yaml`. The flag tests are in [otel-demo-notes.md](otel-demo-notes.md).

| ID | Flag | What goes wrong | Expected service | Expected diagnosis | Accepted actions |
|---|---|---|---|---|---|
| F000 | none | Nothing. Healthy control. | none | none | `no_action` |
| F001 | `productCatalogFailure` | product-catalog fails for one product only | product-catalog | `application_failure` | investigate, rollback_change, restart_service |
| F003 | `paymentFailure` 100% | payment rejects every charge | payment | `application_failure` | investigate, rollback_change, restart_service |
| F004 | `paymentUnreachable` | checkout dials a wrong address for payment. Payment gets no calls and shows no errors. | checkout | `dependency_unreachable` | investigate, rollback_change, escalate |
| F005 | `imageSlowLoad` 10 s | the proxy delays image requests by 10 s. There are no errors. | frontend-proxy | `latency_degradation` | investigate, rollback_change |
| F006 | `productCatalogLockContention` | a database lock makes product-catalog queries wait and time out | product-catalog | `database_contention` | investigate, rollback_change, escalate |

F002 does not exist. Its flag had no visible effect, so we removed it and did not reuse the ID.

## What Jev sees

The observation is a JSON document with the same shape in every run. It contains these parts:

- The time window and the baseline window.
- Up to 12 services. Each service has its request rate, error rate, p50 and p95 latency, the same values for the baseline, and the change between them.
- Up to 20 calls between services (from, to, calls, errors).
- Error logs: up to 3 lines for each of up to 8 services, with each line cut at 240 characters and repeated lines counted.
- Up to 5 failing traces, with the path of services and the spans that failed.

The full observation has a limit of 60,000 bytes. The real requests were 4,356 to 11,249 bytes long (3,096 to 5,636 input tokens).

The observation contains no scenario name, flag name or expected answer. A scrubber removes flag data from spans and logs. The collectors also skip the services that show flag activity: flagd, flagd-ui, telemetry-docs and load-generator. We searched all 24 stored requests for the 18 flag names of the demo and for `feature_flag`, `flagd` and `ground_truth`, and found none. The API key is not in any result file.

Jev answers four questions with fixed answer sets:

| Question | Answer set |
|---|---|
| Is there an incident now? | probability from 0 to 1. 0.5 or more counts as yes. |
| Which service is the origin? | `none` or one of 18 services |
| What is the failure mode? | 7 diagnoses, for example `application_failure` or `dependency_unreachable` |
| What is the next action? | 6 actions, for example `investigate`, `escalate` or `no_action` |

## How we score

Each of the four answers gets its own verdict: PASS, FAIL or UNKNOWN. Detection, localization and diagnosis pass only on an exact match. Action passes if the action is in the accepted list for that scenario. UNKNOWN is only for a failed call, an invalid reply or a missing field, and no run got UNKNOWN. We do not combine the four verdicts into one score. A false positive is a healthy-control run in which Jev reports an incident.

## Results

The table shows passes out of 4 runs for each scenario.

| Scenario | Detection | Localization | Diagnosis | Action |
|---|---|---|---|---|
| F000 healthy control | 0/4 | 2/4 | 1/4 | 0/4 |
| F001 product-catalog | 4/4 | 3/4 | 4/4 | 4/4 |
| F003 payment | 4/4 | 4/4 | 4/4 | 3/4 |
| F004 checkout to payment | 4/4 | 4/4 | 0/4 | 4/4 |
| F005 slow images | 4/4 | 4/4 | 4/4 | 4/4 |
| F006 database lock | 4/4 | 0/4 | 1/4 | 4/4 |
| All 20 fault runs | 20/20 | 15/20 | 13/20 | 19/20 |

Every cell counts PASS verdicts. For the healthy control F000, a detection PASS means "no incident reported", so 0/4 means Jev reported an incident in all four healthy runs.

The wrong answers were these:

| Scenario | Runs | Jev answered | Expected |
|---|---|---|---|
| F000 | 4 of 4 | incident, action `investigate` | no incident, `no_action` |
| F000 | 2 of 4 | service `checkout` or `frontend-proxy`, diagnosis `latency_degradation` | none |
| F000 | 1 of 4 | service `none`, diagnosis `latency_degradation` | none |
| F001 | 1 of 4 | service `frontend` | product-catalog |
| F003 | 1 of 4 | action `escalate` | investigate, rollback_change or restart_service |
| F004 | 4 of 4 | diagnosis `application_failure` | `dependency_unreachable` |
| F006 | 3 of 4 | service `frontend-proxy`, diagnosis `resource_exhaustion` | product-catalog, `database_contention` |
| F006 | 1 of 4 | service `postgresql`, diagnosis `database_contention` | product-catalog |

Jev took 0.46 to 0.88 seconds to answer. Each answer used 340 to 346 output tokens. Jev reports no cost and we set no price, so the cost is unknown.

## What this means

Jev saw every fault. It detected an incident in all 20 fault runs, with an incident probability of 0.80 to 0.97.

Jev also reported an incident in every healthy run. In the 4 healthy-control runs, the probability was 0.59 to 0.88, and the action was always `investigate`. On this testbed, an incident report from Jev therefore does not separate a fault from normal operation. The limits section gives a likely reason.

Jev named the right service when the error starts in the failing service: F001 (3 of 4), F003, F004 and F005. It named the wrong service when a database under the service causes the problem (F006).

The F004 diagnosis was wrong in all four runs. Jev found checkout every time but called the problem an application failure, not an unreachable dependency.

Four runs for each scenario are too few to give a rate. Read the results as a direction.

## Example result

This is run 1 of the repeat batch for F004 (`results/tier2/20261012-195728-7983/F004.json`). Jev received this telemetry:

- checkout had an error rate of 1.0. Its baseline error rate was 0.
- frontend made 19 calls to checkout, and all 19 failed.
- frontend logged "Checkout failed to place order" 20 times.
- All 5 failing traces had a failed `POST /api/checkout` span from frontend, with HTTP status 500.

Jev answered this:

| Question | Answer | Probabilities | Verdict |
|---|---|---|---|
| Incident | yes | 0.97 | PASS |
| Service | checkout | checkout 0.75, frontend 0.20, frontend-proxy 0.05 | PASS |
| Diagnosis | application_failure | application_failure 0.92, dependency_unreachable 0.06 | FAIL |
| Action | escalate | escalate 0.35, investigate 0.29, restart_service 0.24 | PASS |

The result file also has the full request, the raw reply, token counts and a timing block. Timing values that the harness cannot measure are `null`.

## Known limits

The OpenTelemetry Collector of the demo dropped data in every run. Each of the 24 observations contains 59 to 132 error log lines from `otelcol-contrib`, for example "Exporting failed. Dropping data." These lines were also in all four healthy runs. Jev sees them as errors, and they are a likely cause of the false positives. The telemetry is also less complete than it looks.

The healthy demo is slow. In the healthy runs, checkout had a p95 latency of 9.6 to 14.6 seconds in the baseline. A healthy run therefore also shows large latency values and some noise between windows.

The timing fields do not measure the speed of Jev. Jev is asked once, at a fixed 240 seconds after injection. `diagnosis_seconds` is the time from injection to Jev's reply, so it is about 241 seconds in every run. `detection_seconds` is the time from when the harness saw the fault to Jev's reply, so it measures how fast the fault appeared.

Some expected answers are a judgment call:

- In F004, an engineer can say that payment is the problem, because checkout cannot reach it. We chose checkout.
- In F006, Jev once named `postgresql` with the right diagnosis. We chose product-catalog because the lock affects its queries. That answer counted as a localization FAIL.
- F001 fails for one product only, so its signal is small.

Other limits:

- Only Jev ran on Tier 2. No baseline engine exists for Tier 2 yet, so there is nothing to compare against.
- All runs used one laptop, one demo version and one load level.
- The run files are not in the repository, because `results/` is git-ignored.

## Recommended next experiment

First, remove the collector noise and measure the false-positive rate again. Two methods are possible. You can stop the collector from dropping data, or you can leave `otelcol-contrib` logs out of the observation. Leaving data out is a change to the experiment, so it needs a decision and a new leak test first. Then run F000 at least 10 times. If Jev still reports an incident on a healthy system, the false positives come from Jev and not from the testbed.

Second, add a baseline engine to Tier 2, for example Claude through `tier2/engine.py`. Then run it on the same scenarios. Without a second engine, we cannot tell whether F004 and F006 are hard for Jev or hard for any engine.

## How to repeat

Start the demo with `./setup.sh --start`, wait about 3 minutes, and run `./health-check.sh`. Then run `./run-all-experiments --repeats 3`. Jev's answers can change between runs, and the live demo is never the same twice. The figures on this page come from the `evaluation`, `jev`, `request` and `timing` fields of the 24 result files listed above.
