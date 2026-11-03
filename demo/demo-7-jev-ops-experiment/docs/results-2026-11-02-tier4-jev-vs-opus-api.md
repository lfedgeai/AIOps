# Results: Jev vs Claude Opus on Tier 4, live, with Opus through the Anthropic API (2026-11-02)

This page repeats the live Tier 4 comparison of [results-2026-10-19-tier4-jev-vs-opus.md](results-2026-10-19-tier4-jev-vs-opus.md), with two changes:

- Opus answered through the Anthropic API (`transport: anthropic-api`), not `claude -p`. We kept the CLI off `PATH` for the Opus batches, so a missing key would have stopped them instead of falling back.
- Jev is `jev-1.14.0`, not `jev-1.13.0`.

Everything else is unchanged: the same kind cluster, the same eleven scenarios (K001-K010 x3, K000 x10), the same observations, the same four questions (baseline prompt), one action per run from the allowlist, and a 360 s recovery watch. How a run works, and how it differs from the LF Edge harness, is on the October pages and is not repeated here.

**Every MTTR on this page is single-shot recommendation MTTR**: one answer, carried out once, with no feedback or retry. It is not operational MTTR.

## What ran

| | Jev | Claude Opus |
|---|---|---|
| Engine | `jev` (`jev-api`), `jev-1.14.0` on every poll | `claude-opus` (`anthropic-api`), `claude-opus-5-5`, effort `medium` |
| Fault runs | `results/tier4/20261101-201755-3d81/` (2026-11-01 22:17 to 2026-11-02 03:55) | `results/tier4/20261102-040013-7aef/` (06:00 to 11:31) |
| Control runs | `results/tier4/20261102-015505-b860/` (03:55 to 06:00) | `results/tier4/20261102-093118-cb18/` (11:31 to 13:36) |
| Combined | `results/tier4/11.1-jev-combined/`, scorecard `11.1-jev-combined-scorecard.md` | `results/tier4/11.1-opus-combined/`, scorecard `11.1-opus-combined-scorecard.md` |
| Runs | 40/40 `ok`, 0 contaminated | 40/40 `ok`, 0 contaminated |
| Engine calls | 295, none failed, none invalid | 250, none failed, none invalid |
| Tokens in / out | 1.27M / 114k | 1.47M / 17k |
| Answer time per call, median / p95 (min-max) | 384 ms / 499 ms (312-599, 295 calls) | 2,157 ms / 4,999 ms (1,423-7,699, 250 calls) |
| Cost | $0.053 (estimated from `JEV_PRICE_*`) | `n/a` in the records; about $6.21 estimated afterwards from the recorded tokens at $4 / $20 per million (`CLAUDE_OPUS_PRICE_*`, set 2026-11-02) |

Chain script: `results/tier4/chain-20261101.sh`, log `chain-20261101.log`. The fault ledger was clear after every run. Commands: `python -m tier4 scorecard results/tier4/<name>-combined`.

## The four verdicts

| | Jev | Opus |
|---|---|---|
| Detect (fault runs) | 18/18 | 18/18 |
| RCA (localization) | 9/18 | **18/18** |
| Diagnosis | 9/18 | **18/18** |
| Remediation correct | 18/18 | 18/18 |
| MTTD median (min-max) | 16.6 s (15.9-30.6) | 18.7 s (17.7-21.1) |
| False-positive runs (no-incident scenarios and control) | **4/22** | 7/22 |
| Detection precision / F1 | 18/22, 0.90 | 18/25, 0.84 |

"Remediation correct" scores the recommended action against the scenario's acceptable actions, which include `investigate`. It does not say whether the action fixed anything. The next table shows that.

False positives: both engines call K002 (cart pods deleted once, recovered) an incident 3/3 and K010 1/3. Opus also calls K006 (a config variable cart does not read) an incident 3/3. Jev never does. The harness carried out no false-alarm action for either engine (`investigate` does nothing), and no healthy run was disturbed.

## Remediation and MTTR

| 18 fault runs | Jev | Opus |
|---|---|---|
| Action executed | 6/18 | 11/18 |
| Recovered by action | 3/18 | **11/18** |
| Recovered without action | 3/18 | 3/18 |
| Not recovered | 12/18 | 4/18 |
| Relapsed | 0/18 | 0/18 |
| MTTR median (min-max), fault to action written | 16.3 s (15.9-17.6, n=6) | 18.7 s (17.7-23.2, n=11) |
| Verified MTTR median (min-max), fault to recovery verified | 106.2 s (105.9-151.9, n=3) | 122.7 s (20.0-139.2, n=11) |

What each engine did, per scenario (component / action, then outcome):

| Scenario | Jev | Opus |
|---|---|---|
| K001 cart to zero | cart `scale_up` x2: recovered by action; frontend `scale_up` x1: not recovered | cart `scale_up` x3: recovered by action |
| K003 memory limit | cart `investigate` x3: not recovered | cart `rollback_change` x1: recovered by action; cart `investigate` x2: not recovered |
| K005 readiness probe | cart `investigate` x3: not recovered | cart `rollback_change` x1: recovered by action; cart `investigate` x2: not recovered |
| K007 valkey-cart to zero | frontend-proxy x2, frontend x1, `investigate`: not recovered | valkey-cart `scale_up` x3: recovered by action |
| K008 cart to zero under load | cart `scale_up` x1: recovered by action; frontend `scale_up` x2: not recovered | cart `scale_up` x3: recovered by action |
| K009 node taint | product-catalog, frontend-proxy, frontend, `investigate`: recovered without action | the tainted worker, `investigate` x3: recovered without action |

Verified MTTR by scenario, Opus: K001 123.5 s (108.7-123.6), K003 20.0 s (n=1), K005 23.2 s (n=1), K007 123.9 s (109.6-139.2), K008 122.7 s (108.2-123.4). Jev: K001 129.1 s (106.2-151.9, n=2), K008 105.9 s (n=1). The rollbacks recover almost at once, because `rollout undo` brings back the old template. Scaling from zero waits for a new pod to start and pass its checks, so the two engines' times on the same fix are close.

## What changed since October

| | Jev Oct (`1.13.0`) | Jev Nov (`1.14.0`) | Opus Oct (`claude -p`) | Opus Nov (API) |
|---|---|---|---|---|
| RCA | 10/18 | 9/18 | 18/18 | 18/18 |
| Diagnosis | 10/18 | 9/18 | 18/18 | 18/18 |
| Recovered by action | 4/18 | 3/18 | 13/18 | 11/18 |
| Not recovered | 12/18 | 12/18 | 2/18 | 4/18 |
| False-positive runs | 5/22 | 4/22 | 7/22 | 7/22 |
| Verified MTTR median | 106.2 s (n=4) | 106.2 s (n=3) | 110.7 s (n=13) | 122.7 s (n=11) |

The picture is the same as in October. Through the API, Opus again names the failing component and the failure mode on every fault run, and its action fixes 11 of 18 against Jev's 3. The gap is not detection: both detect every fault, and Jev raises fewer false alarms. It is in the component and the action:

- On the three scale-from-zero faults (K001, K007, K008), Opus recovers 9/9 and Jev 3/9, because Jev often names the caller (frontend, frontend-proxy) and scales that instead.
- On the two bad-template faults (K003, K005), Jev recommends `investigate` every time, which the harness does not act on. Opus recommends `rollback_change` in 2 of 6 runs (4 of 6 in October) and those recover within about 20 s.

Run-to-run differences of one or two runs between October and November are within noise (3 runs per scenario).

## Limits

- 3 runs per fault scenario and 10 control runs per engine. The MTTR medians rest on 3 (Jev) and 11 (Opus) runs.
- One prompt (baseline), one Opus effort level (`medium`).
- The two engines ran one after the other on the same cluster, not interleaved, so cluster drift over the day is not controlled.
- The recovery check is service-level, as in October. The October page explains how it differs from the LF Edge harness's checks.
