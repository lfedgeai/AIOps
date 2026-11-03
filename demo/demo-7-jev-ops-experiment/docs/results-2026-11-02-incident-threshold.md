# Results: can the incident threshold remove Jev's false alarms? Tiers 3/4 (2026-11-02)

The prompt variant results ([docs/results-2026-11-01-prompt-variants.md](results-2026-11-01-prompt-variants.md)) found that `strict-incident` lowered Jev's probability of "incident" on recovered failures but never below 0.5, the cutoff the harness uses (`INCIDENT_THRESHOLD`, `docs/jev-api-notes.md`). This page asks the next question: is there a better cutoff? It is an offline analysis of stored polls. It makes no Jev calls, changes no code, and does not change the scoring. Every figure comes from the records named below, through the script at the end.

## Short answer

No. Within every batch, the false alarms (p 0.75 to 0.94) fall inside the range of the true detections (p 0.68 to 0.95). Any cutoff from about 0.5 to 0.68 gives the same results as 0.5. A cutoff high enough to remove false alarms loses true detections first. The cutoff is not where the false alarms can be fixed.

## Sources

| Set | Records | Model | Polls |
|---|---|---|---|
| Tier 3 dev | `results/tier3/10.3-combined` (40 runs) | `jev-1.13.0` | 351 |
| Tier 4 confirm | `results/tier4/6.2-combined` (40 runs) | `jev-1.13.0` | 280 |
| Tier 4 live, 2026-11-02 | `results/tier4/11.1-jev-combined` (40 runs) | `jev-1.14.0` | 295 |

Each set has 18 incident runs (K001, K003, K005, K007, K008, K009) and 22 no-incident runs (10 × K000, plus K002, K004, K006, K010). This page does not use the Opus records, because Claude returns no probabilities.

## What the probabilities look like

| | Tier 3 dev | Tier 4 confirm | Tier 4 11.1 |
|---|---|---|---|
| Lowest p on the detecting poll, incident runs | 0.70 | 0.76 | 0.68 |
| Highest p on any poll of a clean no-incident run | 0.42 | 0.31 | 0.27 |
| False-alarm runs at 0.5, with their highest p | 4: 0.89, 0.91, 0.91, 0.94 | 5: 0.80, 0.80, 0.88, 0.89, 0.90 | 4: 0.75, 0.79, 0.82, 0.89 |
| Which scenarios | K002 ×3, K010 ×1 | K002 ×3, K010 ×2 | K002 ×3, K010 ×1 |

Every false alarm in all three sets is K002 (cart pods deleted once, recovered) or K010 (a cart pod OOMKilled, recovered). On those runs Jev is about as sure of an incident as it is on real, ongoing faults. K002 runs polled 15 times stay at 0.8 to 0.9 on every poll, so the high probability lasts the whole window. It is not a single bad poll.

## Threshold sweep

We score detection as in Tier 3: the first poll with p ≥ t. Live polling stopped at the first poll with p ≥ 0.5 (Tier 4 acts on it), so above that poll's p a run has no later polls to look at. We count it as `censored`, not as missed or detected.

| t | Tier 3 dev: detected (censored) / FP runs | Tier 4 confirm | Tier 4 11.1 |
|---|---|---|---|
| 0.35 | 18/18 (0) / 5/22 | 18/18 (0) / 5/22 | 18/18 (0) / 4/22 |
| **0.50** (used) | 18/18 (0) / 4/22 | 18/18 (0) / 5/22 | 18/18 (0) / 4/22 |
| 0.60 | 18/18 (0) / 4/22 | 18/18 (0) / 5/22 | 18/18 (0) / 4/22 |
| 0.70 | 18/18 (0) / 4/22 | 18/18 (0) / 5/22 | 15/18 (3) / 4/22 |
| 0.75 | 15/18 (3) / 4/22 | 18/18 (0) / 5/22 | 14/18 (4) / 4/22 |
| 0.80 | 14/18 (4) / 4/22 | 14/18 (4) / 5/22 | 11/18 (7) / 2/22 |
| 0.85 | 11/18 (7) / 4/22 | 10/18 (8) / 3/22 | 5/18 (13) / 1/22 |
| 0.90 | 6/18 (12) / 3/22 | 7/18 (11) / 1/22 | 0/18 (18) / 0/22 |

Lowering the cutoff to 0.35 adds a false alarm on the dev set. Raising it to 0.80 or more removes some false alarms only after detections start to drop out. A censored run might still have been detected on a later poll, but polls after a detection were never taken, so we cannot check this from these records.

## What this means

- Keep 0.5. It is in the flat region on every set, so the published figures do not depend on the exact choice.
- The false alarms are about the observation, not the cutoff or the wording. On K002 and K010 the failure has already recovered, but it is probably still visible inside the observation window, and Jev scores it as an ongoing incident. (This is an inference. We did not check here which fields drive it.) A fix has to change what Jev sees. One candidate is the observation's time layout: show clearly what is true now and what happened earlier. That would be a change to the observation builder, with the same leak rules, and has to be tested like any other change.
- The ranking of runs is good. Clean healthy runs never go above 0.42, and every real fault scores at least 0.68. Jev separates "something happened" from "nothing happened" well. It does not separate "is happening" from "has happened".

## Limits

- 18 incident runs and 22 no-incident runs per set. False alarms come from 2 scenarios only.
- Censoring: polling stopped at the first detection, so cutoffs above the detecting poll's p cannot be judged for that run.
- Choosing a cutoff from these sets and reporting it on the same sets would be fitting to the test set. This page proposes no cutoff, so this does not arise.

## Script

Run from the repository root, over the record directories above:

```python
import glob, json, sys

def load(d):
    out = []
    for f in sorted(glob.glob(f"{d}/K*.json")):
        r = json.load(open(f))
        if r.get("status") != "ok":
            continue
        ps = [p["decision"]["raw_response"]["answers"]["incident_detected"]["noul"]
              for p in r["polls"] if p["decision"].get("status") == "ok"]
        out.append((r["experiment_id"], r["ground_truth"]["incident"], ps))
    return out

for d in sys.argv[1:]:
    runs = load(d)
    pos = [ps for _, inc, ps in runs if inc]
    neg = [ps for _, inc, ps in runs if not inc]
    for t in (0.35, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
        det = sum(any(p >= t for p in ps) for ps in pos)
        cens = sum(not any(p >= t for p in ps) and len(ps) < 15 for ps in pos)
        fp = sum(any(p >= t for p in ps) for ps in neg)
        print(f"{d} t={t:.2f} detected {det}/{len(pos)} (censored {cens}) FP runs {fp}/{len(neg)}")
```
