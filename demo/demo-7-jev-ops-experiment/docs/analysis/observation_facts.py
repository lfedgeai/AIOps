"""observation-facts Phase A analysis (docs/results-2026-11-02-observation-facts.md).

Reads replay records only. Per scenario: localization and diagnosis PASS counts on fault
runs, false-positive runs on no-incident runs. Per pair of replays of the same source:
how many run-level verdicts flipped (a noise floor comes from the same prompt and layout
asked twice). Also request sizes, tokens and cost as the records report them.
Run from the repository root: python docs/analysis/observation_facts.py
"""
import glob
import json
from collections import defaultdict

R = "results/tier3/replay/"
SETS = {
    "confirm": {
        "jev raw (earlier, noise)": "20261102-131914-5400",
        "jev raw": "20261102-135216-1356",
        "jev facts": "20261102-135410-0041",
        "jev facts+strict": "20261102-135616-e810",
        "opus raw": "20261102-135824-276d",
        "opus facts": "20261102-141431-4f45",
    },
    "dev": {
        "jev raw (earlier, noise)": "20261102-131652-587a",
        "jev raw": "20261102-134431-028c",
        "jev facts": "20261102-134656-4668",
        "jev facts+strict": "20261102-134934-bffa",
    },
}
PAIRS = [
    ("jev raw (earlier, noise)", "jev raw"),
    ("jev raw", "jev facts"),
    ("jev raw", "jev facts+strict"),
    ("opus raw", "opus facts"),
]


def records(d):
    return [json.load(open(f)) for f in sorted(glob.glob(R + d + "/K*.json"))]


def key(r):
    return r["source"]["path"]


def per_run(r):
    e = r["evaluation"] or {}
    fp = r.get("false_positive")
    return {
        "scenario": r["experiment_id"],
        "fault": fp is None,
        "loc": e.get("localization") == "PASS",
        "diag": e.get("diagnosis") == "PASS",
        "fp": bool(fp and fp["false_positive_run"]),
        "censored": bool(r.get("censored")),
    }


for setname, dirs in SETS.items():
    print(f"\n## {setname}")
    runs = {n: {key(r): r for r in records(d)} for n, d in dirs.items()}
    names = list(dirs)
    table = defaultdict(lambda: {n: [0, 0, 0, 0] for n in names})  # loc, diag, fp, n
    for n in names:
        for r in runs[n].values():
            v = per_run(r)
            c = table[v["scenario"]][n]
            c[0] += v["loc"]; c[1] += v["diag"]; c[2] += v["fp"]; c[3] += 1
    print("scenario | " + " | ".join(names))
    for s in sorted(table):
        fault = any(per_run(r)["fault"] for r in runs[names[0]].values() if r["experiment_id"] == s)
        cells = [(f"loc {c[0]} diag {c[1]}" if fault else f"FP {c[2]}") + f" /{c[3]}"
                 for c in (table[s][n] for n in names)]
        print(f"{s} | " + " | ".join(cells))

    print("\nflips (fault runs: localization, diagnosis; no-incident runs: FP)")
    for a, b in PAIRS:
        if a not in runs or b not in runs:
            continue
        lf = df = ff = 0
        loc_gain = loc_loss = 0
        for k, ra in runs[a].items():
            va, vb = per_run(ra), per_run(runs[b][k])
            if va["fault"]:
                lf += va["loc"] != vb["loc"]; df += va["diag"] != vb["diag"]
                loc_gain += vb["loc"] and not va["loc"]; loc_loss += va["loc"] and not vb["loc"]
            else:
                ff += va["fp"] != vb["fp"]
        print(f"{a} -> {b}: localization {lf} (+{loc_gain}/-{loc_loss}), diagnosis {df}, FP {ff}")

    print("\nrequest bytes, tokens, cost, censored")
    for n in names:
        bs = [p["request_bytes"] for r in runs[n].values() for p in r["polls"]]
        ti = sum((r["usage"] or {}).get("input_tokens") or 0 for r in runs[n].values())
        to = sum((r["usage"] or {}).get("output_tokens") or 0 for r in runs[n].values())
        cost = [(r["usage"] or {}).get("cost_usd") for r in runs[n].values()]
        cs = "n/a" if any(c is None for c in cost) else f"{sum(cost):.3f}"
        cens = sum(per_run(r)["censored"] for r in runs[n].values())
        obs = {json.dumps(r.get("observation") or {"name": "raw", "version": 1}, sort_keys=True)
               for r in runs[n].values()}
        print(f"{n}: polls {len(bs)}, bytes mean {sum(bs) / len(bs):.0f} max {max(bs)}, "
              f"tokens in {ti} out {to}, cost USD {cs}, censored {cens}, observation {sorted(obs)}")
