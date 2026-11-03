"""Option-order probe analysis (docs/results-2026-11-02-option-order.md).

Pairs the same observation asked two ways and counts, per choice question, how often
the answer changed and in which direction. "toward earlier" means the new answer is an
option that appears earlier in the second run's option list than the first answer did.
Run from the repository root: PYTHONPATH=. python docs/analysis/option_order.py
"""
import glob
import json

import jev.client as t1
import tier2.jev_adapter as t2
import tier3.questions as t3
from prompts.loader import load_variant


def option_orders(tier, mod, name):
    qs = load_variant(tier, name, mod.QUESTION_SCHEMA).questions
    return {q: list(v["criteria"]) for q, v in qs.items() if v["type"] == "choice"}


def answers(decision):
    return ((decision or {}).get("raw_response") or {}).get("answers") or {}


def records(d):
    return [json.load(open(f)) for f in sorted(glob.glob(f"{d}/*.json")) if not f.endswith("summary.json")]


def key(r):
    return json.dumps(r["source"], sort_keys=True)


def tier3_pairs(a, b):
    A = {key(r): r for r in records(a)}
    return [(answers(pa["decision"]), answers(pb["decision"]))
            for r in records(b) for pa, pb in zip(A[key(r)]["polls"], r["polls"])]


def tier2_pairs(a, b):
    A = {key(r): r for r in records(a)}
    return [(answers(A[key(r)]["jev"]), answers(r["jev"])) for r in records(b)]


def tier1_pairs(a, b):
    A = [json.loads(line) for line in open(a)]
    B = [json.loads(line) for line in open(b)]
    assert all(x["seed"] == y["seed"] and x["scenario"] == y["scenario"] for x, y in zip(A, B))
    return [(answers(x["decision"]), answers(y["decision"])) for x, y in zip(A, B)]


def report(label, pairs, second_order):
    for q, opts in second_order.items():
        n = changed = earlier = 0
        tv = []
        for a, b in pairs:
            qa, qb = a.get(q), b.get(q)
            if not qa or not qb:
                continue
            n += 1
            if qa["choice"] != qb["choice"]:
                changed += 1
                earlier += opts.index(qb["choice"]) < opts.index(qa["choice"])
            pa, pb = qa.get("probabilities") or {}, qb.get("probabilities") or {}
            if pa and pb:
                tv.append(sum(abs(pa.get(k, 0) - pb.get(k, 0)) for k in set(pa) | set(pb)) / 2)
        mean_tv = sum(tv) / len(tv) if tv else float("nan")
        print(f"{label:24s} {q:20s} changed {changed:3d}/{n:<4d} toward earlier {earlier:3d}  mean TV {mean_tv:.3f}")


R = "results"
rev = {t: option_orders(t, m, "baseline-reversed") for t, m in (("tier1", t1), ("tier2", t2), ("tier3", t3))}
base = {t: option_orders(t, m, "baseline") for t, m in (("tier1", t1), ("tier2", t2), ("tier3", t3))}

print("== baseline (2026-11-02) vs baseline-reversed (2026-11-02); direction read in the reversed order")
report("T3/4 dev", tier3_pairs(f"{R}/tier3/replay/20261102-131652-587a", f"{R}/tier3/replay/20261102-132221-2217"), rev["tier3"])
report("T3/4 confirm", tier3_pairs(f"{R}/tier3/replay/20261102-131914-5400", f"{R}/tier3/replay/20261102-132454-145b"), rev["tier3"])
report("T2 dev", tier2_pairs(f"{R}/tier2/replay/20261102-131643-31c6", f"{R}/tier2/replay/20261102-132210-abb9"), rev["tier2"])
report("T2 confirm", tier2_pairs(f"{R}/tier2/replay/20261102-131645-49c3", f"{R}/tier2/replay/20261102-132213-dc85"), rev["tier2"])
report("T1 seed 1000", tier1_pairs(f"{R}/20261102-151614-70runs.jsonl", f"{R}/20261102-152141-70runs-prompt-baseline-reversed.jsonl"), rev["tier1"])
report("T1 seed 2000", tier1_pairs(f"{R}/20261102-151642-70runs.jsonl", f"{R}/20261102-152210-70runs-prompt-baseline-reversed.jsonl"), rev["tier1"])

print("== noise floor: baseline (2026-11-01) vs baseline (2026-11-02), same order")
report("T3/4 dev", tier3_pairs(f"{R}/tier3/replay/20261101-162733-cf70", f"{R}/tier3/replay/20261102-131652-587a"), base["tier3"])
report("T3/4 confirm", tier3_pairs(f"{R}/tier3/replay/20261101-163423-577f", f"{R}/tier3/replay/20261102-131914-5400"), base["tier3"])
report("T2 dev", tier2_pairs(f"{R}/tier2/replay/20261101-162634-0735", f"{R}/tier2/replay/20261102-131643-31c6"), base["tier2"])
report("T2 confirm", tier2_pairs(f"{R}/tier2/replay/20261101-162641-2d70", f"{R}/tier2/replay/20261102-131645-49c3"), base["tier2"])
report("T1 seed 1000", tier1_pairs(f"{R}/20261101-182620-70runs.jsonl", f"{R}/20261102-151614-70runs.jsonl"), base["tier1"])
report("T1 seed 2000", tier1_pairs(f"{R}/20261101-182121-70runs.jsonl", f"{R}/20261102-151642-70runs.jsonl"), base["tier1"])
