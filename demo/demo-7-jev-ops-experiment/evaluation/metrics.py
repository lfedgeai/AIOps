"""Standard classification, detection, latency, reliability and cost metrics.

Pure functions over run records, so every figure is computed from the records.
Errored runs (no answer) are excluded from quality metrics, as in aggregate().
An invalid answer is scored as its own predicted label "(invalid)": it is never
a correct answer and is never coerced into a real one.
"""
from __future__ import annotations

from math import sqrt

from evaluation.evaluator import _mean, _pct
from scenarios.definitions import ACTIONS, CAUSES, HUMAN_REVIEW, SEVERITIES

LABELS = {"severity": SEVERITIES, "action": ACTIONS, "human_review": HUMAN_REVIEW,
          "probable_cause": CAUSES}
INVALID = "(invalid)"
DISRUPTIVE = ("restart", "escalate")  # actions that change the system or page a human


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion k/n. Sound for small n."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _ratio(a: int, b: int) -> float | None:
    return a / b if b else None


def _scored(records):
    return [r for r in records if r["status"] != "error"]


def _pred(r, field):
    return r["decision"].get(field) if r["status"] == "ok" else INVALID


def confusion(records: list[dict], field: str) -> dict[str, dict[str, int]]:
    """Expected label -> predicted label -> count, over scored runs."""
    m: dict[str, dict[str, int]] = {}
    for r in _scored(records):
        exp = r["expected"].get(field)
        if exp is None:
            continue
        row = m.setdefault(exp, {})
        p = _pred(r, field)
        row[p] = row.get(p, 0) + 1
    return m


def per_class(conf: dict[str, dict[str, int]], labels) -> dict[str, dict]:
    """Precision, recall, F1 and support for every label that is expected or predicted."""
    present = [l for l in labels if l in conf or any(l in row for row in conf.values())]
    out = {}
    for l in present:
        tp = conf.get(l, {}).get(l, 0)
        support = sum(conf.get(l, {}).values())
        predicted = sum(row.get(l, 0) for row in conf.values())
        prec, rec = _ratio(tp, predicted), _ratio(tp, support)
        f1 = (2 * prec * rec / (prec + rec) if prec is not None and rec is not None and prec + rec
              else (0.0 if predicted or support else None))
        out[l] = {"precision": prec, "recall": rec, "f1": f1, "support": support,
                  "predicted": predicted}
    return out


def f1_summary(classes: dict[str, dict]) -> dict:
    f1s = [c["f1"] for c in classes.values() if c["f1"] is not None]
    total = sum(c["support"] for c in classes.values())
    weighted = (sum(c["f1"] * c["support"] for c in classes.values() if c["f1"] is not None) / total
                if total else None)
    return {"macro_f1": _mean(f1s), "weighted_f1": weighted}


def binary(records: list[dict], is_pos_expected, is_pos_predicted) -> dict:
    """Counts and rates for a yes/no question. Invalid answers count as 'not positive'."""
    tp = fp = fn = tn = 0
    for r in _scored(records):
        e, p = is_pos_expected(r), r["status"] == "ok" and is_pos_predicted(r)
        tp += e and p
        fp += (not e) and p
        fn += e and not p
        tn += (not e) and not p
    prec, rec = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if prec is not None and rec is not None and prec + rec else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1,
            "false_positive_rate": _ratio(fp, fp + tn), "miss_rate": _ratio(fn, fn + tp),
            "specificity": _ratio(tn, fp + tn), "recall_ci": wilson(tp, tp + fn),
            "fpr_ci": wilson(fp, fp + tn)}


def severity_bias(records: list[dict]) -> dict:
    rank = {s: i for i, s in enumerate(SEVERITIES)}
    ok = [r for r in records if r["status"] == "ok"]
    over = sum(1 for r in ok if rank[r["decision"]["severity"]] > rank[r["expected"]["severity"]])
    under = sum(1 for r in ok if rank[r["decision"]["severity"]] < rank[r["expected"]["severity"]])
    return {"n": len(ok), "over": over, "under": under,
            "over_rate": _ratio(over, len(ok)), "under_rate": _ratio(under, len(ok))}


def action_risk(records: list[dict]) -> dict:
    """Disruptive actions nobody asked for, and escalations that were missed."""
    s = _scored(records)
    calm = [r for r in s if r["expected"]["action"] not in DISRUPTIVE]
    unwarranted = sum(1 for r in calm if _pred(r, "action") in DISRUPTIVE)
    need_esc = [r for r in s if r["expected"]["action"] == "escalate"]
    missed = sum(1 for r in need_esc if _pred(r, "action") != "escalate")
    return {"unwarranted_disruptive": unwarranted, "calm_runs": len(calm),
            "unwarranted_rate": _ratio(unwarranted, len(calm)),
            "missed_escalation": missed, "escalation_runs": len(need_esc),
            "missed_escalation_rate": _ratio(missed, len(need_esc))}


def brier_human_review(records: list[dict]) -> dict:
    """Mean squared error of P(human review) against the expected yes/no. Needs probabilities."""
    pairs = [(r["decision"]["human_review_probability"], 1.0 if r["expected"]["human_review"] == "yes" else 0.0)
             for r in records if r["status"] == "ok"
             and r["decision"].get("human_review_probability") is not None]
    return {"n": len(pairs), "brier": _mean((p - y) ** 2 for p, y in pairs)}


def latency(records: list[dict]) -> dict:
    xs = sorted(r["decision"]["latency_ms"] for r in records
                if r["decision"].get("latency_ms") is not None)
    return {"n": len(xs), "mean": _mean(xs), "p50": _pct(xs, 0.5), "p90": _pct(xs, 0.9),
            "p95": _pct(xs, 0.95), "p99": _pct(xs, 0.99), "max": xs[-1] if xs else None}


def reliability(records: list[dict]) -> dict:
    n = len(records)
    errored = sum(1 for r in records if r["status"] == "error")
    invalid = sum(1 for r in records if r["status"] == "invalid")
    retried = sum(1 for r in records if (r["decision"].get("attempts") or 0) > 1)
    return {"runs": n, "errored": errored, "invalid": invalid, "retried": retried,
            "availability": _ratio(n - errored, n), "error_rate": _ratio(errored, n),
            "invalid_rate": _ratio(invalid, n - errored), "retry_rate": _ratio(retried, n)}


def cost(records: list[dict]) -> dict:
    costs = [r["decision"]["cost_usd"] for r in records if r["decision"].get("cost_usd") is not None]
    correct = sum(1 for r in records if r["status"] != "error" and r["match"]["overall"])
    total = sum(costs) if costs else None
    complete = len(costs) == len(records)  # per-correct only when every run has a cost
    return {"total": total, "per_run": _mean(costs), "runs_with_cost": len(costs),
            "per_correct": total / correct if total is not None and correct and complete else None,
            "tokens_in": _mean(r["decision"]["input_tokens"] for r in records
                               if r["decision"].get("input_tokens") is not None),
            "tokens_out": _mean(r["decision"]["output_tokens"] for r in records
                                if r["decision"].get("output_tokens") is not None),
            "sources": sorted({r["decision"]["cost_source"] for r in records
                               if r["decision"].get("cost_source")})}


def compute(records: list[dict]) -> dict:
    s = _scored(records)
    correct = sum(1 for r in s if r["match"]["overall"])
    fields = {}
    for f, labels in LABELS.items():
        conf = confusion(records, f)
        classes = per_class(conf, list(labels) + [INVALID])
        k = sum(1 for r in s if r["match"].get(f))
        fields[f] = {"accuracy": _ratio(k, len(s)), "accuracy_ci": wilson(k, len(s)), "correct": k,
                     "n": len(s), "confusion": conf, "classes": classes, **f1_summary(
                         {l: c for l, c in classes.items() if l != INVALID})}
    return {
        "runs": len(records), "scored": len(s), "correct": correct,
        "accuracy": _ratio(correct, len(s)), "accuracy_ci": wilson(correct, len(s)),
        "fields": fields,
        "detection": binary(records, lambda r: r["expected"]["severity"] != "normal",
                            lambda r: r["decision"]["severity"] != "normal"),
        "human_review": binary(records, lambda r: r["expected"]["human_review"] == "yes",
                               lambda r: r["decision"]["human_review"] == "yes"),
        "severity_bias": severity_bias(records),
        "action_risk": action_risk(records),
        "calibration": brier_human_review(records),
        "latency": latency(records),
        "reliability": reliability(records),
        "cost": cost(records),
    }
