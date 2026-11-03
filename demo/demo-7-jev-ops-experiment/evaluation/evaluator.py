"""Compare decisions to expected outcomes and aggregate results.

Pure functions over plain data. Every figure is computed from the records.
"""
from __future__ import annotations

from collections import Counter
from statistics import mean

FIELDS = ("severity", "action", "human_review")  # the operational decision: PASS needs all three
DIAGNOSIS_FIELDS = ("probable_cause",)  # scored, but not part of PASS/FAIL
ALL_FIELDS = FIELDS + DIAGNOSIS_FIELDS


def compare(expected, decision) -> dict:
    """Per-field correctness plus overall. Invalid and error decisions are all wrong."""
    exp = expected.to_dict() if hasattr(expected, "to_dict") else dict(expected)
    dec = decision.to_dict() if hasattr(decision, "to_dict") else dict(decision)
    ok = dec.get("status") == "ok"
    match = {f: bool(ok and exp.get(f) is not None and dec.get(f) == exp[f]) for f in ALL_FIELDS}
    match["overall"] = all(match[f] for f in FIELDS)
    return match


def make_record(run: int, scenario: str, seed: int, observations: dict, expected, decision) -> dict:
    dec = decision.to_dict() if hasattr(decision, "to_dict") else dict(decision)
    exp = expected.to_dict() if hasattr(expected, "to_dict") else dict(expected)
    return {
        "run": run,
        "scenario": scenario,
        "seed": seed,
        "observations": observations,
        "expected": exp,
        "decision": dec,
        "status": dec.get("status"),
        "match": compare(exp, dec),
    }


def _mean(xs):
    xs = list(xs)
    return mean(xs) if xs else None


def _pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _stats(xs) -> dict:
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "total": sum(xs) if xs else None, "mean": _mean(xs),
            "p50": _pct(xs, 0.5), "p95": _pct(xs, 0.95), "max": max(xs) if xs else None}


def telemetry(records: list[dict]) -> dict:
    """Tokens, cost, timing and transport figures. Covers every record, errors included."""
    ds = [r["decision"] for r in records]
    costs = [d.get("cost_usd") for d in ds if d.get("cost_usd") is not None]
    rt = [d.get("latency_ms") for d in ds]
    server = [d.get("server_elapsed_ms") for d in ds]
    overhead = [a - b for a, b in zip(rt, server) if a is not None and b is not None]

    by_scenario: dict[str, dict] = {}
    for name in dict.fromkeys(r["scenario"] for r in records):
        sd = [r["decision"] for r in records if r["scenario"] == name]
        by_scenario[name] = {
            "mean_input_tokens": _mean(d["input_tokens"] for d in sd if d.get("input_tokens") is not None),
            "mean_output_tokens": _mean(d["output_tokens"] for d in sd if d.get("output_tokens") is not None),
            "mean_latency_ms": _mean(d["latency_ms"] for d in sd if d.get("latency_ms") is not None),
            "mean_cost_usd": _mean(d["cost_usd"] for d in sd if d.get("cost_usd") is not None),
            "mean_log_lines": _mean(r.get("log_lines_sent", 0) for r in records if r["scenario"] == name),
        }

    return {
        "input_tokens": _stats(d.get("input_tokens") for d in ds),
        "output_tokens": _stats(d.get("output_tokens") for d in ds),
        "cost_usd": {
            "total": sum(costs) if costs else None,
            "mean_per_run": _mean(costs),
            "runs_with_cost": len(costs),
            "sources": dict(Counter(d["cost_source"] for d in ds if d.get("cost_source"))),
        },
        "round_trip_ms": _stats(rt),
        "server_elapsed_ms": _stats(server),
        "overhead_ms": _stats(overhead),  # round trip minus the engine's own reported time
        "total_ms": _stats(d.get("total_ms") for d in ds),
        "attempts": sum(d.get("attempts") or 0 for d in ds),
        "runs_retried": sum(1 for d in ds if (d.get("attempts") or 0) > 1),
        "http_status": dict(Counter(str(d.get("http_status")) for d in ds)),
        "request_bytes": _stats(d.get("request_bytes") for d in ds),
        "by_scenario": by_scenario,
    }


def aggregate(records: list[dict]) -> dict:
    """Errored runs are excluded from accuracy; invalid runs count as incorrect."""
    errored = [r for r in records if r["status"] == "error"]
    scored = [r for r in records if r["status"] != "error"]
    invalid = [r for r in scored if r["status"] == "invalid"]
    correct = [r for r in scored if r["match"]["overall"]]

    by_field = {}
    for f in ALL_FIELDS:
        c = sum(1 for r in scored if r["match"].get(f))
        by_field[f] = {"correct": c, "total": len(scored)}

    by_scenario: dict[str, dict] = {}
    for r in records:
        s = by_scenario.setdefault(r["scenario"], {"correct": 0, "total": 0, "errored": 0})
        if r["status"] == "error":
            s["errored"] += 1
        else:
            s["total"] += 1
            s["correct"] += int(r["match"]["overall"])

    def conf(rs):
        return [r["decision"]["confidence"] for r in rs
                if r["status"] == "ok" and r["decision"].get("confidence") is not None]

    latencies = [r["decision"]["latency_ms"] for r in scored
                 if r["decision"].get("latency_ms") is not None]

    return {
        "runs": len(records),
        "scored": len(scored),
        "correct": len(correct),
        "incorrect": len(scored) - len(correct),
        "invalid": len(invalid),
        "errored": len(errored),
        "accuracy": len(correct) / len(scored) if scored else None,
        "by_field": by_field,
        "by_scenario": by_scenario,
        "confidence": {
            "mean": _mean(conf(scored)),
            "mean_correct": _mean(conf(correct)),
            "mean_incorrect": _mean(conf([r for r in scored if not r["match"]["overall"]])),
        },
        "latency_ms": {
            "mean": _mean(latencies),
            "p50": _pct(latencies, 0.5),
            "p95": _pct(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "telemetry": telemetry(records),
    }


def _fmt(x, spec=".2f", suffix=""):
    return "n/a" if x is None else f"{x:{spec}}{suffix}"


def format_summary(s: dict) -> str:
    lines = ["Overall", "-------",
             f"Runs:          {s['runs']}",
             f"Correct:       {s['correct']}",
             f"Incorrect:     {s['incorrect']}"]
    if s["invalid"]:
        lines.append(f"  of which invalid replies: {s['invalid']}")
    if s["errored"]:
        lines.append(f"Errored:       {s['errored']} (excluded from accuracy)")
    lines.append(f"Accuracy:      {_fmt(s['accuracy'] * 100 if s['accuracy'] is not None else None, '.1f', '%')}")

    lines += ["", "By decision type", "----------------"]
    for f, v in s["by_field"].items():
        note = "  (diagnosis, not in PASS/FAIL)" if f in DIAGNOSIS_FIELDS else ""
        lines.append(f"{f:<15} {v['correct']}/{v['total']}{note}")

    lines += ["", "By scenario", "-----------"]
    for name, v in s["by_scenario"].items():
        extra = f"  ({v['errored']} errored)" if v["errored"] else ""
        lines.append(f"{name:<16} {v['correct']}/{v['total']}{extra}")

    c, lat = s["confidence"], s["latency_ms"]
    lines += ["", "Confidence (mean of severity and action confidence)", "----------",
              f"All:           {_fmt(c['mean'])}",
              f"Correct:       {_fmt(c['mean_correct'])}",
              f"Incorrect:     {_fmt(c['mean_incorrect'])}",
              "", *format_telemetry(s["telemetry"], s["runs"])]
    return "\n".join(lines)


def _int_fmt(x) -> str:
    return "n/a" if x is None else f"{x:,.0f}"


def format_cost(c: dict) -> str:
    if c["total"] is None:
        return "n/a (no cost reported; set JEV_PRICE_INPUT_PER_MTOK and JEV_PRICE_OUTPUT_PER_MTOK to estimate)"
    src = ", ".join(f"{k} x{v}" for k, v in c["sources"].items())
    return f"${c['total']:.6f} total, ${c['mean_per_run']:.6f}/run ({src})"


def format_telemetry(t: dict, runs: int) -> list[str]:
    tin, tout, rt, sv, ov = (t["input_tokens"], t["output_tokens"], t["round_trip_ms"],
                             t["server_elapsed_ms"], t["overhead_ms"])
    status = ", ".join(f"{k} x{v}" for k, v in sorted(t["http_status"].items()))
    return [
        "Telemetry", "---------",
        f"Tokens in:     {_int_fmt(tin['total'])} total, {_int_fmt(tin['mean'])}/run",
        f"Tokens out:    {_int_fmt(tout['total'])} total, {_int_fmt(tout['mean'])}/run",
        f"Cost:          {format_cost(t['cost_usd'])}",
        f"Round trip:    mean {_fmt(rt['mean'], '.0f')} ms, p50 {_fmt(rt['p50'], '.0f')}, "
        f"p95 {_fmt(rt['p95'], '.0f')}, max {_fmt(rt['max'], '.0f')} ms",
        f"Engine time:   mean {_fmt(sv['mean'], '.0f')} ms, p95 {_fmt(sv['p95'], '.0f')} ms (as the engine reports it)",
        f"Overhead:      mean {_fmt(ov['mean'], '.0f')} ms (round trip minus engine time: network, startup, queueing)",
        f"Attempts:      {t['attempts']} for {runs} runs ({t['runs_retried']} retried)",
        f"HTTP status:   {status or 'n/a'}",
        f"Request size:  mean {_int_fmt(t['request_bytes']['mean'])} bytes",
    ]
