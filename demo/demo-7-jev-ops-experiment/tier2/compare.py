"""Tier 2 prompt-variant comparison (`python -m tier2 compare`). Per-dimension PASS
counts, false positives on the healthy control, and run-level detection precision,
recall and F1. Every figure is computed from the records; layout, grouping and checks
are in prompts/compare.py."""
from __future__ import annotations

import json
from pathlib import Path

from prompts.compare import cell, f1_cell, render
from prompts.loader import VariantError

DIMENSIONS = ("detection", "localization", "diagnosis", "action")


def engine_name(rec: dict) -> str:
    """Tier 2 live records predate the engine field; only Jev ran them."""
    return (rec.get("engine") or {}).get("name") or "jev"


def load_items(paths: list[str | Path]) -> list[tuple[Path, dict]]:
    seen, out = set(), []
    for root in paths:
        root = Path(root)
        files = [root] if root.is_file() else sorted(root.rglob("*.json"))
        for f in files:
            if f.name == "summary.json" or f.name.startswith(".") or f.resolve() in seen:
                continue
            rec = json.loads(f.read_text())
            if isinstance(rec, dict) and "experiment_id" in rec and "ground_truth" in rec and "polls" not in rec:
                seen.add(f.resolve())
                out.append((f, rec))
    return out


def _ok(recs):
    return [r for r in recs if r.get("status") == "ok"]


def _dim(dim):
    def go(recs):
        rows = [r for r in _ok(recs) if r["evaluation"][dim] != "UNKNOWN"]
        return cell((sum(1 for r in rows if r["evaluation"][dim] == "PASS"), len(rows)))
    return go


def _detected(r) -> bool:
    j = r.get("jev") or {}
    return j.get("status") == "ok" and bool(j.get("incident_detected"))


def confusion(recs) -> dict:
    ok = _ok(recs)
    pos = [r for r in ok if r["ground_truth"]["incident"]]
    neg = [r for r in ok if not r["ground_truth"]["incident"]]
    tp = sum(1 for r in pos if _detected(r))
    fp = sum(1 for r in neg if _detected(r))
    return {"tp": tp, "fn": len(pos) - tp, "fp": fp, "tn": len(neg) - fp, "neg": len(neg)}


def _c(fn):
    return lambda recs: fn(confusion(recs)) if recs else "n/a"


ROWS = [
    ("runs ok", lambda recs: cell((len(_ok(recs)), len(recs)))),
    *[(d, _dim(d)) for d in DIMENSIONS],
    ("FP runs (healthy control)", _c(lambda c: cell((c["fp"], c["neg"])))),
    ("detection precision", _c(lambda c: cell((c["tp"], c["tp"] + c["fp"])))),
    ("detection recall", _c(lambda c: cell((c["tp"], c["tp"] + c["fn"])))),
    ("detection F1", _c(lambda c: f1_cell((2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"])))),
    ("invalid replies", lambda recs: cell((sum(1 for r in recs if (r.get("jev") or {}).get("status") == "invalid"),
                                           len(recs)))),
    ("engine calls", lambda recs: str(sum(1 for r in recs if r.get("jev")))),
]

NOTES = [
    "Each cell is a count over its denominator, with the percentage rounded. The four "
    "dimensions count PASS among runs with a decided verdict (UNKNOWN left out). FP runs count "
    "healthy-control runs where an incident was reported. Detection precision, recall and F1 "
    "are per run (F1 = 2TP / (2TP + FP + FN)); F1 ignores true negatives, so read it with FP runs.",
    "Tier 2 asks once per run, so replay has no censoring. Replay has no timings.",
]


def dev_records_of(name: str, version: int | None):
    from tier2.jev_adapter import load_prompt
    try:
        v = load_prompt(name)
    except VariantError:
        return None
    return v.dev_records if version in (None, v.version) else None


def format_compare(paths: list[str], dev: list[str] | None = None,
                   confirm: list[str] | None = None, unpaired: bool = False) -> str:
    items = load_items([*paths, *(dev or []), *(confirm or [])])
    return render("Tier 2 prompt variant comparison", items, ROWS, engine_name=engine_name,
                  dev=dev, confirm=confirm, dev_records_of=dev_records_of, notes=NOTES,
                  unpaired=unpaired)
