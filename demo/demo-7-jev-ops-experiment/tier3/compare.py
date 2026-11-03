"""Tier 3/4 prompt-variant comparison (`python -m tier3 compare`). Rows in the LF Edge
group's terms plus run-level detection precision, recall and F1 and the censored count,
computed per column with tier3/report.py's own counting. Every figure is computed from
the records; the layout, grouping and checks are in prompts/compare.py."""
from __future__ import annotations

import json
from pathlib import Path

from prompts.compare import cell, f1_cell, render
from prompts.loader import VariantError
from tier3.report import _block, engine_name


def load_items(paths: list[str | Path]) -> list[tuple[Path, dict]]:
    """Every Tier 3/4 record, live or replay, under the given paths. A file reached
    twice (through a symlinked `*-combined` directory and its original) counts once."""
    seen, out = set(), []
    for root in paths:
        root = Path(root)
        files = [root] if root.is_file() else sorted(root.rglob("*.json"))
        for f in files:
            if f.name == "summary.json" or f.name.startswith("."):
                continue
            real = f.resolve()
            if real in seen:
                continue
            rec = json.loads(f.read_text())
            if isinstance(rec, dict) and "experiment_id" in rec and "polls" in rec:
                seen.add(real)
                out.append((f, rec))
    return out


def _b(recs):
    return _block(recs) if recs else None


def _row(fn):
    def go(recs):
        b = _b(recs)
        return "n/a" if b is None else fn(b, recs)
    return go


ROWS = [
    ("runs ok", _row(lambda b, r: cell((b["runs_ok"], b["runs"])))),
    ("detect (fault runs)", _row(lambda b, r: cell(b["detect"]))),
    ("RCA (localization)", _row(lambda b, r: cell(b["rca"]))),
    ("diagnosis", _row(lambda b, r: cell(b["diagnosis"]))),
    ("remediation correct (action)", _row(lambda b, r: cell(b["remediation_correct"]))),
    ("FP runs (no-incident runs)", _row(lambda b, r: cell(b["false_positive_runs"]))),
    ("FP polls", _row(lambda b, r: cell(b["false_positive_polls"]))),
    ("detection precision", _row(lambda b, r: cell(b["detection_precision"]))),
    ("detection recall", _row(lambda b, r: cell(b["detection_recall"]))),
    ("detection F1", _row(lambda b, r: f1_cell(b["detection_f1"]))),
    ("censored runs (replay)", _row(lambda b, r: cell((sum(1 for x in r if x.get("censored")), len(r)))
                                    if any(x.get("mode") == "replay" for x in r) else "n/a")),
    ("invalid replies (polls)", _row(lambda b, r: cell(b["invalid_replies"]))),
    ("engine calls", _row(lambda b, r: str(b["engine_calls"]))),
]

NOTES = [
    "Each cell is a count over its denominator, with the percentage rounded. detect, RCA, "
    "diagnosis and remediation correct count fault runs; FP counts no-incident runs and the "
    "control. Detection precision, recall and F1 are per run (F1 = 2TP / (2TP + FP + FN)).",
    "Censored: a replayed run whose live polling stopped at a detection, where the variant "
    "reported no incident on every stored poll. It counts as not detected (fault runs) or as "
    "no false positive (no-incident runs) on the stored polls; it is neither a confirmed miss "
    "nor a confirmed true negative.",
    "Replay has no wall-clock MTTD and no remediation; compare those only between live runs.",
]


def dev_records_of(name: str, version: int | None):
    from tier3.questions import load_prompt
    try:
        v = load_prompt(name)
    except VariantError:
        return None
    return v.dev_records if version in (None, v.version) else None


def format_compare(paths: list[str], dev: list[str] | None = None,
                   confirm: list[str] | None = None, unpaired: bool = False) -> str:
    from tier3.layouts import record_layout
    items = load_items([*paths, *(dev or []), *(confirm or [])])
    return render("Tier 3/4 prompt variant and observation layout comparison", items, ROWS,
                  engine_name=engine_name, dev=dev, confirm=confirm, dev_records_of=dev_records_of,
                  notes=NOTES, unpaired=unpaired, layout_of=lambda r: record_layout(r)[0])
