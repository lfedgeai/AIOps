"""Replay stored Tier 3 and Tier 4 observations under a prompt variant (design D5).

Every stored poll's `request` (exactly what an engine received in the live run, after
scrubbing, with no ground truth) is sent again, in order, to an engine asking the chosen
variant's questions. No cluster, Docker or fault injection is involved. The answers are
scored with the live rules after every call has been made: the first detecting poll, else
the last; false-positive polls on no-incident runs.

The engine sees only the stored `request`. Ground truth is read from the source record in
a separate step, after the engine has answered every poll (`test_replay_leaks_nothing`).
Source records are never modified. Replay records go under `<results>/replay/<run-id>/`.

What replay cannot measure stays null: wall-clock MTTD and anything about remediation.
A run whose live polling stopped at its first detection stored only the polls up to it;
if the variant detects on none of them, the run is `censored` (we cannot know what it
would have said later), never a confirmed miss or a confirmed true negative.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from prompts.loader import engine_prompt, record_prompt
from tier2.runner import new_run_id
from tier3.engine import STATUS_ERROR, STATUS_OK, Decision
from tier3.evaluate import evaluate, false_positive_polls
from tier3.layouts import DEFAULT_LAYOUT, LayoutError, apply_layout, get_layout, record_layout
from tier3.runner import _decision_dict, _usage, engine_info
from tier3.scenarios import GroundTruth

REPLAY_DIR = "replay"
MODE = "replay"


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds")


def is_replay(rec: dict) -> bool:
    return rec.get("mode") == MODE


def load_sources(path: str | Path) -> list[tuple[Path, dict]]:
    """Live Tier 3/4 run records under `path`, sorted by path. Replay records,
    summaries and the fault ledger are left out."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.rglob("*.json"))
    out = []
    for f in files:
        if f.name == "summary.json" or f.name.startswith("."):
            continue
        rec = json.loads(f.read_text())
        if isinstance(rec, dict) and "experiment_id" in rec and "polls" in rec and not is_replay(rec):
            out.append((f, rec))
    return out


def _detects(poll: dict) -> bool:
    d = poll["decision"]
    return d["status"] == STATUS_OK and bool(d["incident_detected"])


def source_truncated(src: dict) -> bool:
    """True when live polling stopped at a detection, so later observations were never
    collected: the source's first detecting poll is its last stored poll. (Tier 3 fault
    runs and every Tier 4 run stop there; Tier 3 no-incident runs poll the whole window.)"""
    polls = src.get("polls") or []
    first = next((p for p in polls if _detects(p)), None)
    return first is not None and first is polls[-1]


def check_source_layout(src: dict, source_path=None) -> None:
    """A layout is applied to the stored request, so the source must hold a raw one.
    Laying out an already laid-out observation would record the wrong layout."""
    ident, _ = record_layout(src)
    if ident["name"] != DEFAULT_LAYOUT:
        raise LayoutError(f"{source_path or src.get('run_id')}: source observation layout is "
                          f"{ident['name']!r}; replay needs raw source records")


def replay_record(src: dict, engine, *, run_id: str, source_path: str | None = None,
                  clock: Callable[[], float] | None = None, layout: str = DEFAULT_LAYOUT) -> dict:
    import time
    clock = clock or time.time
    started = _iso(clock())
    lay = get_layout(layout)
    if not lay.replayable:
        raise LayoutError(f"{layout} needs live collection; it cannot be replayed")
    check_source_layout(src, source_path)

    # 1. Ask. The engine gets each stored request and nothing else.
    polls: list[dict] = []
    for sp in src["polls"]:
        obs, layout_notes = apply_layout(layout, sp["request"])
        try:
            d: Decision = engine.decide(obs)
        except Exception as e:  # noqa: BLE001 - recorded on the poll, as in live runs
            d = Decision(status=STATUS_ERROR, message=f"{type(e).__name__}: {e}")
        polls.append({"n": sp["n"], "source_seconds_since_inject": sp.get("seconds_since_inject"),
                      "request": obs, "request_bytes": len(json.dumps(obs).encode()),
                      **({"trim_notes": layout_notes} if layout_notes else {}),
                      "decision": _decision_dict(d), "_d": d})

    # 2. Score, only now reading the source's ground truth.
    gt = src["ground_truth"]
    truth = GroundTruth(incident=gt["incident"], affected_component=gt["component"],
                        diagnosis=gt["diagnosis"], acceptable_actions=tuple(gt["acceptable_actions"]))
    first = next((p for p in polls if _detects(p)), None)
    scored = first or (polls[-1] if polls else None)
    decisions = [p.pop("_d") for p in polls]
    scored_d = decisions[polls.index(scored)] if scored else None

    rec: dict = {
        "run_id": run_id, "experiment_id": src["experiment_id"], "mode": MODE,
        "source": {"path": source_path, "run_id": src.get("run_id"), "mode": src.get("mode"),
                   "prompt": record_prompt(src)[0], "observation": record_layout(src)[0],
                   "engine": src.get("engine"),
                   "scored_poll": src.get("scored_poll")},
        "fault": src.get("fault"), "ground_truth": dict(gt),
        "status": "ok", "message": "",
        "engine": engine_info(engine), "prompt": engine_prompt(engine),
        "observation": lay.identity,
        "polls": polls,
        "scored_poll": scored["n"] if scored else None,
        "jev": scored["decision"] if scored else None,
        "request": scored["request"] if scored else None,
        "evaluation": evaluate(scored_d, truth),
        "false_positive": false_positive_polls(polls, truth),
        "censored": first is None and source_truncated(src),
        "detect_poll": first["n"] if first else None,
        "replay_detect_seconds_since_inject": first["source_seconds_since_inject"] if first else None,
        "timing": {"started_at": started, "ended_at": _iso(clock()),
                   "t_inject": None, "t_detect": None, "mttd_seconds": None},
        "usage": _usage(polls),
    }
    if not polls or all(p["decision"]["status"] == STATUS_ERROR for p in polls):
        rec["status"] = "error"
        rec["message"] = polls[-1]["decision"]["message"] if polls else "source record has no polls"
    return rec


def plan(sources: list[tuple[Path, dict]]) -> dict:
    """What a replay would do: records to replay, records skipped, engine calls."""
    use = [(p, r) for p, r in sources if r.get("status") == "ok" and r.get("polls")]
    kept = {id(r) for _, r in use}
    skipped = [{"path": str(p), "experiment_id": r["experiment_id"], "status": r.get("status"),
                "reason": "status not ok" if r.get("status") != "ok" else "no polls"}
               for p, r in sources if id(r) not in kept]
    return {"replay": use, "skipped": skipped, "engine_calls": sum(len(r["polls"]) for _, r in use)}


def write_replay(rec: dict, dest: Path, source: Path) -> Path:
    """Write a replay record under the source file's own name, so `K001-2.json` replays
    `K001-2.json`. A second source with the same name (from another run directory) gets
    `-r2`, `-r3`, ..."""
    d = Path(dest) / rec["run_id"]
    d.mkdir(parents=True, exist_ok=True)
    p, n = d / Path(source).name, 1
    while p.exists():
        n += 1
        p = d / f"{Path(source).stem}-r{n}.json"
    p.write_text(json.dumps(rec, indent=2, default=str))
    return p


def replay_batch(source: str | Path, engine, results_dir: str | Path, *, run_id: str | None = None,
                 clock: Callable[[], float] | None = None, out=print,
                 layout: str = DEFAULT_LAYOUT) -> dict:
    """Replay every ok record under `source`. Writes one replay record per source record
    and a summary to `<results_dir>/replay/<run_id>/`."""
    from tier3.report import scorecard_data

    run_id = run_id or new_run_id()
    p = plan(load_sources(source))
    for path, src in p["replay"]:  # refuse before the first engine call
        check_source_layout(src, path)
    dest = Path(results_dir) / REPLAY_DIR
    records = []
    for i, (path, src) in enumerate(p["replay"], 1):
        rec = replay_record(src, engine, run_id=run_id, source_path=str(path), clock=clock, layout=layout)
        write_replay(rec, dest, path)
        records.append(rec)
        det = "detected" if rec["detect_poll"] else ("censored" if rec["censored"] else "no incident")
        out(f"[{i}/{len(p['replay'])}] {src['experiment_id']:<5} {len(rec['polls']):>2} polls  "
            f"{rec['status']:<5} {det}")
    summary = {"run_id": run_id, "mode": MODE, "source": str(source),
               "prompt": engine_prompt(engine), "observation": get_layout(layout).identity,
               "engine": engine_info(engine),
               "replayed": len(records), "skipped": p["skipped"],
               "censored": sum(1 for r in records if r["censored"]),
               "scorecard": scorecard_data(records)}
    d = dest / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
