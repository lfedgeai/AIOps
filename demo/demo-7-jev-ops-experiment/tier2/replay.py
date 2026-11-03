"""Replay stored Tier 2 observations under a prompt variant (design D5).

Each Tier 2 record holds the single observation the engine received (`request`, after
scrubbing, no ground truth). Replay sends it again with the chosen variant's questions
and scores the answer with tier2/evaluate.py, reading the source's ground truth only
after the engine has answered. No Docker, demo or flag change is involved. Source
records are never modified; replay records go under `<results>/replay/<run-id>/`.
Timings that replay cannot measure stay null. Tier 2 asks once per run, so there is no
censoring here.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from prompts.loader import engine_prompt, record_prompt
from tier2.engine import STATUS_ERROR, Decision
from tier2.evaluate import aggregate, evaluate, is_false_positive
from tier2.runner import new_run_id
from tier2.scenarios import GroundTruth

REPLAY_DIR = "replay"
MODE = "replay"


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds")


def engine_info(engine) -> dict:
    info = getattr(engine, "info", None)
    return dict(info) if isinstance(info, dict) else {"name": "jev", "transport": "jev-api"}


def load_sources(path: str | Path) -> list[tuple[Path, dict]]:
    """Live Tier 2 run records under `path`, sorted by path. Replay records and summaries
    are left out."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.rglob("*.json"))
    out = []
    for f in files:
        if f.name == "summary.json" or f.name.startswith("."):
            continue
        rec = json.loads(f.read_text())
        if (isinstance(rec, dict) and "experiment_id" in rec and "ground_truth" in rec
                and "polls" not in rec and rec.get("mode") != MODE):
            out.append((f, rec))
    return out


def plan(sources: list[tuple[Path, dict]]) -> dict:
    use = [(p, r) for p, r in sources if r.get("status") == "ok" and isinstance(r.get("request"), dict)]
    kept = {id(r) for _, r in use}
    skipped = [{"path": str(p), "experiment_id": r["experiment_id"], "status": r.get("status"),
                "reason": "status not ok" if r.get("status") != "ok" else "no stored request"}
               for p, r in sources if id(r) not in kept]
    return {"replay": use, "skipped": skipped, "engine_calls": len(use)}


def replay_record(src: dict, engine, *, run_id: str, source_path: str | None = None,
                  clock: Callable[[], float] | None = None) -> dict:
    clock = clock or time.time
    started = _iso(clock())
    obs = src["request"]
    try:
        d: Decision = engine.decide(obs)  # the stored observation, nothing else
    except Exception as e:  # noqa: BLE001
        d = Decision(status=STATUS_ERROR, message=f"{type(e).__name__}: {e}")

    gt = src["ground_truth"]  # read only after the engine has answered
    truth = GroundTruth(incident=gt["incident"], affected_service=gt["service"],
                        diagnosis=gt["diagnosis"], acceptable_actions=tuple(gt["acceptable_actions"]))
    rec = {
        "run_id": run_id, "experiment_id": src["experiment_id"], "mode": MODE,
        "source": {"path": source_path, "run_id": src.get("run_id"),
                   "prompt": record_prompt(src)[0]},
        "fault": src.get("fault"), "ground_truth": dict(gt),
        "status": "ok" if d.status != STATUS_ERROR else "error",
        "message": d.message if d.status == STATUS_ERROR else "",
        "engine": engine_info(engine), "prompt": engine_prompt(engine),
        "jev": {"incident_detected": d.incident_detected, "affected_service": d.affected_service,
                "diagnosis": d.diagnosis, "recommended_action": d.recommended_action,
                "confidence": d.confidence, "reasoning": d.reasoning, "status": d.status,
                "message": d.message, "raw_response": d.raw_response, "telemetry": d.telemetry},
        "request": obs,
        "evaluation": evaluate(d, truth),
        "false_positive": is_false_positive(d, truth),
        "timing": {"started_at": started, "ended_at": _iso(clock()), "t_inject": None,
                   "t_manifest": None, "t_observation": None, "t_decision": None,
                   "detection_seconds": None, "diagnosis_seconds": None, "total_seconds": None},
    }
    return rec


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
                 clock: Callable[[], float] | None = None, out=print) -> dict:
    run_id = run_id or new_run_id()
    p = plan(load_sources(source))
    dest = Path(results_dir) / REPLAY_DIR
    records = []
    for i, (path, src) in enumerate(p["replay"], 1):
        rec = replay_record(src, engine, run_id=run_id, source_path=str(path), clock=clock)
        write_replay(rec, dest, path)
        records.append(rec)
        out(f"[{i}/{len(p['replay'])}] {src['experiment_id']:<5} {rec['status']:<5} "
            f"incident={rec['jev']['incident_detected']}")
    summary = {"run_id": run_id, "mode": MODE, "source": str(source),
               "prompt": engine_prompt(engine), "engine": engine_info(engine),
               "replayed": len(records), "skipped": p["skipped"], "aggregate": aggregate(records)}
    d = dest / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
