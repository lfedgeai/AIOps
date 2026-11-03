"""Side-by-side comparison of prompt variants (design D6, D7). Tier-neutral: a tier
supplies its records and its rows; this module groups, aligns and formats.

- Groups are (live or replay, engine, prompt name, prompt version), plus the observation
  layout when the tier has layouts (Tier 3/4). Live and replay records never share a group.
- Every group is compared over the source records all groups share. A live record's
  source is itself; a replay record's source is the record it replayed.
- Two records with the same prompt name and version but different hashes stop the
  report (`HashConflict`): the variant file was edited without a version bump.
- Each cell is `k/n (p%)`, or `n/a` for an empty denominator. No bare percentages.
- With dev and confirm sets, the confirm table comes first, and a variant written
  against records in the confirm set is flagged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from prompts.loader import ASSUMED_NOTE, record_prompt


class HashConflict(ValueError):
    pass


@dataclass
class Group:
    mode: str  # "live" or "replay"
    engine: str
    prompt: str
    version: int | None
    records: list[tuple[str, dict]] = field(default_factory=list)  # (source key, record)
    assumed: int = 0  # records read as baseline because they had no prompt field
    style: str = "wording"
    layout: str | None = None  # observation layout; None for tiers without layouts
    layout_version: int | None = None

    @property
    def label(self) -> str:
        v = f" v{self.version}" if self.version is not None else ""
        tag = ", policy stated" if self.style == "policy" else ""
        lay = ""
        if self.layout not in (None, "raw"):
            lv = f" v{self.layout_version}" if self.layout_version is not None else ""
            lay = f" + {self.layout}{lv}"
        return f"{self.engine} / {self.prompt}{v}{lay} ({self.mode}{tag})"

    def sub(self, records) -> "Group":
        return Group(self.mode, self.engine, self.prompt, self.version, records, self.assumed,
                     self.style, self.layout, self.layout_version)


def cell(pair) -> str:
    k, n = pair
    return "n/a" if not n else f"{k}/{n} ({round(100 * k / n)}%)"


def f1_cell(pair) -> str:
    k, n = pair
    return "n/a" if not n else f"{k}/{n} ({k / n:.2f})"


def _key(p) -> str:
    return str(Path(p).resolve())


def source_key(path: Path, rec: dict) -> str:
    if rec.get("mode") == "replay":
        src = (rec.get("source") or {}).get("path")
        return _key(src) if src else f"{(rec.get('source') or {}).get('run_id')}/{rec['experiment_id']}"
    return _key(path)


def _check_hash(hashes: dict, kind: str, ident: dict, path) -> None:
    if not ident.get("sha256"):
        return
    hk = (kind, ident["name"], ident["version"])
    seen = hashes.setdefault(hk, (ident["sha256"], str(path)))
    if seen[0] != ident["sha256"]:
        what = "A variant file" if kind == "prompt" else "A layout's code"
        raise HashConflict(
            f"{kind} {ident['name']!r} v{ident['version']} has two different hashes: "
            f"{seen[0][:12]}... in {seen[1]} and {ident['sha256'][:12]}... in {path}. "
            f"{what} was edited without bumping its version.")


def group_records(items: list[tuple[Path, dict]], engine_name: Callable[[dict], str],
                  layout_of: Callable[[dict], dict] | None = None) -> list[Group]:
    groups: dict[tuple, Group] = {}
    hashes: dict[tuple, tuple[str, str]] = {}
    for path, rec in items:
        ident, assumed = record_prompt(rec)
        lay = layout_of(rec) if layout_of else None
        if lay:
            _check_hash(hashes, "layout", lay, path)
        mode = "replay" if rec.get("mode") == "replay" else "live"
        k = (mode, engine_name(rec), ident["name"], ident["version"],
             lay["name"] if lay else None, lay["version"] if lay else None)
        g = groups.setdefault(k, Group(mode, k[1], k[2], k[3], style=ident.get("style", "wording"),
                                       layout=k[4], layout_version=k[5]))
        g.records.append((source_key(path, rec), rec))
        g.assumed += assumed
        _check_hash(hashes, "prompt", ident, path)
    order = {"live": 0, "replay": 1}
    return sorted(groups.values(), key=lambda g: (order[g.mode], g.prompt != "baseline",
                                                  g.layout not in (None, "raw"), g.engine,
                                                  g.prompt, g.version or 0, g.layout or ""))


def common_sources(groups: list[Group]) -> tuple[set[str], list[str]]:
    """Source keys every group covers, and warning lines naming what differs."""
    sets = [{k for k, _ in g.records} for g in groups]
    common = set.intersection(*sets) if sets else set()
    warnings = []
    for g, s in zip(groups, sets):
        extra = sorted(s - common)
        if extra:
            shown = ", ".join(Path(e).name if "/" in e else e for e in extra[:6])
            more = f" and {len(extra) - 6} more" if len(extra) > 6 else ""
            warnings.append(f"{g.label} has {len(extra)} source record(s) the other columns lack "
                            f"({shown}{more}); compared over the {len(common)} they share.")
    return common, warnings


def render_table(groups: list[Group], rows: list[tuple[str, Callable[[list[dict]], str]]],
                 keep: set[str]) -> list[str]:
    cols = [[r for k, r in g.records if k in keep] for g in groups]
    out = ["| | " + " | ".join(g.label for g in groups) + " |", "|" + "---|" * (len(groups) + 1)]
    for name, fn in rows:
        out.append(f"| {name} | " + " | ".join(fn(c) for c in cols) + " |")
    return out


def members(root: str | Path) -> set[str]:
    """Resolved paths of every record file under `root`, following symlinks (the
    `*-combined` directories are links to the original run files)."""
    r = Path(root)
    files = [r] if r.is_file() else [f for f in r.rglob("*.json") if f.name != "summary.json"]
    return {_key(f) for f in files}


def render(title: str, items: list[tuple[Path, dict]], rows, *, engine_name,
           dev: list[str] | None = None, confirm: list[str] | None = None,
           dev_records_of: Callable[[str, int | None], tuple[str, ...] | None] | None = None,
           notes: list[str] | None = None, unpaired: bool = False,
           layout_of: Callable[[dict], dict] | None = None) -> str:
    """The comparison as Markdown. `rows` are (name, fn(records) -> cell text)."""
    groups = group_records(items, engine_name, layout_of)
    out = [f"# {title}", ""]
    if not groups:
        return "\n".join(out + ["No run records found."]) + "\n"
    assumed = sum(g.assumed for g in groups)
    if assumed:
        out += [f"> {assumed} record(s) had {ASSUMED_NOTE}.", ""]

    sections: list[tuple[str, list[str] | None]] = []
    if confirm:
        sections.append(("Confirm set (not used to write variants)", list(confirm)))
    if dev:
        sections.append(("Dev set", list(dev)))
    if not sections:
        sections.append(("All records", None))

    for heading, roots in sections:
        member = set().union(*(members(x) for x in roots)) if roots else None
        sub = []
        for g in groups:
            recs = [(k, r) for k, r in g.records if member is None or k in member]
            if recs:
                sub.append(g.sub(recs))
        out += [f"## {heading}", ""]
        if roots is not None:
            out += ["Sources: " + ", ".join(f"`{x}`" for x in sorted(roots)), ""]
        if not sub:
            out += ["No records from this set.", ""]
            continue
        if unpaired:
            keep = {k for g in sub for k, _ in g.records}
            warnings = ["--unpaired: each column uses all of its own records. The columns did not "
                        "see the same observations, so differences mix the prompt (or engine) "
                        "with run-to-run noise."] if len(sub) > 1 else []
        else:
            keep, warnings = common_sources(sub)
        out += [f"> **Warning:** {w}" for w in warnings] + ([""] if warnings else [])
        if roots is not None and heading.startswith("Confirm") and dev_records_of:
            for g in sub:
                written_on = [d for d in (dev_records_of(g.prompt, g.version) or ())
                              if Path(d).exists() and members(d) & member]
                if written_on:
                    out += [f"> **Flag:** {g.label} was written on this set "
                            f"(`dev_records`: {', '.join(written_on)}). Its figures here are not a "
                            "held-out test.", ""]
        live_base = [g for g in sub if g.mode == "live" and g.prompt == "baseline" and g.layout in (None, "raw")]
        replay_base = [g for g in sub if g.mode == "replay" and g.prompt == "baseline" and g.layout in (None, "raw")]
        if live_base and replay_base:
            out += ["> Both the original live baseline and a replayed baseline are shown. Compare "
                    "variants with the replayed baseline: it was asked at the same time as them.", ""]
        out += render_table(sub, rows, keep) + [""]
    for n in notes or []:
        out += [n, ""]
    return "\n".join(out).rstrip() + "\n"
