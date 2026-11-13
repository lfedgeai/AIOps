"""BM25 context corpora (C1–C4) — real retrieval for the enrichment slot.

Purpose
-------
Replace the placeholder enrichment with actual corpus retrieval, using BM25
(the eval harness used TF-IDF; BM25 is the requested upgrade). Corpora are
slices of a git-pinned OpenTelemetry demo checkout, matching the eval
definitions:

  C1 source        src/**/*.{py,go,ts,js,rs,java,cs}
  C2 docs          README.md, docs/**/*.md, CONTRIBUTING.md, .github/**/*.md
  C3 architecture  topology material (architecture/diagram/service-map files)
  C4 dependencies  compose*.yaml, kubernetes/**, **/Dockerfile*, collector configs

Chunks: 900 chars / 120 overlap (eval parity). Indexes are built lazily per
context set and cached in-process.

How it connects
---------------
- ``slots/enrichment.py`` retrieves top-k chunks for the enabled sets
  (UI C1–C4 buttons → ``PUT /context``) and embeds the text in the prompt.
- Corpus root is configurable (``enrichment.corpus_root`` in the assembly
  YAML); falls back to placeholder labels when the corpus is absent.
"""

from __future__ import annotations

import fnmatch
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
DEFAULT_K = 5

DEFAULT_CORPUS_ROOT = (
    "/home/redhat/git/LFEdge/AIOps/research/anomaly_detection/"
    "performance_comparison/chaos_engineering/otel-demo"
)

CONTEXT_GLOBS: dict[str, dict[str, list[str]]] = {
    "1": {
        "include": ["src/**/*.py", "src/**/*.go", "src/**/*.ts", "src/**/*.js",
                     "src/**/*.rs", "src/**/*.java", "src/**/*.cs"],
        "exclude": [],
        "label": "source",
    },
    "2": {
        "include": ["README.md", "docs/**/*.md", "CONTRIBUTING.md", ".github/**/*.md"],
        "exclude": ["**/CHANGELOG.md"],
        "label": "docs",
    },
    "3": {
        "include": ["*architecture*", "*diagram*", "*service-map*", "docker-compose*.yaml",
                     "docker-compose*.yml", "**/*architecture*", "**/*diagram*"],
        "exclude": [],
        "label": "architecture",
    },
    "4": {
        "include": ["compose*.yaml", "compose*.yml", "kubernetes/**", "**/Dockerfile*",
                     "src/otelcollector/*.yaml", "prometheus*.yaml"],
        "exclude": [],
        "label": "dependencies",
    },
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _chunks_of(path: Path) -> list[str]:
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return []
    chunks = []
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    for i in range(0, max(1, len(text)), step):
        chunk = text[i : i + CHUNK_SIZE]
        if chunk.strip():
            chunks.append(chunk)
        if i + CHUNK_SIZE >= len(text):
            break
    return chunks


def _collect_files(root: Path, spec: dict[str, list[str]]) -> list[Path]:
    files: list[Path] = []
    seen = set()
    for pattern in spec["include"]:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            if rel in seen:
                continue
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(p.name, pattern):
                excluded = any(fnmatch.fnmatch(rel, ex) or fnmatch.fnmatch(p.name, ex)
                               for ex in spec.get("exclude", []))
                if not excluded and p.stat().st_size < 2_000_000:
                    seen.add(rel)
                    files.append(p)
    return files[:400]  # cap per corpus


class BM25Index:
    """Minimal BM25 (k1=1.5, b=0.75) over pre-chunked text."""

    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.doc_tokens = [_tokenize(c) for c in chunks]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 1.0
        self.df: Counter = Counter()
        for toks in self.doc_tokens:
            self.df.update(set(toks))
        self.n = len(chunks)

    def search(self, query: str, k: int = DEFAULT_K) -> list[tuple[int, float]]:
        q_tokens = _tokenize(query)
        if not q_tokens or not self.n:
            return []
        k1, b = 1.5, 0.75
        scores: list[float] = [0.0] * self.n
        for term in q_tokens:
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i in range(self.n):
                tf = self.doc_tokens[i].count(term)
                if tf == 0:
                    continue
                denom = tf + k1 * (1 - b + b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * tf * (k1 + 1) / denom
        ranked = sorted(range(self.n), key=lambda i: -scores[i])
        return [(i, scores[i]) for i in ranked[:k] if scores[i] > 0]


_INDEX_CACHE: dict[str, Optional[BM25Index]] = {}


def get_index(context_set: str, corpus_root: Optional[str] = None) -> Optional[BM25Index]:
    if context_set in _INDEX_CACHE:
        return _INDEX_CACHE[context_set]
    root = Path(corpus_root or DEFAULT_CORPUS_ROOT)
    if not root.is_dir():
        _INDEX_CACHE[context_set] = None
        return None
    spec = CONTEXT_GLOBS.get(context_set)
    if spec is None:
        _INDEX_CACHE[context_set] = None
        return None
    files = _collect_files(root, spec)
    chunks: list[str] = []
    for f in files:
        prefix = f.relative_to(root).as_posix()
        for c in _chunks_of(f):
            chunks.append(f"[{prefix}] {c}")
    idx = BM25Index(chunks) if chunks else None
    _INDEX_CACHE[context_set] = idx
    return idx


def retrieve(
    context_sets: list[str], query: str, k: int = DEFAULT_K
) -> dict[str, list[str]]:
    """Top-k BM25 chunks per enabled context set. Missing corpora → labels."""
    out: dict[str, list[str]] = {}
    for cs in context_sets:
        idx = get_index(cs)
        if idx is None:
            out[cs] = [f"C{cs} ({CONTEXT_GLOBS[cs]['label']}): corpus not available — placeholder"]
            continue
        hits = idx.search(query, k=k)
        out[cs] = [idx.chunks[i] for i, _ in hits] or [
            f"C{cs} ({CONTEXT_GLOBS[cs]['label']}): no matching chunks"
        ]
    return out


# ---------------------------------------------------------------------------
# Active context-set state (UI C1–C4 buttons → PUT /context)
# ---------------------------------------------------------------------------

import threading

_ACTIVE_LOCK = threading.Lock()
_ACTIVE: set[str] = {"2", "3", "4"}  # default: high-RCA contexts per Oct-26


def get_active_sets() -> list[str]:
    with _ACTIVE_LOCK:
        return sorted(_ACTIVE)


def set_active_sets(sets: list[str]) -> list[str]:
    valid = {s for s in (str(x) for x in sets) if s in CONTEXT_GLOBS}
    with _ACTIVE_LOCK:
        _ACTIVE.clear()
        _ACTIVE.update(valid or {"2"})
        return sorted(_ACTIVE)


__all__ = [
    "CONTEXT_GLOBS",
    "BM25Index",
    "retrieve",
    "get_index",
    "get_active_sets",
    "set_active_sets",
    "DEFAULT_CORPUS_ROOT",
]
