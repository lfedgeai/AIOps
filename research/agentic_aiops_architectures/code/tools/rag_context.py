"""
TF-IDF RAG over the pinned application git repo, split by corpus (C1–C4).
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
RAG_CACHE_DIR = ROOT / "data" / "rag_cache"

# Bump when corpus globs/filters change so cached indexes rebuild (repo SHA alone is not enough).
INDEX_VERSION = "no-changelog-v1"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
MAX_FILE_BYTES = 400_000


class EmptyCorpusError(RuntimeError):
    """Raised when a requested RAG corpus has no indexable chunks (invalid eval cell)."""

CORPUS_GLOBS: dict[str, list[str]] = {
    "source": [
        "src/**/*.py",
        "src/**/*.go",
        "src/**/*.ts",
        "src/**/*.js",
        "src/**/*.rs",
        "src/**/*.java",
        "src/**/*.cs",
    ],
    "docs": [
        "README.md",
        "docs/**/*.md",
        "CONTRIBUTING.md",
        ".github/**/*.md",
        # CHANGELOG.md excluded: noise for harness-injected K8s faults (see docs/Learnings_across_runs_report.md)
    ],
    # OTEL demo has no docs/architecture tree; system shape lives in README + compose.
    "architecture": [
        "README.md",
        "CONTRIBUTING.md",
        "docs/**/*.md",
        "docs/**/*.svg",
        "docs/**/*.png",
        "**/architecture/**",
        "**/*architecture*",
        "**/*diagram*",
        "src/**/README.md",
        "compose.yaml",
    ],
    # Current demo uses compose.yaml / compose.*.yaml (not docker-compose* / kubernetes/).
    "dependencies": [
        "compose.yaml",
        "compose.*.yaml",
        "docker-compose*.yml",
        "docker-compose*.yaml",
        "kubernetes/**/*.yaml",
        "kubernetes/**/*.yml",
        "otel-config.yml",
        "**/Dockerfile",
        "**/Dockerfile.*",
        "src/otel-collector/*.yml",
        "src/otel-collector/*.yaml",
        "src/frontend-proxy/*.yaml",
        "src/frontend-proxy/*.yml",
        "src/prometheus/*.yaml",
        "src/prometheus/*.yml",
        "src/jaeger/*.yml",
        "src/jaeger/*.yaml",
    ],
}

CORPUS_ID_TO_NAME: dict[int, str] = {
    1: "source",
    2: "docs",
    3: "architecture",
    4: "dependencies",
}

_CHUNK_FILTERS: dict[str, Callable[[str, str], bool]] = {}


def _architecture_chunk_filter(path: str, text: str) -> bool:
    """Keep topology/overview material; always keep primary demo overview files."""
    pl = path.lower().replace("\\", "/")
    tl = text.lower()
    # Curated overview / service-map sources for the current OTEL demo layout
    if pl in ("readme.md", "contributing.md", "compose.yaml") or pl.endswith("/readme.md"):
        return True
    if any(x in pl for x in ("architecture", "diagram", "service-map", "components", "topology")):
        return True
    if "```mermaid" in tl or "graph " in tl or "flowchart" in tl:
        return True
    # Service topology keywords common in demo docs / compose excerpts
    if any(
        s in tl
        for s in (
            "cart",
            "checkout",
            "product-catalog",
            "opentelemetry-demo",
            "astronomy shop",
            "services:",
        )
    ):
        return True
    if pl.endswith((".svg", ".png")):
        return True
    return False


_CHUNK_FILTERS["architecture"] = _architecture_chunk_filter


def _repo_sha(repo: Path) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if r.returncode == 0 and (r.stdout or "").strip():
        return (r.stdout or "").strip()
    digest = hashlib.sha256(str(repo).encode()).hexdigest()[:12]
    return f"nosha-{digest}"


def _chunk_text(text: str, path: str) -> list[dict[str, Any]]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []
    chunks: list[dict[str, Any]] = []
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    for i in range(0, len(text), step):
        piece = text[i : i + CHUNK_SIZE].strip()
        if len(piece) < 40:
            continue
        chunks.append({"path": path, "chunk_index": len(chunks), "text": piece})
        if i + CHUNK_SIZE >= len(text):
            break
    return chunks


def _collect_chunks(repo: Path, corpus: str) -> list[dict[str, Any]]:
    patterns = CORPUS_GLOBS.get(corpus, [])
    chunk_filter = _CHUNK_FILTERS.get(corpus)
    all_chunks: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for pattern in patterns:
        for path in repo.glob(pattern):
            if not path.is_file():
                continue
            rel = str(path.relative_to(repo))
            if rel in seen_paths:
                continue
            seen_paths.add(rel)
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                if rel.lower().endswith((".svg", ".png")):
                    text = f"[binary image asset: {rel}]"
                else:
                    text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for ch in _chunk_text(text, rel):
                if chunk_filter and not chunk_filter(rel, ch["text"]):
                    continue
                ch["chunk_id"] = f"{rel}#{ch['chunk_index']}"
                ch["corpus"] = corpus
                all_chunks.append(ch)
    return all_chunks


def _cache_path(repo: Path, corpus: str) -> Path:
    sha = _repo_sha(repo)
    return RAG_CACHE_DIR / sha / INDEX_VERSION / corpus


def build_index(repo: Path, corpus: str, *, force: bool = False) -> dict[str, Any]:
    """Build or load TF-IDF index for one corpus."""
    if corpus not in CORPUS_GLOBS:
        raise ValueError(f"Unknown corpus: {corpus}")
    cache = _cache_path(repo, corpus)
    meta_file = cache / "meta.json"
    index_file = cache / "index.pkl"
    if not force and meta_file.is_file() and index_file.is_file():
        with open(meta_file, encoding="utf-8") as f:
            return json.load(f)

    from sklearn.feature_extraction.text import TfidfVectorizer

    chunks = _collect_chunks(repo, corpus)
    if not chunks:
        raise EmptyCorpusError(
            f"No indexable chunks for corpus={corpus} under {repo}. "
            "Aborting: empty corpora are invalid for context-engineering evaluation."
        )

    corpus_texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(corpus_texts)

    cache.mkdir(parents=True, exist_ok=True)
    with open(index_file, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix, "chunks": chunks}, f)

    meta = {
        "corpus": corpus,
        "repo_path": str(repo),
        "repo_sha": _repo_sha(repo),
        "chunk_count": len(chunks),
        "cache_dir": str(cache),
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(
        f"[rag] indexed {len(chunks)} chunks corpus={corpus} from {repo.name} @ {meta['repo_sha'][:8]}",
        flush=True,
    )
    return meta


def _load_index(repo: Path, corpus: str) -> tuple[Any, Any, list[dict[str, Any]]]:
    cache = _cache_path(repo, corpus)
    index_file = cache / "index.pkl"
    if not index_file.is_file():
        build_index(repo, corpus)
    with open(index_file, "rb") as f:
        data = pickle.load(f)
    return data["vectorizer"], data["matrix"], data["chunks"]


def retrieve(
    repo: Path,
    query: str,
    corpus: str,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return top-k chunks with cosine similarity scores for a corpus."""
    from sklearn.metrics.pairwise import cosine_similarity

    q = (query or "").strip()
    if not q:
        return []
    vectorizer, matrix, chunks = _load_index(repo, corpus)
    q_vec = vectorizer.transform([q])
    scores = cosine_similarity(q_vec, matrix).flatten()
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out: list[dict[str, Any]] = []
    for i in ranked[: max(1, limit)]:
        score = float(scores[i])
        if score <= 0.0:
            continue
        hit = dict(chunks[i])
        hit["score"] = round(score, 4)
        out.append(hit)
    return out


def retrieve_by_corpus_id(
    repo: Path,
    query: str,
    corpus_id: int,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    name = CORPUS_ID_TO_NAME.get(corpus_id)
    if not name:
        return []
    return retrieve(repo, query, name, limit=limit)
