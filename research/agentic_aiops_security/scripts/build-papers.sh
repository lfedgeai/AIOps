#!/usr/bin/env bash
# Render the working papers to HTML. Run from anywhere.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/papers"
mkdir -p "$OUT"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc is required to build the working papers." >&2
  exit 1
fi

cd "$ROOT/docs/papers"
for paper in *.md; do
  name="${paper%.md}"
  pandoc "$paper" \
    --citeproc \
    --standalone \
    -o "$OUT/$name.html"
  echo "wrote $OUT/$name.html"
done
