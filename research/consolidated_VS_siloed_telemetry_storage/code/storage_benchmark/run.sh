#!/usr/bin/env bash
# Fresh stack -> generate -> wait -> benchmark. Usage: ./run.sh [traces] [runs per round] [rounds] [seed]
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out
docker run --rm -u "$(id -u):$(id -g)" -e GOCACHE=/tmp/gc -e CGO_ENABLED=0 -v "$PWD/correlate_go:/src:ro" -v "$PWD/out:/out" -w /src golang:1.27-alpine sh -c 'go vet ./... && go build -o /out/correlate .'
docker compose down -v >/dev/null 2>&1
docker compose up -d >/dev/null 2>&1
for _ in $(seq 60); do curl -sf -o /dev/null http://127.0.0.1:33100/ready && curl -sf -o /dev/null http://127.0.0.1:33200/ready && curl -sf -X POST -H 'Content-Type: application/json' -d '{}' -o /dev/null http://127.0.0.1:34318/v1/logs && break; sleep 2; done
n=${1:-20000}; seed=${4:-45}; tag="${n}_s${seed}"
python3 generate.py --traces "$n" --seed "$seed" --out "out/corpus_$tag.json"
sleep "${SETTLE:-45}"   # let the stores cut blocks and merge parts
python3 bench.py --corpus "out/corpus_$tag.json" --runs "${2:-10}" --rounds "${3:-4}" --out "out/results_$tag.json"
