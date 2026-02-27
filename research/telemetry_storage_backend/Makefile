SHELL := /bin/bash
ROOT  := $(abspath .)
OUT   := $(ROOT)/out
DATA_DIR ?= $(ROOT)/telemetry_data

.PHONY: up down up-compare schema load query bench bench-compare

up:
	docker compose up -d doris
	@echo "Waiting for Doris health..."
	@for i in {1..60}; do \
	  curl -sf http://localhost:8030/api/health && echo "Doris is healthy" && break || sleep 2; \
	done

up-compare:
	docker compose up -d
	@echo "Waiting for Doris health..."
	@for i in {1..60}; do \
	  curl -sf http://localhost:8030/api/health && echo "Doris is healthy" && break || sleep 2; \
	done
	@echo "Waiting for ClickHouse health..."
	@for i in {1..60}; do \
	  curl -sf http://localhost:8123/ping && echo "ClickHouse is healthy" && break || sleep 2; \
	done

down:
	docker compose down -v

schema:
	@echo "Applying Doris schema..."
	DORIS_USER="root" DORIS_PASS="root" python3 runner/bench.py --init --data-dir "$(DATA_DIR)"

load:
	@echo "Loading data into Doris from $(DATA_DIR)"
	DORIS_FE_HTTP="http://localhost:8030" DORIS_FE_INTERNAL_HTTP="http://127.0.0.1:8030" DORIS_USER="root" DORIS_PASS="root" python3 loaders/replay.py --data-dir "$(DATA_DIR)" --batch 5000

query:
	@echo "Running canonical queries..."
	DORIS_USER="root" DORIS_PASS="root" python3 runner/bench.py --queries --data-dir "$(DATA_DIR)"

bench:
	@mkdir -p "$(OUT)"
	DORIS_USER="root" DORIS_PASS="root" python3 runner/bench.py --all --data-dir "$(DATA_DIR)" --out "$(OUT)"

bench-compare:
	@mkdir -p "$(OUT)"
	@echo "Running Doris vs ClickHouse comparison..."
	DORIS_USER="root" DORIS_PASS="" CLICKHOUSE_PASSWORD="changeme" python3 runner/bench_compare.py --all --data-dir "$(DATA_DIR)" --out "$(OUT)"

