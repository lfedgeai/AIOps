SHELL := /bin/bash
ROOT  := $(abspath .)
OUT   := $(ROOT)/out
DATA_DIR ?= $(ROOT)/telemetry_data
BATCH ?= 5000

.PHONY: up down up-compare up-otel schema load query bench bench-compare bench-otlp

up:
	docker compose up -d doris
	@echo "Waiting for Doris health..."
	@for i in {1..60}; do \
	  curl -sf http://localhost:8030/api/health && echo "Doris is healthy" && break || sleep 2; \
	done

up-compare:
	mkdir -p druid_data/var druid_data/shared
	docker compose -f docker-compose.yml -f docker-compose.druid.yml -f docker-compose.oceanbase.yml -f docker-compose.loki.yml up -d
	@echo "Waiting for Doris health..."
	@for i in {1..60}; do \
	  curl -sf http://localhost:8030/api/health && echo "Doris is healthy" && break || sleep 2; \
	done
	@echo "Waiting for ClickHouse health..."
	@for i in {1..60}; do \
	  curl -sf http://localhost:28223/ping && echo "ClickHouse is healthy" && break || sleep 2; \
	done
	@echo "Waiting for Druid health..."
	@for i in {1..120}; do \
	  curl -sf http://localhost:8888/status/health && echo "Druid is healthy" && break || sleep 5; \
	done
	@echo "Waiting for OceanBase (port 2881, bootstrap ~3-5 min)..."
	@for i in {1..60}; do \
	  nc -z 127.0.0.1 2881 && echo "OceanBase is ready" && break || sleep 5; \
	done
	@echo "Waiting for Loki (port 3100)..."
	@for i in {1..30}; do \
	  nc -z 127.0.0.1 3100 && echo "Loki is ready" && break || sleep 2; \
	done

up-otel:
	docker compose -f docker-compose.yml -f docker-compose.druid.yml -f docker-compose.oceanbase.yml -f docker-compose.otel.yml up -d
	@echo "Waiting for Doris health..."
	@for i in {1..60}; do \
	  curl -sf http://localhost:8030/api/health && echo "Doris is healthy" && break || sleep 2; \
	done
	@echo "Waiting for ClickHouse health..."
	@for i in {1..60}; do \
	  curl -sf http://localhost:28223/ping && echo "ClickHouse is healthy" && break || sleep 2; \
	done
	@echo "Waiting for OTLP collector..."
	@for i in {1..30}; do \
	  nc -z 127.0.0.1 4317 && echo "OTLP collector ready" && break || sleep 2; \
	done

down:
	docker compose -f docker-compose.yml -f docker-compose.druid.yml -f docker-compose.otel.yml -f docker-compose.oceanbase.yml -f docker-compose.loki.yml down -v

schema:
	@echo "Applying Doris schema..."
	DORIS_USER="root" DORIS_PASS="root" python3 runner/bench.py --init --data-dir "$(DATA_DIR)"

load:
	@echo "Loading data into Doris from $(DATA_DIR)"
	DORIS_FE_HTTP="http://localhost:8030" DORIS_FE_INTERNAL_HTTP="http://127.0.0.1:8030" DORIS_USER="root" DORIS_PASS="root" python3 loaders/replay_doris.py --data-dir "$(DATA_DIR)" --batch 5000

query:
	@echo "Running canonical queries..."
	DORIS_USER="root" DORIS_PASS="root" python3 runner/bench.py --queries --data-dir "$(DATA_DIR)"

bench:
	@mkdir -p "$(OUT)"
	DORIS_USER="root" DORIS_PASS="root" python3 runner/bench.py --all --data-dir "$(DATA_DIR)" --out "$(OUT)"

bench-compare:
	@mkdir -p "$(OUT)"
	@echo "Running Doris vs ClickHouse vs Druid comparison..."
	DORIS_USER="root" DORIS_PASS="" CLICKHOUSE_HTTP="http://localhost:28223" CLICKHOUSE_PASSWORD="changeme" \
	LOKI_HTTP="http://localhost:3100" \
	python3 runner/bench_compare.py --all --data-dir "$(DATA_DIR)" --out "$(OUT)" --batch $(BATCH) $(SCALE_ARGS) $(STREAMING_ARGS)

# Scaling: BATCH=10000 make bench-compare
# Scale to 50k rows per type: make bench-compare SCALE_TO=50000
# Or: DATA_DIR=/path/to/larger/data make bench-compare
SCALE_TO ?=
SCALE_ARGS := $(if $(SCALE_TO),--scale-to $(SCALE_TO),)
STREAMING_BATCH ?=
STREAMING_ARGS := $(if $(STREAMING_BATCH),--streaming-batch $(STREAMING_BATCH),)

bench-otlp:
	@mkdir -p "$(OUT)"
	@echo "Running benchmark with OTLP ingestion ($(OTLP_COUNT) spans, $(OTLP_COUNT) logs, $(OTLP_COUNT) metrics)..."
	DORIS_USER="root" DORIS_PASS="" CLICKHOUSE_HTTP="http://localhost:28223" CLICKHOUSE_PASSWORD="changeme" python3 runner/bench_compare.py \
	  --data-dir "$(DATA_DIR)" --out "$(OUT)" --all --otlp --otlp-count $(OTLP_COUNT)
OTLP_COUNT ?= 1000

bench-clickhouse:
	@mkdir -p "$(OUT)"
	@echo "Running ClickHouse-only benchmark (use when Doris BE not ready)..."
	CLICKHOUSE_HTTP="http://localhost:28223" CLICKHOUSE_PASSWORD="changeme" python3 runner/bench_compare.py --all --clickhouse-only --data-dir "$(DATA_DIR)" --out "$(OUT)"

