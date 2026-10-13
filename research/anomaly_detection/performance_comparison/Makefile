SHELL := /bin/bash
PY := python3
HARNESS := $(abspath .)
OUT := $(HARNESS)/out
DATASET ?= eval

export TRAIN_LOG_ERROR_MAX ?= 3

.PHONY: help
help:
	@echo "AIOps Harness (minimal)"
	@echo "  make features DATASET=eval     # build features/train+eval using datasets/<DATASET>"
	@echo "  make visualize                 # generate HTML for latest report/features"
	@echo "  make compare                   # run detector comparison on latest features/report"

TS := $(shell date +%Y%m%d%H%M%S)

.PHONY: features
features:
	@mkdir -p "$(OUT)"
	OTEL_DATASET_DIR="$(HARNESS)/datasets/$(DATASET)" \
	OTEL_OUT_FEATURES_CSV="$(OUT)/features_$(TS).csv" \
	OTEL_OUT_REPORT_JSON="$(OUT)/report_isoforest_$(TS).json" \
	$(PY) $(HARNESS)/feature_pipeline.py

.PHONY: visualize
visualize:
	@mkdir -p "$(OUT)"
	LATEST_FEATURES="$$(ls -1t '$(OUT)'/features_*.csv | head -n 1)"; \
	LATEST_REPORT="$$(ls -1t '$(OUT)'/report_isoforest_*.json | head -n 1)"; \
	[ -n "$$LATEST_FEATURES" ] && [ -n "$$LATEST_REPORT" ] || { echo "No features/report found in $(OUT)"; exit 1; }; \
	OTEL_FEATURES_CSV="$$LATEST_FEATURES" \
	OTEL_REPORT_JSON="$$LATEST_REPORT" \
	OTEL_OUT_HTML="$(OUT)/report_isoforest_$$(date +%Y%m%d%H%M%S).html" \
	$(PY) $(HARNESS)/visualize_isoforest.py

.PHONY: compare
compare:
	@mkdir -p "$(OUT)"
	OUT_DIR="$(OUT)/ad_compare_$(TS)"; \
	mkdir -p "$$OUT_DIR"; \
	AIOPS_OUT_DIR="$$OUT_DIR" OTEL_DATASET_DIR="$(HARNESS)/datasets/$(DATASET)" OTEL_OUT_FEATURES_CSV="$$OUT_DIR/features.csv" $(PY) $(HARNESS)/feature_pipeline.py; \
	$(PY) $(HARNESS)/comparison/compare_detectors.py --features "$$OUT_DIR/features.csv" --out "$$OUT_DIR"; \
	$(PY) $(HARNESS)/aggregate_reports.py; \
	echo "Bundled outputs in $$OUT_DIR"
