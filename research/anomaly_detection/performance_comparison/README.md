## AIOps Performance Comparison

A lightweight harness to build features from OTEL demo telemetry, train an anomaly detector, and compare detectors on curated or freshly collected data.

- Build features from logs, metrics, and traces
- Train and evaluate IsolationForest (and alternatives)
- Generate HTML reports (PR/ROC, confusion matrix)
- (Optional) collect new ground-truth data from a running OTEL demo

### Repository layout

- `feature_pipeline.py` — discover samples, build features, train/evaluate IsolationForest
- `comparison/compare_detectors.py` — compare detectors (IsolationForest/EIF, COPOD, RRCF)
- `Makefile` — common tasks
- `run.py` — CLI to collect data from an OTEL demo and run decoupled pipelines
- `datasets/` — curated snapshots for offline runs:
  - `train/` — baseline windows (labels like `train##_baseline`)
  - `eval/` — mixed baselines and injected faults (labels `eval##_*`)
  - `test/` — held-out final evaluation (labels `test##_*`)
- `out/` — pipeline outputs (features, reports, comparison HTML)

## Data model and how it fits together

There are two sources of data:

- Ground truth (live collection): raw windows produced by the OTEL demo collector (logs/traces/metrics plus `metadata_*.json` describing the scenario). Not versioned in git.
- Curated datasets (offline): folders under `datasets/` that copy/symlink selected ground-truth windows for repeatable experiments.

Each window has:
- `metadata_*.json` with fields:
  - `flag`: scenario name (e.g., `cartFailure`)
  - `variant`: `"on"` if the fault was injected; `"off"` for baseline
  - `ground_truth_root_cause`: `"none"` for baselines, or the fault name when injected
  - `label`: e.g. `train01_baseline`, `eval07_cartFailure_on`, etc.
- `logs_*.txt`, `traces_*.json`, `metrics_*.json`

Training vs evaluation is determined by labels and ground truth (not folder names):
- A row is considered training if it is a baseline (`ground_truth_root_cause == "none"`) and its label starts with `train` (or contains `_train`). These rows calibrate the detector and its threshold.
- All other rows (including baselines without `train` labels and all fault windows) are used for evaluation metrics and curves.

Metrics are computed on the evaluation subset:
- Operating-point metrics at a threshold calibrated from train baselines (default quantile `0.995`): precision, recall, F1, accuracy, FPR/TNR, confusion matrix
- Curve metrics from continuous scores: PR-AUC and ROC-AUC

## Prerequisites

- Python 3.9+
- For comparison extras: `numpy pandas scikit-learn joblib pyod rrcf isotree`
- Optional (for collecting ground truth): Docker + Docker Compose, and a running OTEL demo

Install Python deps (virtualenv recommended):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel
pip install numpy pandas scikit-learn joblib pyod rrcf isotree
```

## Quickstart (offline, using curated datasets)

From this directory:

```bash
# Build features and train/evaluate on the eval split
make features DATASET=eval

# Or combine all curated splits (train + eval + test)
make features DATASET=.

# Run detector comparison on latest features/report
make compare
```

Outputs (timestamped) go to `out/`:
- `features_*.csv`, `report_isoforest_*.json`
- `ad_compare_<ts>/ad_compare.html` (detector comparison report)

## Ground truth collection with the OTEL demo

You only need this if you want fresh data from a live environment. Otherwise, use `datasets/`.

1) Start the OTEL demo (Docker Compose)

- Install Docker/Compose, then launch the demo (see the OpenTelemetry Demo docs at `https://github.com/open-telemetry/opentelemetry-demo`):

```bash
# Example (from the demo repo)
docker compose up -d
```

2) Collect windows from the running demo

Run from this directory:

```bash
# Collect non-deleting baseline windows (defaults: cooldown 60s, duration 60s)
python3 run.py collect-baselines --windows 5

# Or run the full suite (baselines + faults) driven by flagd
python3 run.py suite
```

By default, artifacts are written under:
- `research/anomaly_detection/otel-demo/otel_ground_truth_data/` (latest/current)
- or symlinked runs under `research/anomaly_detection/otel-demo/otel_ground_truth_runs/run-*/`

3) Use collected data in the pipeline

Option A: point the pipeline directly at the collected folder:

```bash
OTEL_DATASET_DIR="research/anomaly_detection/otel-demo/otel_ground_truth_data" \
OTEL_OUT_FEATURES_CSV="out/features_$(date +%Y%m%d%H%M%S).csv" \
OTEL_OUT_REPORT_JSON="out/report_isoforest_$(date +%Y%m%d%H%M%S).json" \
python3 feature_pipeline.py
```

Option B: symlink a collected run into `datasets/` for repeatability:

```bash
ln -s research/anomaly_detection/otel-demo/otel_ground_truth_runs/run-YYYYMMDDHHMMSS datasets/myrun
make features DATASET=myrun
make compare
```

### ASCII diagrams

End-to-end data collection and usage:

```
+-------------------------+           +--------------------------------------+
|  OTEL Demo (Docker)     |           |  Collector Orchestrator              |
|  services emit telemetry|  <------> |  run.py suite / collect-baselines    |
+-----------+-------------+    HTTP   +-------------------+------------------+
            |                         queries to demo     |
            | logs / traces / metrics                      |
            v                                              v
   research/anomaly_detection/otel-demo/          research/anomaly_detection/otel-demo/
   otel_ground_truth_runs/run-YYYYMMDDHHMMSS  ->  otel_ground_truth_data (symlink to latest)
                      (raw ground truth: logs_*.txt, traces_*.json, metrics_*.json, metadata_*.json)

                                   (curate/symlink/copy)
                                              |
                                              v
                        research/anomaly_detection/performance_comparison/datasets/
                               ├── train/   (baseline windows: labels like train##_baseline)
                               ├── eval/    (mix of baselines + faults)
                               └── test/    (held-out mix for final reporting)

                                              |
                                              v
                        feature_pipeline.py  (discover → features → train/eval → report)
                                   |
                                   v
                               out/features_*.csv, out/report_isoforest_*.json
                                   |
                                   v
                        comparison/compare_detectors.py  (metrics/curves per detector)
                                   |
                                   v
                               out/ad_compare_<ts>/ad_compare.html
```

How training vs evaluation is determined:

```
                                 Selected DATASET path (e.g., datasets/.)
                                                |
                                                v
                              Discover metadata_*.json recursively
                                                |
          +------------------ in_training = True if -------------------+
          |  (ground_truth_root_cause == "none") AND                   |
          |  (label starts with "train" OR contains "_train")          |
          +------------------------------------------------------------+
                 | yes                                   | no
                 v                                       v
           TRAIN (baselines)                         EVALUATION (baselines + faults)
           - Fit IsolationForest                     - Compute scores
           - Calibrate threshold (e.g., 0.995)       - Thresholded metrics (Precision/Recall/F1/Accuracy/FPR/TNR)
                                                     - Curves and AUCs (PR-AUC, ROC-AUC)
```

## Tuning and troubleshooting

- Not enough training baselines: ensure the selected dataset includes ≥5 baseline windows with labels starting `train` (e.g., run with `DATASET=.` or collect more with `run.py collect-baselines`).
- Training tolerance: `TRAIN_LOG_ERROR_MAX` (default `3`) controls how noisy a baseline can be and still be used for training.
- Threshold: change operating point with `THRESHOLD_QUANTILE` (e.g., `export THRESHOLD_QUANTILE=0.997`), then rebuild and compare.

