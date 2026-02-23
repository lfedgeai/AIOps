# 🔬 Project Title: Performance comparison of Anomaly Detection approaches

> **Executive Summary:** A one-sentence pitch of what this research solves and why it matters to the world.

---
For all your approved research tasks, please use the below template in the top-level directory of your research project.


## 📌 Research Overview
* **Status:** `🏗️ In-Progress` 
* **Domain:** Anomaly Detection
* **Core Question:** What is the best approach for anomaly detection in distributed and evolving environments?

---

## 🎯 Desired Outcomes
* **Primary Objective:** to prove out that the winning approach is best suitable in F1 score across evolving distributed system landscapes
* **Key Deliverables:**
    - [ ] **Trained Weights:** Optimized model files for deployment.
    - [X] **Cleaned Dataset:** Normalized data ready for replication.
    - [X] **Blog Post:** Formal documentation of findings.
    - [X] **Visual Dashboard:** Interactive UI for results visualisation.

---

## 🛠 Methodology & Framework
*Describe the technical approach and the "How" behind the research.*
I am using the OTEL demo app to generate data, train models, develop different feature sets and compare different algorithms and AI models (predictive) comparing the results calculating precision, recall, F1 score and Area Under the Curve (AUC).

* **Approach:** collect ground truth data to train model and algortihm, then evaluate performance with a separate data set
* **Core Logic:** ./comparison/compare_detectors.py
* **Tech Stack:** OTEL Demo App, Python (Pandas/NumPy), Docker/Podman, Pytorch, Make

---

## 📊 Data Management & Transparency
1.  **Source:** OTEL Demo App logs, metrics & traces
2.  **Processing:** How was the data cleaned? /chaos_engineering/collect* scripts
3.  **Ethical Considerations:** OTEL demo app data allows for repeatible, ethical, copyright free data collection.

---

## 📂 Repository Structure
```text
├── chaos_engineering/    # collect data and inject faults
├── copmarison/           # copmares different algorithms and models
├── datasets/             # previously collected datasets (static)
├── results/              # Graphs, Tables, and Model Outputs
├── out/                  # reports and rolling reports, models, and other artefacts created or updated during each run.
└── README.md             # This file
└── Makefile              # A makefile for ease of orchestrating runs

```

#Project Details
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
- `chaos_engineering/chaos_orchestrator.py` — CLI to collect data from an OTEL demo and run decoupled pipelines (fault injection, suite orchestration)
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
python3 chaos_engineering/chaos_orchestrator.py collect-baselines --windows 5

# Or run the full suite (baselines + faults) driven by flagd
python3 chaos_engineering/chaos_orchestrator.py suite
```

By default, artifacts are written under:
- `research/anomaly_detection/performance_comparison/chaos_engineering/otel-demo/otel_ground_truth_data/` (latest/current)
- or symlinked runs under `research/anomaly_detection/performance_comparison/chaos_engineering/otel-demo/otel_ground_truth_runs/run-*/`

#### Recommended settings and tips for new collections

- Baselines for training: collect at least 5–10 baseline windows.
- Window sizing: use `--duration 60–90` seconds and `--cooldown >= duration` to avoid overlap.
- Load generator: ensure the demo’s load generator is running (so telemetry is not empty).
- Backend health: verify Prometheus, OpenSearch, and OTEL Collector are up before starting.
- Stability: avoid restarting services mid-run; keep system load stable to reduce noise.
- Storage: each run writes logs/traces/metrics per window; plan for hundreds of MB across full suites.

Examples:

```bash
# 10 clean baselines, 60s cooldown + 60s window
python3 chaos_engineering/chaos_orchestrator.py collect-baselines --windows 10 --cooldown 60 --duration 60

# Full suite with injected faults + baselines (takes longer)
python3 chaos_engineering/chaos_orchestrator.py suite
```

3) Use collected data in the pipeline

Option A: point the pipeline directly at the collected folder:

```bash
OTEL_DATASET_DIR="research/anomaly_detection/performance_comparison/chaos_engineering/otel-demo/otel_ground_truth_data" \
OTEL_OUT_FEATURES_CSV="out/features_$(date +%Y%m%d%H%M%S).csv" \
OTEL_OUT_REPORT_JSON="out/report_isoforest_$(date +%Y%m%d%H%M%S).json" \
python3 feature_pipeline.py
```

Option B: symlink a collected run into `datasets/` for repeatability:

```bash
ln -s research/anomaly_detection/performance_comparison/chaos_engineering/otel-demo/otel_ground_truth_runs/run-YYYYMMDDHHMMSS datasets/myrun
make features DATASET=myrun
make compare
```

## From ground truth to train/eval/test (curation guide)

Ground truth windows are timestamped and live-collected. Curating creates stable, versionable splits you can reuse.

1) Identify baselines vs faults from each window’s `metadata_*.json`:
- Baseline: `ground_truth_root_cause == "none"` and `variant == "off"`
- Fault: `ground_truth_root_cause != "none"` (usually `variant == "on"`)

2) Create curated folders:
```bash
mkdir -p datasets/train datasets/eval datasets/test
```

3) Put baselines into `datasets/train/` and label them `train##_baseline` (names matter for training detection). You can copy or symlink:
```bash
# Copy a baseline window
cp path/to/run/metadata_train01_baseline.json datasets/train/
cp path/to/run/logs_train01_baseline.txt     datasets/train/
cp path/to/run/traces_train01_baseline.json  datasets/train/
cp path/to/run/metrics_train01_baseline.json datasets/train/

# Or symlink instead of copying
ln -s ../../otel-demo/otel_ground_truth_runs/run-YYYY.../metadata_train01_baseline.json datasets/train/
ln -s ../../otel-demo/otel_ground_truth_runs/run-YYYY.../logs_train01_baseline.txt     datasets/train/
```

4) Put a mix of baselines and faults into `datasets/eval/` and `datasets/test/` (e.g., earlier runs → eval, later runs → test). Use labels like `eval##_...` and `test##_...`.

5) Verify:
```bash
make features DATASET=.   # combines train+eval+test
make compare
```

Notes:
- Training rows are determined by both baseline status and the label starting with `train`; everything else (including non-train baselines) is evaluated.
- Symlinks work the same as copies for the pipeline and avoid duplication.

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
   research/anomaly_detection/performance_comparison/chaos_engineering/otel-demo/          research/anomaly_detection/performance_comparison/chaos_engineering/otel-demo/
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

## Reports and how they’re generated

There are two primary HTML reports:

- IsolationForest report (single-detector), produced by `feature_pipeline.py`:
  - Inputs: features built from the selected `DATASET`, training rows detected from metadata.
  - Steps:
    1) Fit IsolationForest on TRAIN baseline rows.
    2) Calibrate a threshold from train scores at `THRESHOLD_QUANTILE` (default `0.995`).
    3) Score EVAL rows; compute:
       - Confusion matrix and operating-point metrics at the calibrated threshold: Precision, Recall, F1, Accuracy, FPR, TNR.
       - PR and ROC curves and their AUCs using continuous anomaly scores.
  - Outputs (under `out/`):
    - `features_*.csv` (all rows and feature columns)
    - `report_isoforest_*.json` (metrics, curves, threshold, feature columns)
    - `report_isoforest_*.html` (if you run `make visualize`)

- Detector comparison report, produced by `comparison/compare_detectors.py` (via `make compare`):
  - Inputs: the latest `features_*.csv` and `report_isoforest_*.json` (for column order).
  - For each detector:
    1) Train on TRAIN rows (baselines).
    2) Calibrate threshold from TRAIN scores at the same quantile.
    3) Score EVAL rows; compute operating-point metrics and PR/ROC AUCs.
  - Output: `out/ad_compare_<ts>/ad_compare.html` and `ad_compare.json` (+ model artifacts).

### Feature set (high level)
Extracted from logs/traces/metrics per window (examples):
- Traces: total spans, 5xx spans, unique services/span names
- Logs: total lines, error counts per service, error ratio
- Metrics: series counts, simple scalar sums (e.g., request rate)

## Anomaly detection algorithms included

- IsolationForest (sklearn)
  - Unsupervised, tree-based isolation of anomalies.
  - Scores are `-decision_function(X)` so higher means “more anomalous”.
  - Used as the baseline model; model file saved to `out/IsolationForestModel_<ts>.pkl`.
  - Intuition: random splits isolate rare points in fewer splits (shorter path length ⇒ more anomalous).
  - Complexity: roughly O(n_estimators · n_samples · log n_samples) for training.
  - Strengths: simple, fast, robust for tabular data; few hyperparameters.
  - Limitations: axis-aligned splits can underperform when anomalies lie along oblique directions.

- Extended Isolation Forest (isotree)
  - If `isotree` is installed, uses `isotree.IsolationForest`; otherwise falls back to sklearn IF.
  - Uses `predict_scores(X)` where higher means “more anomalous”.
  - Difference vs IF: uses extended (oblique) hyperplanes, often improving separability on correlated features.
  - Strengths: better geometry for high-dimensional or correlated features; strong default behavior.
  - Consider when: sklearn IF misses anomalies that are not axis-aligned.

- COPOD (pyod)
  - Uses empirical copulas and outlier degrees.
  - Higher `decision_function` values indicate more anomalous points.
  - Intuition: models each feature’s empirical distribution and tail behavior, combining via copulas.
  - Strengths: parameter-light, nonparametric; can work well when feature scales/distributions differ.
  - Consider when: you prefer distribution-free methods with minimal tuning.

- RRCF (Robust Random Cut Forest)
  - Forest of random cut trees; anomaly score is CoDisp (average displacement), higher is more anomalous.
  - This implementation normalizes features (median/MAD), builds trees on train, and scores by temporary insertions.
  - Strengths: supports streaming variants; good at capturing sudden changes and point-level anomalies.
  - Limitations: batch scoring can be slower; sensitive to feature scaling (we apply robust scaling).
  - Consider when: you value streaming friendliness or want a different tree-based perspective than IF.

All detectors:
- Train only on TRAIN baseline rows.
- Calibrate threshold at a quantile of TRAIN scores (default `0.995`).
- Evaluate on EVAL rows (baselines + faults) for metrics and curves.

### Configuration knobs
- `THRESHOLD_QUANTILE` (env): operating-point sensitivity (default `0.995`).
- `TRAIN_LOG_ERROR_MAX` (env): tolerates some noisy baselines when few clean samples exist (default `3`).
- `DATASET` (Make): which dataset subtree to scan (`eval`, `test`, `train`, or `.`).

## Tuning and troubleshooting

- Not enough training baselines: ensure the selected dataset includes ≥5 baseline windows with labels starting `train` (e.g., run with `DATASET=.` or collect more with `run.py collect-baselines`).
- Training tolerance: `TRAIN_LOG_ERROR_MAX` (default `3`) controls how noisy a baseline can be and still be used for training.
- Threshold: change operating point with `THRESHOLD_QUANTILE` (e.g., `export THRESHOLD_QUANTILE=0.997`), then rebuild and compare.

