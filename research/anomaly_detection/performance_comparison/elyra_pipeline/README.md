# Elyra Pipeline: Isolation Forest on RHOAI

Five Jupyter notebooks for an Elyra deployment pipeline on Red Hat OpenShift AI (RHOAI). **Fully self-contained on RHOAI** — no external `feature_pipeline.py` required.

1. **00_prepare_features.ipynb** — Extract features from raw OTEL datasets (runs inside the pipeline)
2. **01_train_isolation_forest.ipynb** — Train the Isolation Forest model
3. **02_test_model.ipynb** — Test the model and report results
4. **03_copy_to_s3.ipynb** — Copy the model and artifacts to S3 (requires AWS credentials)
5. **04_deploy_to_serving.ipynb** — Deploy from S3 to RHOAI model serving (KServe)

## How Training Data Is Used

The pipeline extracts features from raw OTEL datasets **inside node 00**. No external scripts needed.

### Data flow (all inside the pipeline)

1. **Node 00 — Prepare Features** — Reads `OTEL_DATASET_DIR` (path to datasets):
   - Discovers samples via `metadata_*.json` (train, eval, test patterns)
   - Extracts features from `logs_*.txt`, `traces_*.json`, `metrics_*.json`
   - Writes `features.csv` with `in_training`, `root_cause`, and numeric features
   - **If no datasets exist**: generates synthetic features for pipeline demo
2. **Node 01 — Train** — Loads `features.csv`, uses rows with `in_training == True`, fits Isolation Forest
3. **Node 02 — Test** — Uses rows with `in_training == False`; `root_cause != "none"` = anomaly for metrics

### RHOAI: Datasets from git clone

The datasets are part of this repo at `research/anomaly_detection/performance_comparison/datasets`. On RHOAI:

1. **Clone the repo** into your workbench or a shared PVC (e.g. `git clone <repo-url>`).
2. **Run the pipeline** from the `elyra_pipeline/` directory (or set `OTEL_DATASET_DIR` to the datasets path).
3. Default `OTEL_DATASET_DIR=../datasets` resolves to `performance_comparison/datasets` when running from `elyra_pipeline/`.
4. If the clone root differs, set `OTEL_DATASET_DIR` to the full path, e.g. `/path/to/clone/research/anomaly_detection/performance_comparison/datasets`.

### Local runs

Locally, run 00 with `OTEL_DATASET_DIR=../datasets` (default), or run `feature_pipeline.py` first and skip node 00.

---

## Quick Start

### Run notebooks locally

```
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in order (00 → 01 → 02 → 03 → 04). Set `OTEL_DATASET_DIR=../datasets` for node 00 to use existing data. If no datasets exist, node 00 generates synthetic features.

### Run as Elyra Pipeline on RHOAI

1. **Upload the notebooks** to your RHOAI Jupyter environment (Kubeflow Notebook Server with Elyra).
2. **Create a pipeline** using the Elyra visual editor, or import `isolation_forest_pipeline.pipeline`.
3. **Connect nodes**: `00_prepare` → `01_train` → `02_test` → `03_copy_to_s3` → `04_deploy_to_serving`.
4. **Configure pipeline parameters**:

   | Node | Parameter | Description |
   |------|------------|-------------|
   | 00_prepare | `OTEL_DATASET_DIR` | Path to datasets from git clone (default: `../datasets`) |
   | 03_copy | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET` | S3 credentials and bucket |
   | 04_deploy | `INFERENCE_SERVICE_NAMESPACE` | OpenShift project for serving |
   | 04_deploy | `S3_BUCKET`, `S3_PREFIX` | S3 location (or `S3_MODEL_URI`) |

5. **AWS credentials**: Store in an OpenShift Secret and map as environment variables for the copy node.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_DATASET_DIR` | `../datasets` | Path to datasets from git clone (`performance_comparison/datasets`) |
| `OUT_FEATURES_CSV` | `artifacts/features.csv` | Features output from node 00 |
| `TRAIN_DATA_PATH` | (from pipeline) | Training features CSV |
| `TEST_DATA_PATH` | (from pipeline) | Test features CSV |
| `MODEL_OUTPUT_PATH` | `artifacts/model.pkl` | Model save path |
| `MODEL_PATH` | `artifacts/model.pkl` | Model load path |
| `FEATURE_COLS_PATH` | `artifacts/feature_cols.json` | Feature columns metadata |
| `THRESHOLD_PATH` | `artifacts/threshold.json` | Calibrated threshold |
| `TEST_REPORT_PATH` | `artifacts/test_report.json` | Test metrics report |
| `AWS_ACCESS_KEY_ID` | (required for copy) | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | (required for copy) | AWS secret key |
| `S3_BUCKET` | (required for copy) | S3 bucket name |
| `S3_PREFIX` | `models/isolation-forest/` | S3 object key prefix |
| `S3_MODEL_URI` | (from 03) | Full S3 URI for deploy node |
| `INFERENCE_SERVICE_NAME` | `isolation-forest` | KServe InferenceService name |
| `INFERENCE_SERVICE_NAMESPACE` | (required for deploy) | OpenShift project |

## Where the Model Is Stored

| Stage | Location | Description |
|-------|----------|-------------|
| **During pipeline** | `/tmp/artifacts/model.pkl` | Ephemeral; passed between nodes 01 → 02 → 03 |
| **After copy (03)** | `s3://<bucket>/models/isolation-forest/model.pkl` | Persistent storage in your S3 bucket |
| **In serving (04)** | KServe loads from S3 | Model is pulled from S3 into the inference pod at deploy time; no separate copy is stored in RHOAI |

Artifacts copied to S3: `model.pkl`, `feature_cols.json`, `threshold.json`, `test_report.json`.

## Data Format

The features CSV should have:
- **Numeric columns** used as features (e.g. `trace_count`, `log_error_count`, `metrics_series_len`, etc.)
- **`in_training`** (bool): `True` for training rows (baseline samples)
- **`root_cause`**: `"none"` for normal, other values for anomaly (used for test evaluation)

## Local Pipeline Test

Run the core pipeline (00–02) locally:

```bash
pip install -r requirements.txt
python run_pipeline_test.py
```

Uses synthetic data if `OTEL_DATASET_DIR` is unset. Use real datasets:

```bash
OTEL_DATASET_DIR=../datasets python run_pipeline_test.py
```

Notebooks 03 and 04 are skipped without AWS credentials and cluster access (expected).

## OpenShift / RHOAI Compatibility Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| **Datasets** | ✓ | Git clone provides `../datasets`; set `OTEL_DATASET_DIR` if path differs |
| **Elyra + Kubeflow Pipelines** | ✓ | Pipeline uses `runtime_ref: kfp`; compatible with RHOAI Elyra |
| **Runtime image** | ✓ | `quay.io/opendatahub/elyra-runtime-image:2024.1` (or your RHOAI image) |
| **Artifact passing** | ✓ | Uses `/tmp/artifacts/` paths; Kubeflow passes outputs between nodes |
| **Node 03 (S3)** | ✓ | Needs `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET` from RHOAI secrets |
| **Node 04 (KServe)** | ✓ | Needs `oc`/`kubectl` in pod, cluster RBAC, S3 credentials for predictor |
| **KServe sklearn** | ✓ | MLServer sklearn runtime accepts both `model.pkl` and `model.joblib` |

**RHOAI setup**: Clone the repo, configure pipeline env vars (especially AWS + `INFERENCE_SERVICE_NAMESPACE`), ensure the runtime image has `pandas`, `scikit-learn`, `joblib`, `boto3`, and `oc`/`kubectl`.

## Dependencies

- `scikit-learn`: Isolation Forest (or `isotree` for Extended Isolation Forest)
- `boto3`: S3 copy
- `pandas`, `numpy`, `joblib`: Data and serialization
