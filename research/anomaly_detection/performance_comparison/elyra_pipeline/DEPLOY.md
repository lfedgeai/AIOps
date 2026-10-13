# Step-by-Step: Deploy Isolation Forest Pipeline onto RHOAI

Follow these steps in order to run the Elyra pipeline on Red Hat OpenShift AI.

---

## Elyra Pipeline Troubleshooting

### "Invalid runtime type 'local'" when opening the pipeline

**Fix:** Open `isolation_forest_pipeline_local.pipeline` instead. It uses `runtime_ref: "local"`.

- **For RHOAI deployment:** After editing, change `runtime_ref` to `"kfp"` before running (or use `isolation_forest_pipeline.pipeline` which is already set for KFP).
- **Alternative:** Configure the Kubeflow Pipelines runtime in Elyra: **Metadata** → **Pipeline Runtimes** → add KFP with your Data Science Pipelines API endpoint.

### "Property runtime image is required" when dragging notebooks onto an empty pipeline

Each pipeline node needs a **Runtime image**. For each node you add:

1. Double-click the node (or right-click → **Properties**)
2. Find **Runtime image** (or **Runtime configuration**)
3. Set it to your workbench's runtime image—e.g. select from the dropdown, or use: `quay.io/opendatahub/elyra-runtime-image:2024.1`

On RHOAI, use the runtime image from your workbench's image list (e.g. the JupyterLab + TensorFlow image you selected).

---

## Phase 1: Prepare the Cluster

### Step 1: Install RHOAI

Ensure your OpenShift cluster has RHOAI installed with Workbenches and KServe:

```bash
./install-rhoai.sh
```

**Prerequisites** (must pass before install):
- OpenShift 4.20 or 4.21
- Identity provider configured; cluster-admin user (kubeadmin not allowed)
- Default storage class
- SNO: 32+ CPU, 128+ GiB RAM (or 2+ workers with 8 CPU, 32 GiB each)

Wait until the DataScienceCluster is Ready:

```bash
oc get dsc -n redhat-ods-applications
oc get pods -n redhat-ods-applications
```

### Step 2: Configure Object Storage (S3)

The pipeline needs S3-compatible storage for:
- **Node 03**: Copy model artifacts to S3
- **Node 04 / KServe**: Load model from S3 into the inference pod

**Option A — RHOAI's internal object storage** (recommended):

1. In RHOAI: **Data Science Projects** → your project → **Data connections**
2. Create an **S3-compatible** connection using the cluster's object storage (MinIO, Ceph, ODF, etc.)
3. Note the bucket name, endpoint, access key, and secret
4. Ensure the bucket exists and the credentials have read/write access

**Option B — External S3** (AWS, MinIO, etc.):

Use your existing S3 bucket, endpoint, and credentials.

### Step 3: Create a Data Science Project

1. In RHOAI: **Data Science Projects** → **Create data science project**
2. Name it (e.g. `anomaly-detection`)
3. Add yourself as a user with appropriate roles
4. Create the project

---

## Phase 2: Set Up the Workbench

### Step 4: Launch a Workbench (Notebook Server)

1. In your Data Science Project: **Workbenches** → **Create workbench**
2. Choose an image that includes **Elyra** (e.g. "Jupyter - Minimal notebook")
3. Attach the S3 data connection from Step 2 (if RHOAI supports attaching connections to workbenches)
4. Allocate resources (e.g. 2 CPU, 4 GiB)
5. Create and wait for the workbench to be Running

### Step 5: Clone the Repository

1. Open the workbench (click **Open**)
2. In JupyterLab, open a terminal
3. Clone this repo:

```bash
cd /opt/app-root/src  # or your workdir
git clone <your-repo-url> AIOps
cd AIOps/research/anomaly_detection/performance_comparison/elyra_pipeline
```

If the repo is already available (e.g. from a PVC or another clone), adjust the path and set `OTEL_DATASET_DIR` accordingly in the pipeline.

### Step 6: Verify Datasets (Optional)

If using real OTEL datasets:

```bash
ls ../datasets/
# Should see metadata_*.json, logs_*.txt, traces_*.json, metrics_*.json
```

If `../datasets` is empty or missing, node 00 will generate synthetic features automatically (pipeline still runs).

---

## Phase 3: Configure and Run the Pipeline

### Step 7: Create or Import the Pipeline

1. In JupyterLab with Elyra: **File** → **New** → **Pipeline**
2. Or: **File** → **Open** → navigate to `elyra_pipeline/` and open `isolation_forest_pipeline.pipeline`

3. Ensure nodes are connected: **Prepare Features** → **Train** → **Test Model** → **Copy to S3** → **Deploy to Serving**

### Step 8: Set Runtime and Environment Variables

For **each node**, open **Properties** (double-click or right-click → Properties):

**Node 00 (Prepare Features):**

| Variable          | Value                                                                 |
|-------------------|-----------------------------------------------------------------------|
| `OTEL_DATASET_DIR`| `../datasets` (or full path to datasets)                             |
| `OUT_FEATURES_CSV`| `/tmp/artifacts/features.csv` (default from pipeline)                |

**Node 03 (Copy to S3):**

| Variable              | Value                                |
|-----------------------|--------------------------------------|
| `AWS_ACCESS_KEY_ID`   | *(from your S3 connection)*          |
| `AWS_SECRET_ACCESS_KEY`| *(from your S3 connection)*         |
| `S3_BUCKET`           | *(your bucket name)*                |
| `S3_PREFIX`           | `models/isolation-forest/`          |

For RHOAI internal S3, you may also need:
- `AWS_ENDPOINT_URL` or `S3_ENDPOINT` (if not default)
- `AWS_REGION` (if required)

**Node 04 (Deploy to Serving):**

| Variable                     | Value                                         |
|-----------------------------|-----------------------------------------------|
| `S3_BUCKET`                 | *(same as node 03)*                           |
| `S3_PREFIX`                 | `models/isolation-forest/`                    |
| `INFERENCE_SERVICE_NAMESPACE`| *(your Data Science Project namespace, e.g. `anomaly-detection`)* |
| `INFERENCE_SERVICE_NAME`    | `isolation-forest` (optional)                 |

### Step 9: Ensure Runtime Image Has Dependencies

The pipeline uses `quay.io/opendatahub/elyra-runtime-image:2024.1`. It should include:
- `pandas`, `scikit-learn`, `numpy`, `joblib`
- `boto3`

If **kubernetes** is missing (needed for node 04), either:
- Use a custom runtime image that adds `pip install kubernetes`, or
- Add a first cell in `04_deploy_to_serving.ipynb`: `!pip install kubernetes -q`

### Step 10: Run the Pipeline

1. **File** → **Save** to save any property changes
2. Click **Run pipeline** (or **Run** button in the pipeline editor)
3. Select the **Kubeflow Pipelines** runtime
4. Submit the run

### Step 11: Monitor the Run

1. RHOAI typically provides a link to the pipeline run (Kubeflow Pipelines UI or RHOAI Pipelines)
2. Watch each node: Prepare → Train → Test → Copy to S3 → Deploy to Serving
3. Check logs for any node that fails

---

## Phase 4: Verify Deployment

### Step 12: Verify the InferenceService

Once the pipeline completes:

```bash
oc get inferenceservice -n <your-project-namespace>
oc get pods -n <your-project-namespace>
```

The `isolation-forest` InferenceService should become Ready.

### Step 13: Test Inference (Optional)

Use the KServe/RHOAI model serving UI or call the predictor endpoint:

```bash
# Get the inference URL (format varies by RHOAI setup)
oc get inferenceservice isolation-forest -n <namespace> -o jsonpath='{.status.url}'
```

Send a prediction request with the expected feature vector format (see the feature columns in `feature_cols.json`).

---

## Troubleshooting

| Issue | Check |
|-------|-------|
| **Node 00 fails** | `OTEL_DATASET_DIR` path; if missing, synthetic data is used |
| **Node 03 fails** | S3 credentials, bucket name, endpoint; verify S3 connection works from a notebook |
| **Node 04 fails** | `INFERENCE_SERVICE_NAMESPACE` set; KServe installed; pipeline pod has RBAC to create InferenceServices |
| **Artifact not found** | Ensure paths match `/tmp/artifacts/` and inputs/outputs are wired in the pipeline |
| **Runtime image missing deps** | Build custom image or add `pip install` in the first cell of the notebook |

---

## Quick Reference: Env Vars by Node

| Node | Required Env Vars |
|------|-------------------|
| 00 | `OTEL_DATASET_DIR`, `OUT_FEATURES_CSV` |
| 01 | (from pipeline) |
| 02 | (from pipeline) |
| 03 | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET`, `S3_PREFIX` |
| 04 | `S3_BUCKET`, `S3_PREFIX`, `INFERENCE_SERVICE_NAMESPACE` |
