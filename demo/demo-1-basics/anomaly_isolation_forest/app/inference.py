# inference.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import os
import tempfile
from datetime import datetime
from mlflow.tracking import MlflowClient

# --------------------------------------------------
# Configure MLflow Tracking
# --------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow_server:5000"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --------------------------------------------------
# Set MLflow Experiment
# --------------------------------------------------
EXPERIMENT_NAME = "anomaly-detection-inference"

mlflow.set_experiment(EXPERIMENT_NAME)

# --------------------------------------------------
# Model Configuration
# --------------------------------------------------
MODEL_NAME = "anomaly-detection-model"

# Use alias instead of hardcoded version/stage
MODEL_ALIAS = "champion"

# Dynamic model URI
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

print(f"Loading model from MLflow Registry: {model_uri}")

# --------------------------------------------------
# Load Model Dynamically
# --------------------------------------------------
try:
    model = mlflow.sklearn.load_model(model_uri)

    print("Model loaded successfully from MLflow Registry")

except Exception as e:

    print(f"Error loading model: {e}")

    raise e

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
app = FastAPI(
    title="Anomaly Detection API",
    version="1.0"
)

# --------------------------------------------------
# Request Schema
# --------------------------------------------------
class Metrics(BaseModel):

    cpu_usage: float
    memory_usage: float

# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------
@app.post("/predict")
def predict_anomaly(metrics: Metrics):

    # ----------------------------------------------
    # Prepare Input Features
    # ----------------------------------------------
    features = np.array([
        [
            metrics.cpu_usage,
            metrics.memory_usage
        ]
    ])

    # ----------------------------------------------
    # Predict
    # ----------------------------------------------
    prediction = model.predict(features)

    result = "anomaly" if prediction[0] == -1 else "normal"

    anomaly_flag = 1 if prediction[0] == -1 else 0

    # ----------------------------------------------
    # Get Active Model Version from Alias
    # ----------------------------------------------
    client = MlflowClient()

    model_version = None

    try:

        model_version = client.get_model_version_by_alias(
            MODEL_NAME,
            MODEL_ALIAS
        ).version

    except Exception:
        model_version = "unknown"

    # ----------------------------------------------
    # MLflow Logging
    # ----------------------------------------------
    with mlflow.start_run(run_name="inference"):

        # ------------------------------------------
        # Log Parameters
        # ------------------------------------------
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("model_alias", MODEL_ALIAS)
        mlflow.log_param("model_version", model_version)

        # ------------------------------------------
        # Log Metrics
        # ------------------------------------------
        mlflow.log_metric("cpu_usage", metrics.cpu_usage)
        mlflow.log_metric("memory_usage", metrics.memory_usage)
        mlflow.log_metric("anomaly_detected", anomaly_flag)

        # ------------------------------------------
        # Create Inference DataFrame
        # ------------------------------------------
        inference_df = pd.DataFrame([
            {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "prediction": int(prediction[0]),
                "result": result,
                "model_version": model_version
            }
        ])

        # ------------------------------------------
        # Save Prediction CSV Artifact
        # ------------------------------------------
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False
        ) as tmp_file:

            inference_df.to_csv(
                tmp_file.name,
                index=False
            )

            # --------------------------------------
            # Log Artifact to MLflow
            # --------------------------------------
            mlflow.log_artifact(
                tmp_file.name,
                artifact_path="inference_logs"
            )

    # ----------------------------------------------
    # API Response
    # ----------------------------------------------
    return {
        "cpu_usage": metrics.cpu_usage,
        "memory_usage": metrics.memory_usage,
        "prediction": int(prediction[0]),
        "result": result,
        "model_version": model_version
    }