# inference.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import mlflow
import mlflow.sklearn
import os

# -----------------------------
# Configure MLflow
# -----------------------------
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000"))

# -----------------------------
# Load model from MLflow Registry
# -----------------------------
MODEL_NAME = "anomaly-detection-model"
MODEL_STAGE = "Production"

model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

print(f"Loading model from MLflow Registry: {model_uri}")

model = mlflow.sklearn.load_model(model_uri)

print("Model loaded successfully from MLflow Registry")

# -----------------------------
# Request schema
# -----------------------------
class Metrics(BaseModel):
    cpu_usage: float
    memory_usage: float


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Anomaly Detection API")


# -----------------------------
# Prediction API
# -----------------------------
@app.post("/predict")
def predict_anomaly(metrics: Metrics):

    features = np.array([[metrics.cpu_usage, metrics.memory_usage]])

    prediction = model.predict(features)

    result = "anomaly" if prediction[0] == -1 else "normal"

    anomaly_flag = 1 if prediction[0] == -1 else 0

    # -----------------------------
    # Log inference metrics
    # -----------------------------
    with mlflow.start_run(run_name="inference", nested=True):

        mlflow.log_metric("cpu_usage", metrics.cpu_usage)
        mlflow.log_metric("memory_usage", metrics.memory_usage)
        mlflow.log_metric("anomaly_detected", anomaly_flag)

    return {
        "cpu_usage": metrics.cpu_usage,
        "memory_usage": metrics.memory_usage,
        "result": result
    }