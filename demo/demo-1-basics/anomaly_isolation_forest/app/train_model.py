import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os
import time
import mlflow
import mlflow.sklearn


# -----------------------------
# Initialize MLflow connection
# -----------------------------
def init_mlflow():

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")

    for attempt in range(10):
        try:
            print(f"Connecting to MLflow at {mlflow_uri} (attempt {attempt+1})")

            mlflow.set_tracking_uri(mlflow_uri)

            # Check connection
            mlflow.search_experiments()

            mlflow.set_experiment("anomaly-detection")

            print("Connected to MLflow successfully")
            return

        except Exception as e:
            print("Waiting for MLflow...", e)
            time.sleep(3)

    raise RuntimeError("Could not connect to MLflow")


# -----------------------------
# Initialize MLflow
# -----------------------------
init_mlflow()


# -----------------------------
# Create output directory
# -----------------------------
os.makedirs("model", exist_ok=True)


# -----------------------------
# Generate synthetic data
# -----------------------------
np.random.seed(42)

cpu_usage = np.random.normal(50, 10, 1000)
memory_usage = np.random.normal(2048, 512, 1000)

cpu_anomalies = np.random.uniform(80, 100, 50)
memory_anomalies = np.random.uniform(4096, 8192, 50)

cpu_usage = np.concatenate([cpu_usage, cpu_anomalies])
memory_usage = np.concatenate([memory_usage, memory_anomalies])

data = pd.DataFrame({
    "cpu_usage": cpu_usage,
    "memory_usage": memory_usage
})

features = data[["cpu_usage", "memory_usage"]].values


# -----------------------------
# Train model with MLflow
# -----------------------------
with mlflow.start_run(run_name="isolation-forest-training"):

    # -----------------------------
    # Log parameters
    # -----------------------------
    mlflow.log_param("model", "IsolationForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("contamination", 0.05)
    mlflow.log_param("dataset_size", len(data))

    # -----------------------------
    # Train model
    # -----------------------------
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )

    model.fit(features)

    # -----------------------------
    # Calculate training metrics
    # -----------------------------
    predictions = model.predict(features)

    anomaly_count = int(np.sum(predictions == -1))
    normal_count = int(np.sum(predictions == 1))
    total_samples = len(predictions)

    anomaly_ratio = anomaly_count / total_samples

    # -----------------------------
    # Log metrics to MLflow
    # -----------------------------
    mlflow.log_metric("anomaly_count", anomaly_count)
    mlflow.log_metric("normal_count", normal_count)
    mlflow.log_metric("anomaly_ratio", anomaly_ratio)
    mlflow.log_metric("training_samples", total_samples)

    # -----------------------------
    # Save model locally
    # -----------------------------
    model_path = "model/isolation_forest_model.joblib"
    joblib.dump(model, model_path)

    # Log artifact
    mlflow.log_artifact(model_path)

    # -----------------------------
    # Log & Register model
    # -----------------------------
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="isolation_forest_model",
        registered_model_name="anomaly-detection-model"
    )

    print("Model trained, logged, and registered in MLflow")

print("Model saved locally at:", model_path)