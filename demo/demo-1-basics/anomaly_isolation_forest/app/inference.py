# anomaly_api.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

# Load the model
model_path = "model/isolation_forest_model.joblib"
if not os.path.exists(model_path):
    raise FileNotFoundError("Model not found.")
model = joblib.load(model_path)

# Define request schema
class Metrics(BaseModel):
    cpu_usage: float
    memory_usage: float

# Initialize FastAPI app
app = FastAPI(title="Anomaly Detection API")

@app.post("/predict")
def predict_anomaly(metrics: Metrics):
    features = np.array([[metrics.cpu_usage, metrics.memory_usage]])
    prediction = model.predict(features)  # -1 = anomaly, 1 = normal
    result = "anomaly" if prediction[0] == -1 else "normal"
    return {
        "cpu_usage": metrics.cpu_usage,
        "memory_usage": metrics.memory_usage,
        "result": result
    }
