# app/model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from app.utils import extract_features


MODEL_PATH = "models/isolation_forest_model.joblib"


def train_anomaly_model():

    # Generate synthetic normal data
    np.random.seed(42)
    cpu_usage = np.random.normal(50, 10, 1000)         # Normal CPU usage pattern
    memory_usage = np.random.normal(2048, 512, 1000)   # Normal memory usage pattern

    # Generate anomaly data
    cpu_anomalies = np.random.uniform(80, 100, 50)
    memory_anomalies = np.random.uniform(4096, 8192, 50)

    # Combine normal and anomalous data
    cpu_usage = np.concatenate([cpu_usage, cpu_anomalies])
    memory_usage = np.concatenate([memory_usage, memory_anomalies])

    # Create a DataFrame
    data = pd.DataFrame({
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage
    })

    # Prepare features
    features = data[['cpu_usage', 'memory_usage']].values

    # Train Isolation Forest
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(features)

    # Save the model
    joblib.dump(model, MODEL_PATH)

    print("Isolation Forest model trained and saved to model/isolation_forest_model.joblib")

        


# Function to train and save model
def train_model():
    df = pd.DataFrame({
        'cpu_usage': np.random.rand(100) * 100,
        'memory_usage': np.random.rand(100) * 1024
    })
    features = extract_features(df)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(features)
    joblib.dump(model, MODEL_PATH)

# Function to load model
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        train_anomaly_model()
        model = joblib.load(MODEL_PATH)
    return model

# Function to make predictions
def predict_anomaly(model, request):
    data = np.array([[request.cpu_usage, request.memory_usage]])
    prediction = model.predict(data)
    result = "Anomaly" if prediction[0] == -1 else "Normal"
    return {
        "cluster_name": request.cluster_name,
        "pod_name": request.pod_name,
        "app_name": request.app_name,
        "timestamp": request.timestamp,
        "cpu_usage": request.cpu_usage,
        "memory_usage": request.memory_usage,
        "result": result
    }
def predict_bulk_anomalies(model, requests):
    
    data = [
        # [req.cpu_usage, req.memory_usage] for req in requests
        [req["cpu_usage"], req["memory_usage"]] for req in requests

    ]
    predictions = model.predict(data)
    results = []
    for req, pred in zip(requests, predictions):
        # result = "Anomaly" if pred == -1 else "Normal"
        if pred == -1:  # Only add anomalies
            
            anomaly_type = classify_anomaly_type(req["cpu_usage"], req["memory_usage"])

            results.append({
                "cluster_name": req["cluster_name"],
                "pod_name": req["pod_name"],
                "app_name": req["app_name"],
                "timestamp": req["timestamp"],
                "cpu_usage": req["cpu_usage"],
                "memory_usage": req["memory_usage"],
                "is_anomaly": "Anomaly",
                "anomaly_type": anomaly_type
                
            })
    return results
def classify_anomaly_type(cpu, mem, cpu_thresh=75, mem_thresh=3500):
    high_cpu = cpu > cpu_thresh
    high_mem = mem > mem_thresh

    if high_cpu and high_mem:
        return "high_cpu_and_memory"
    elif high_cpu:
        return "high_cpu_usage"
    elif high_mem:
        return "memory_leak"
    else:
        return "high_cpu_usage"
