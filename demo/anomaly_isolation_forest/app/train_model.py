# train_anomaly_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

# Create output directory if it doesn't exist
os.makedirs("model", exist_ok=True)

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
joblib.dump(model, 'model/isolation_forest_model.joblib')

print("Isolation Forest model trained and saved to model/isolation_forest_model.joblib")
