import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

# Generate synthetic training data
np.random.seed(42)
cpu_usage = np.random.normal(50, 10, 1000)  # Normal CPU usage pattern
memory_usage = np.random.normal(2048, 512, 1000)  # Normal memory usage pattern

# Introduce anomalies
cpu_anomalies = np.random.uniform(80, 100, 50)
memory_anomalies = np.random.uniform(4096, 8192, 50)

# Combine data
cpu_usage = np.concatenate([cpu_usage, cpu_anomalies])
memory_usage = np.concatenate([memory_usage, memory_anomalies])

# Create DataFrame
data = pd.DataFrame({
    'cpu_usage': cpu_usage,
    'memory_usage': memory_usage
})

# Prepare features
features = data[['cpu_usage', 'memory_usage']].values

# Train Isolation Forest model
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(features)

# Save the trained model
joblib.dump(model, './model/isolation_forest_model.joblib')

print("Isolation Forest model trained and saved successfully.")
