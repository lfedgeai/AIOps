import mlflow
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Note: Tracking URI and S3 Endpoints are picked up from ENV variables
mlflow.set_experiment("Docker_In_House_Experiment")

with mlflow.start_run():
    # Log a parameter
    mlflow.log_param("model_type", "LogisticRegression")
    
    # Create a dummy model
    model = LogisticRegression()
    
    # Log a metric
    mlflow.log_metric("accuracy", 0.85)
    
    # Log the model (this uploads to MinIO)
    mlflow.sklearn.log_model(model, "model")
    
    print("Successfully logged run to MLflow Server from Client Container!")
