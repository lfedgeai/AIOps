# FastAPI Routes for Anomaly Detection
from fastapi import FastAPI,APIRouter, HTTPException
from app.models.models import AnomalyData

# router = APIRouter()
from typing import Dict

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Isolation Forest Anomaly Detection API is running."}

@app.post("/detect-anomaly/")
async def handle_anomaly(anomaly: AnomalyData):
    try:
        app_name = anomaly.get("app_name", "unknown_app")
        pod_name = anomaly.get("pod_name", "unknown_pod")
        cluster_info = anomaly.get("cluster_info", "unknown_cluster")
        # response = analyze_anomaly_wi
        # th_llm(anomaly)
        return {"resolution": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

