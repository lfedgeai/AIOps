from fastapi import FastAPI, HTTPException
from app.models.models import AnomalyData
from app.services.rag_pipeline import analyze_anomaly_with_llm,get_vector_store, vector_store
from datetime import datetime
import os
from typing import Dict

import asyncio
import json
import psutil, os
from app.services.consumer import consume_anomalies
import logging


app = FastAPI()


logging.basicConfig(level=logging.DEBUG)  # Change INFO to DEBUG

logger = logging.getLogger("main")

logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    print("started vector store")
    logger.info("Started vector store")
    get_vector_store()
    process = psutil.Process(os.getpid())
    print(f"Memory used after startup: {process.memory_info().rss / 1024 ** 2:.2f} MB")
    logger.info(f"Memory used after startup: {process.memory_info().rss / 1024 ** 2:.2f} MB")
    print("completed vector store")
    
    print("Starting  consumer...")
    logger.info("Starting  consumer")
    asyncio.create_task(consume_anomalies())
    logger.info("Starting  consumer")

    print("complete  consumer")



@app.get("/health")
def health_check():
    return {"status": "ok", "vector_store_initialized": vector_store is not None}

@app.get("/")
async def root():
    return {"message": "LLM Service is running"}
@app.post("/process-anomaly")
async def process_anomaly(anomaly_data: dict ):
    try:
        app_name = anomaly_data.get("app_name", "unknown_app")
        pod_name = anomaly_data.get("pod_name", "unknown_pod")
        cluster_info = anomaly_data.get("cluster_info", "unknown_cluster")
        
        response = analyze_anomaly_with_llm(anomaly_data)
        action_req = os.getenv("ACTION_REQ")
        print("scale_pod--begin")
        
        logger.info(f"Creating case for Anomaly: {anomaly_data} ")
        
        logger.info(f"LLM content Begin---------------***********__________________________________")

        logger.info(f"Case content from LLM: {response} ")
        
        logger.info(f"LLM content End---------------***********__________________________________")
            
        return {"resolution": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
# @app.post("/anomaly-detected")
# async def handle_anomaly(anomaly: AnomalyData):
#     anomaly_dict = anomaly.model_dump()
    
#     key = save_anomaly(anomaly_dict)
    
#     return anomaly_dict
   
# @app.post("/anomaly/{timestamp}")
# def fetch_anomaly(timestamp: str):
#     key = f"anomaly:{timestamp}"
#     data = get_anomaly(key)
#     if data:
#         return data
#     raise HTTPException(status_code=404, detail="Anomaly not found")


