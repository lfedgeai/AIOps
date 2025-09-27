from fastapi import FastAPI,BackgroundTasks , Depends,Query ,HTTPException# type: ignore
from app.model import load_model, predict_anomaly, predict_bulk_anomalies,train_anomaly_model
from app.scripts.insert_redis_data import generate_random_data , insert_data_to_redis , insert_data_to_observability_event_redis ,generate_observability_event_data
from redis import Redis # type: ignore
import redis.asyncio as aioredis
import random
import joblib
from typing import Optional

import json
import logging
from logging.handlers import TimedRotatingFileHandler
import os
from datetime import date, datetime, time, timedelta
from contextlib import asynccontextmanager
import json
import logging
# import httpx
from httpx import AsyncClient,Timeout, ReadTimeout
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.db import models, crud
from app.db.schemas import AnomalyRequest, BulkAnomalyRequest

import asyncio

from sqlalchemy.orm import Session
from app.db.db import SessionLocal, engine
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.DEBUG)  # Change INFO to DEBUG

logger = logging.getLogger("apscheduler")


models.Base.metadata.create_all(bind=engine)


# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


PROCESS_URL = os.environ["PROCESS_ANOMALY_URL"]
MODEL_PATH = "models/isolation_forest_model.joblib"

timeout = Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)


# Load the trained Isolation Forest model
model = load_model()
redis_host = os.getenv("REDIS_HOST", "redis")  # Use 'redis' as the default hostname
redis_port = int(os.getenv("REDIS_PORT", 6379))

redis_obs_host = os.getenv("REDIS_OBS_HOST", "redis")  # Use 'redis' as the default hostname
redis_obs_port = int(os.getenv("REDIS_OBS_PORT", 6379))

CHANNEL_NAME = os.getenv("CHANNEL_NAME", "observability_channel") 
REDIS_URL = f"redis://{redis_obs_host}:{redis_obs_port}"
redis_obs_client = None

subscriber_task = None


# Connect to Redis
redis_client = Redis(host=redis_host, port=redis_port, decode_responses=True)


scheduler = AsyncIOScheduler()


@app.post("/persistAnomalies/")
def insert_anomaly(anomaly: AnomalyRequest, db: Session = Depends(get_db)):
    return crud.create_anomaly(db, anomaly)

@app.get("/allAnomalies/")
def list_anomalies(db: Session = Depends(get_db)):
    return crud.get_all_anomalies(db)

@app.get("/anomalies/")
def list_anomalies(db: Session = Depends(get_db)):
    return crud.get_all_anomalies_sorted(db)


# @app.get("/anomalies/date-range")
# def list_anomalies_between_dates(
#     start_date: datetime = Query(..., description="Start datetime (ISO format)"),
#     end_date: datetime = Query(..., description="End datetime (ISO format)"),
#     skip: int = 0,
#     limit: int = 100,
#     db: Session = Depends(get_db)
# ):
#     return crud.get_anomalies_between_dates(db, start_date, end_date, skip, limit)

@app.get("/anomalies/date-range")
def list_anomalies_between_dates(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    minutes: Optional[int] = Query(None, description="Window in minutes before now"),
    cluster_name: Optional[str] = Query(None),
    pod_name: Optional[str] = Query(None),
    app_name: Optional[str] = Query(None),
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    now = datetime.utcnow()

    if start_date is not None and end_date is not None:
        # Use full days for start and end date
        start_dt = datetime.combine(start_date, time.min)
        end_dt = datetime.combine(end_date, time.max)
    elif minutes is not None:
        # Ignore start_date and end_date, use now and now - minutes
        start_dt = now - timedelta(minutes=minutes)
        end_dt = now
    else:
        # Default behavior - no filters or some default range
        start_dt = None
        end_dt = None

    print(f"Querying anomalies from {start_dt} to {end_dt}")

    return crud.get_anomalies_filtered(
        db=db,
        start_date=start_dt,
        end_date=end_dt,
        cluster_name=cluster_name,
        pod_name=pod_name,
        app_name=app_name,
        skip=skip,
        limit=limit
    )


@app.post("/train_model")
def start_training(background_tasks: BackgroundTasks):
    global training_in_progress

    if training_in_progress:
        return {"status": "training already in progress"}

    training_in_progress = True

    def run_training():
        global model, training_in_progress
        try:
            train_anomaly_model()
            model = joblib.load(MODEL_PATH)
        finally:
            training_in_progress = False

    background_tasks.add_task(run_training)
    return {"status": "training started"}

@app.get("/training_status")
def get_training_status():
    return {"training": training_in_progress}

@app.get("/anomalies/filter")
def filter_anomalies(
    start_date: datetime = Query(..., description="Start of date range"),
    end_date: datetime = Query(..., description="End of date range"),
    cluster_name: Optional[str] = None,
    pod_name: Optional[str] = None,
    app_name: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    return crud.get_anomalies_filtered(
        db=db,
        start_date=start_date,
        end_date=end_date,
        cluster_name=cluster_name,
        pod_name=pod_name,
        app_name=app_name,
        skip=skip,
        limit=limit
    )


# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Isolation Forest Anomaly Detection API is running."}

# Prediction endpoint
@app.post("/predict")
async def predict(request: AnomalyRequest):
    result = predict_anomaly(model, request)
    logger.info(f"Single anomaly prediction processed for pod: {request.pod_name}")
    return result

# Bulk prediction endpoint
@app.post("/predict/bulk")
async def predict_bulk(requests: BulkAnomalyRequest):
    results = predict_bulk_anomalies(model, requests.data)
    logger.info(f"Bulk anomaly prediction processed for {len(results)} entries.")
    return {"results": results}

def process_scheduled_anomaly(anomaly_data):
    db = SessionLocal()
    try:
        crud.create_anomaly(db, anomaly_data)
    finally:
        db.close()
def create_anomaly_from_dict(anomaly_dict: dict):
    db = SessionLocal()
    try:
        anomaly = AnomalyRequest(**anomaly_dict)
        return crud.create_anomaly(db, anomaly)
    finally:
        db.close()


async def process_redis_data() -> None:
    anomaly_data = []
    # print(f"Periodic Task Triggered: {datetime.datetime.now()}")  # Add this line

    while True:
        item = redis_client.lpop("anomaly_queue")  # Pull data from Redis list
        if item is None:
            break  # No more data to process
        data = json.loads(item)
        anomaly_data.append(data)

    if anomaly_data:
        logger.info(f"Processing data {anomaly_data} entries from Redis.")

        results = predict_bulk_anomalies(model, anomaly_data)
        logger.info(f"Processed {len(results)} entries from Redis.")
        logger.info(f"Processed {results} entries from Redis.")
        
        # Persist to db
        for anomaly in anomaly_data:
            # anomaly_dt= AnomalyRequest(**anomaly)
            # insert_anomaly(anomaly_dt)
            # process_scheduled_anomaly(anomaly)
            create_anomaly_from_dict(anomaly)

        

        default_values = {
            "anomaly_type": "Unknown",
            "description": "No description available",
            "resolution": "Pending"
        }

        
        for res, entry in zip(results, anomaly_data):
            logger.info(f"Anomaly detected: {res['is_anomaly']} for pod {entry['pod_name']} at {entry['timestamp']}")
            
            # Ensure `timestamp` exists and is a string
            res["timestamp"] = entry.get("timestamp", datetime.utcnow().isoformat())  # Convert to ISO string
            # res["anomaly_type"] = "high_cpu_usage" if res["is_anomaly"] == "Anomaly" else "Normal"
            logger.info(f"Anomaly detected-->: {res['anomaly_type'] }")

            # Merge default values
            for key, value in default_values.items():
                res.setdefault(key, value)
        


        # if results:
        #     logger.info("entering into rpocess")
        #     tasks = [call_anomaly_api(data) for data in results]

        #     responses = await asyncio.gather(*tasks, return_exceptions=True)
            
        #     for inp, out in zip(results, responses):
        #         if isinstance(out, Exception):
        #             logger.error("Error for %s: %s", inp, out)
        #         else:
        #             logger.info("Success for %s: %s", inp, out)
        if results:
            logger.info("Publishing anomaly data to Redis queue")
            for data in results:
                try:
                    redis_client.rpush("llm_inference_queue", json.dumps(data))
                    redis_client.publish("anomaly_catalog_topic", json.dumps(data))
                    logger.info("Pushed to Redis queue: %s", data)
                except Exception as e:
                    logger.error("Redis push failed for %s: %s", data, e)



    else:
        logger.info("No new data to process from Redis.")

    # Set up background scheduler
from datetime import datetime

def convert_datetime_to_str(data):
    """Convert all datetime fields in a dictionary to string format"""
    for key, value in data.items():
        if isinstance(value, datetime):
            data[key] = value.isoformat()  # Converts datetime to ISO 8601 string
    return data

@app.get("/generate-anomaly-data/{num_samples}")
def generate_data(num_samples: int):
    """API to generate random data and log it"""
    data = generate_random_data(num_samples)
    insert_data_to_redis(data , redis_client)
    return data

@app.get("/generate-observability-data/{num_samples}")
def generate_data(num_samples: int):
    """API to generate random observability metrics data and log it"""
    data = generate_observability_event_data(num_samples)
    insert_data_to_observability_event_redis(data , redis_client)
    return data


async def call_anomaly_api(anomaly_data):
    logger.info("call call_anomaly_api")
    logger.info(f"call_anomaly_api: {PROCESS_URL}")

    try:
        async with AsyncClient(timeout=Timeout(30.0))  as client:
            response = await client.post(PROCESS_URL, json=anomaly_data)
            response.raise_for_status()

            if response.status_code == 200:
                logger.info(f"Anomaly processed: {response.json()}")

                print(f"Anomaly processed: {response.json()}")
            else:
                print(f"Error processing anomaly: {response.status_code}, {response.text}")
    except ReadTimeout:
        logger.error(f"ReadTimeout when calling {PROCESS_URL} for data {anomaly_data}")
    except Exception as e:
        logger.error(f"Failed to call anomaly API at {PROCESS_URL}: {e}")
    # except Exception as e:
    #     print(f"Failed to call anomaly API: {e}")

scheduler = AsyncIOScheduler()

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()
    task = app.state.subscriber_task
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            print("redis_subscriber task cancelled cleanly.")
    
@app.on_event("startup")
async def start_scheduler():
    
    # asyncio.create_task(redis_subscriber())
    app.state.subscriber_task = asyncio.create_task(redis_subscriber())

    print("Redis subscriber started")

    # Start your async job scheduler
    scheduler.add_job(process_redis_data, "interval", seconds=5)
    scheduler.start()
    print("Scheduler started:", scheduler.running)

# Test endpoint for validation
@app.get("/test")
async def test():
    return {"message": "Test endpoint is working."}

def convert_to_anomaly_model_format(observability_event: dict) -> dict:
    def safe_get(key, default=None):
        return observability_event.get(key, default)

    def safe_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0
    # cpu_limit in cores
    def convert_cpu_to_percent(val, cpu_limit=1.0):  
        try:
            return round((float(val) / cpu_limit) * 100, 2)
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0

    def convert_bytes_to_mb(val):
        try:
            return round(float(val) / (1024 ** 2), 2)
        except (TypeError, ValueError):
            return 0.0

    def safe_timestamp(val):
        if isinstance(val, str):
            return val
        elif isinstance(val, datetime):
            return val.isoformat()
        return datetime.utcnow().isoformat()

    return {
        "cluster_name": safe_get("clusterName", "unknown-cluster"),
        "pod_name": safe_get("podName", "unknown-pod"),
        "app_name": safe_get("serviceName", "unknown-app"),
        "cpu_usage": convert_cpu_to_percent(safe_get("cpuUsage"), cpu_limit=1.0),
        "memory_usage": convert_bytes_to_mb(safe_get("memoryUsage")),
        "timestamp": safe_timestamp(safe_get("createdtime"))
    }


async def redis_subscriber():
    global redis_obs_client
    redis_obs_client = await aioredis.from_url(REDIS_URL)

    try:
        pubsub = redis_obs_client.pubsub()
        
        print(f" Recieved records from  Redis.{CHANNEL_NAME}")


        await pubsub.subscribe(CHANNEL_NAME)
        
        async for message in pubsub.listen():
            print(f" Received message Redis------{message}")

            if message["type"] == "message":
                
                try:
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")

                    event = json.loads(data)
                    
                    event = json.loads(data)
                    print(f"Received message: {event}")

                    # Handle both single dict and list of dicts
                    events = event if isinstance(event, list) else [event]
                    print(f"Received message dictionary: {events}")
                    # following code commented out to incorporate APM events
                    # formatted =   convert_to_anomaly_model_format(event)
                    # anomaly = json.dumps(formatted)
                
                    # Optional: push to another Redis queue
                    # redis_client.rpush("anomaly_queue", anomaly)
                    # print(f"Inserted {anomaly} records into Redis.")
                    
                    for single_event in events:
                        formatted = convert_to_anomaly_model_format(single_event)
                        
                        print(f" anomaly record from APM : {formatted}")
                        
                        # if data["cpu_usage"] > 0 and data["memory_usage"] > 0:
                        if formatted["cpu_usage"] > 0 and formatted["memory_usage"] > 0:

                            anomaly = json.dumps(formatted)
                            redis_client.rpush("anomaly_queue", anomaly)
                            print(f"Inserted anomaly record into Redis: {anomaly}")
                        else:
                            print(f" Anomaly detected , but its not an anomaly: {formatted}")
                       

                except json.JSONDecodeError:
                    # logger.error("Failed to decode JSON from message: %s", message["data"])
                    print("Failed to decode JSON from message: %s", message["data"])
                except Exception as e:
                    print("Unexpected error processing message: %s", e)


    except asyncio.CancelledError:
        # Optional: handle cleanup
        await pubsub.unsubscribe("anomaly-channel")
        raise
