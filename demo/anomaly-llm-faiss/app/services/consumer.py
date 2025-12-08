# service/consumer.py
import asyncio
import json
import os
import redis
from app.services.rag_pipeline import analyze_anomaly_with_llm

import logging


logging.basicConfig(level=logging.DEBUG)  # Change INFO to DEBUG

logger = logging.getLogger("consumer")

logger = logging.getLogger(__name__)


# Redis client setup
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0))
)

# Use environment variable for queue name
QUEUE_NAME = os.getenv("LLM_INFERENCE_QUEUE", "llm_inference_queue")
PROCESSED_QUEUE = os.getenv("PROCESSED_QUEUE", "anomaly_with_response_queue")


async def consume_anomalies():
    print("Redis consumer started.")
    logger.info("Redis consumer started.")
    while True:
        try:
            item = redis_client.blpop(QUEUE_NAME, timeout=5)
            if item:
                _, raw_data = item
                anomaly_data = json.loads(raw_data)

                app_name = anomaly_data.get("app_name", "unknown_app")
                pod_name = anomaly_data.get("pod_name", "unknown_pod")
                cluster_info = anomaly_data.get("cluster_info", "unknown_cluster")

                response = analyze_anomaly_with_llm(anomaly_data)
                
                anomaly_data["llm_response"] = response
                
                # Push processed anomaly into another Redis queue
                redis_client.rpush(PROCESSED_QUEUE, json.dumps(anomaly_data))


                # create_case(response, anomaly_data)

                # if os.getenv("ACTION_REQ") == "Y":
                #     scale_pod(response, app_name, 2)

                # print(f"Processed anomaly for {app_name}: {response}")
                # logger.info(f"Processed anomaly for {app_name}: {response}")
                logger.info(f"Processed anomaly for {anomaly_data.get('app_name', 'unknown')}")

        except Exception as e:
            print(f"Error processing anomaly: {e}")
            logger.debug(f"Error processing anomaly: {e}")
        await asyncio.sleep(0.1)

def processed_anomalies():
    anomalies = []
    total_count = redis_client.llen(PROCESSED_QUEUE)

    items = redis_client.lrange(PROCESSED_QUEUE, 0, -1)
    for raw_value in items:
        try:
            anomalies.append(json.loads(raw_value))
        except json.JSONDecodeError:
            logger.error("Invalid JSON in Redis: %s", raw_value)
            anomalies.append({"error": "Invalid JSON", "raw": raw_value})

    return {
        "total_count": total_count,
        "anomalies": anomalies
    }
