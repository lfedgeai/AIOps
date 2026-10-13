
import json
import asyncio
from datetime import datetime

def transform_to_anomaly_format(observability_event):
    return {
        # "cluster_name": f"cluster_{random.randint(1, 10)}",
        "cluster_name": observability_event.get("clusterName") or f"cluster_{random.randint(1, 10)}",
        "pod_name": observability_event["podName"],
        "app_name": observability_event["servicename"],  # or derive differently if needed
        "cpu_usage": round(float(observability_event["cpuusage"]), 2),
        "memory_usage": round(float(observability_event["memoryusage"]), 2),
        "timestamp": observability_event["createdtime"] if isinstance(source["createdtime"], str) else observability_event["createdtime"].isoformat()
    }


async def consume_observability_event_redis(obser_redis_client , redis_client):
    print("Redis consumer started")
    while True:
        try:
            item = await redis_client.blpop("anomaly_queue")
            if item:
                _, raw_data = item
                try:
                    data = json.loads(raw_data)
                    results = await predict_bulk_anomalies(model, [data])
                    anomaly = results[0]

                    if anomaly["is_anomaly"] == "Anomaly":
                        await redis_client.rpush("anomaly_results", json.dumps(anomaly))
                        print(f"Anomaly detected and pushed: {anomaly}")
                    else:
                        print(f"Normal data: {anomaly}")
                except Exception as e:
                    print(f"Error processing item: {e}")
        except Exception as e:
            print(f"Redis connection error: {e}")
            await asyncio.sleep(5)  # Retry delay