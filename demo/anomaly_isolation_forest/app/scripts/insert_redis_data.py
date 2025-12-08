import redis
import json
import random
from datetime import datetime, timedelta
import os
import numpy as np
# import pandas as pd


# Connect to Redis
# redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
CHANNEL_NAME = os.getenv("CHANNEL_NAME", "observability_channel") 

# Function to generate synthetic anomaly data
# def generate_random_data(num_samples):
#     data = []
#     base_time = datetime.now()

#     for _ in range(num_samples):
#         entry = {
#             "cluster_name": f"cluster_{random.randint(1, 10)}",
#             "pod_name": f"order-srv-{random.randint(1, 50)}",
#             "app_name": f"order-srv-{random.randint(1, 20)}",
#             "cpu_usage": round(random.uniform(0, 100), 2),
#             "memory_usage": round(random.uniform(100, 2048), 2),
#             "timestamp": (base_time - timedelta(minutes=random.randint(0, 1000))).isoformat()
#         }
#         data.append(entry)
    
#     return data


def generate_random_data(num_samples=100, anomaly_ratio=0.05, start_time=None, freq_seconds=10):
    """
    Generate streaming-like test data for anomaly detection.

    Parameters:
    - num_samples: total number of data points
    - anomaly_ratio: fraction of points that are anomalies
    - start_time: datetime to start timestamps
    - freq_seconds: seconds between consecutive entries
    """
    np.random.seed(123)
    random.seed(123)

    if start_time is None:
        start_time = datetime.now()

    num_anomalies = int(num_samples * anomaly_ratio)
    num_normals = num_samples - num_anomalies

    # Normal data
    cpu_normal = np.random.normal(50, 10, num_normals)
    memory_normal = np.random.normal(2048, 512, num_normals)

    # Anomalies
    cpu_anom = np.random.uniform(80, 100, num_anomalies)
    memory_anom = np.random.uniform(4096, 8192, num_anomalies)

    # Combine
    cpu_usage = np.concatenate([cpu_normal, cpu_anom])
    memory_usage = np.concatenate([memory_normal, memory_anom])

    # Shuffle
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    cpu_usage = cpu_usage[indices]
    memory_usage = memory_usage[indices]

    # Generate data entries
    data = []
    for i in range(num_samples):
        timestamp = start_time + timedelta(seconds=freq_seconds * i)
        entry = {
            "cluster_name": f"cluster_{random.randint(1,10)}",
            "pod_name": f"order-srv-{random.randint(1,50)}",
            "app_name": f"order-srv-{random.randint(1,20)}",
            "cpu_usage": round(cpu_usage[i], 2),
            "memory_usage": round(memory_usage[i], 2),
            "timestamp": timestamp.isoformat()
        }
        data.append(entry)
    return data

# Insert data into Redis
def insert_data_to_redis(data, redis_client ):

    for entry in data:
        redis_client.rpush("anomaly_queue", json.dumps(entry))
    print(f"Inserted {len(data)} records into Redis.")

def generate_observability_event_data(num_samples):
    data = []
    base_time = datetime.now()
    
    for _ in range(num_samples):
        entry = {
            # "cluster_name": f"cluster_{random.randint(1, 10)}",
            "servicename": f"order-srv-{random.randint(1, 50)}",
            "cpu_usage": round(random.uniform(0, 100), 2),
            "memory_usage": round(random.uniform(100, 2048), 2),
            "createdtime": (base_time - timedelta(minutes=random.randint(0, 1000))).isoformat()
        }
        data.append(entry)
    
    return data

# Insert data into Redis
def insert_data_to_observability_event_redis(data, redis_client ):

    for entry in data:
        # redis_client.rpush("anomaly_queue", json.dumps(entry))
        print(f"published {entry} records into Redis.{CHANNEL_NAME}")

        redis_client.publish(CHANNEL_NAME, json.dumps(entry, default=str))  
    print(f"Inserted {len(data)} records into Redis.")


if __name__ == "__main__":
    num_samples = 5  # Adjust for more data
    random_data = generate_random_data(num_samples)
    insert_data_to_redis(random_data,redis_client)
