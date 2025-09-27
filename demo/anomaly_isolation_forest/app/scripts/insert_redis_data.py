import redis
import json
import random
from datetime import datetime, timedelta
import os
# Connect to Redis
# redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
CHANNEL_NAME = os.getenv("CHANNEL_NAME", "observability_channel") 

# Function to generate synthetic anomaly data
def generate_random_data(num_samples):
    data = []
    base_time = datetime.now()

    for _ in range(num_samples):
        entry = {
            "cluster_name": f"cluster_{random.randint(1, 10)}",
            "pod_name": f"order-srv-{random.randint(1, 50)}",
            "app_name": f"order-srv-{random.randint(1, 20)}",
            "cpu_usage": round(random.uniform(0, 100), 2),
            "memory_usage": round(random.uniform(100, 2048), 2),
            "timestamp": (base_time - timedelta(minutes=random.randint(0, 1000))).isoformat()
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
            "cpuusage": round(random.uniform(0, 100), 2),
            "memoryusage": round(random.uniform(100, 2048), 2),
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
