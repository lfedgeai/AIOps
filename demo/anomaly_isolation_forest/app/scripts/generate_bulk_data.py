import random
import json
from datetime import datetime, timedelta

def generate_random_data(num_samples):
    data = []
    base_time = datetime.now()

    for _ in range(num_samples):
        entry = {
            "cluster_name": f"cluster_{random.randint(1, 10)}",
            "pod_name": f"pod_{random.randint(1, 50)}",
            "app_name": f"app_{random.randint(1, 20)}",
            "cpu_usage": round(random.uniform(0, 100), 2),
            "memory_usage": round(random.uniform(100, 2048), 2),
            "timestamp": (base_time - timedelta(minutes=random.randint(0, 1000))).isoformat()
        }
        data.append(entry)
    
    return data

def save_to_file(filename, data):
    with open(filename, "w") as f:
        json.dump({"data": data}, f, indent=4)

if __name__ == "__main__":
    num_samples = 1000  # Adjust the number for larger datasets
    bulk_data = generate_random_data(num_samples)
    save_to_file("bulk_test_data.json", bulk_data)
    print(f"Generated {num_samples} records in 'bulk_test_data.json'")
