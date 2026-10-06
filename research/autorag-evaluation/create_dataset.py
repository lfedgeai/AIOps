import os
import pandas as pd


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


corpus = [
    {
        "doc_id": "cpu_001",
        "contents": """
CPU utilization exceeded 95%.

Resolution:
1. Check CPU usage using:
   oc adm top pods -n <namespace>

2. Inspect current CPU limits using:
   oc get deployment <deployment-name> -n <namespace> -o yaml

3. Increase CPU limits if required.

4. Configure Horizontal Pod Autoscaler:
   oc autoscale deployment <deployment-name> --cpu-percent=80 --min=2 --max=10 -n <namespace>

5. Verify HPA:
   oc get hpa -n <namespace>

6. Monitor pod scaling activity.
""",
        "metadata": {
            "anomaly_type": "high_cpu_usage",
            "category": "resource"
        }
    },

    {
        "doc_id": "memory_001",
        "contents": """
Memory usage is continuously increasing and may cause OOM kills.

Resolution:
1. Check current memory usage.
2. Review container memory limits.
3. Review application memory allocations.
4. Investigate potential memory leaks.
5. Restart the pod when appropriate.
6. Use heap dumps or application profiling to identify memory leaks.
""",
        "metadata": {
            "anomaly_type": "memory_leak",
            "category": "resource"
        }
    },

    {
        "doc_id": "cpu_memory_001",
        "contents": """
CPU utilization is high and memory usage is continuously increasing.

Resolution:
1. Check CPU and memory usage.
2. Inspect CPU and memory requests and limits.
3. Check whether the application has a resource leak.
4. Increase resources if required.
5. Configure autoscaling when appropriate.
""",
        "metadata": {
            "anomaly_type": "high_cpu_and_memory",
            "category": "resource"
        }
    },

    {
        "doc_id": "network_001",
        "contents": """
Intermittent connectivity issues are observed between services.

Resolution:
1. Verify OpenShift network policies.
2. Check firewall rules.
3. Investigate DNS resolution.
4. Use nslookup to verify DNS.
5. Ensure services are correctly registered.
6. Check service mesh configuration when applicable.
""",
        "metadata": {
            "anomaly_type": "network_latency",
            "category": "network"
        }
    },

    {
        "doc_id": "disk_001",
        "contents": """
Disk pressure occurs when available disk space becomes low.

Resolution:
1. Check disk usage.
2. Remove unnecessary files.
3. Clear old logs when appropriate.
4. Monitor disk usage.
5. Review PersistentVolumeClaim capacity.
6. Optimize I/O-intensive applications.
""",
        "metadata": {
            "anomaly_type": "disk_pressure",
            "category": "storage"
        }
    }
]


qa = [
    {
        "qid": "q001",
        "query": "Why is my OpenShift pod experiencing high CPU usage?",
        "generation_gt": (
            "CPU utilization has exceeded 95%. "
            "Check CPU usage and CPU resource limits and "
            "consider increasing CPU limits or configuring HPA."
        ),
        "references": [
            corpus[0]["contents"]
        ]
    },

    {
        "qid": "q002",
        "query": "What should I check when a pod is being OOM killed?",
        "generation_gt": (
            "Check memory usage, container memory limits, "
            "application memory allocation and possible memory leaks."
        ),
        "references": [
            corpus[1]["contents"]
        ]
    },

    {
        "qid": "q003",
        "query": "How do I troubleshoot high CPU and increasing memory usage?",
        "generation_gt": (
            "Check CPU and memory usage, inspect resource requests "
            "and limits, investigate resource leaks and configure "
            "additional resources or autoscaling when required."
        ),
        "references": [
            corpus[2]["contents"]
        ]
    },

    {
        "qid": "q004",
        "query": "How can I troubleshoot network latency between OpenShift services?",
        "generation_gt": (
            "Check network policies, firewall rules, DNS resolution "
            "and service registration or service mesh configuration."
        ),
        "references": [
            corpus[3]["contents"]
        ]
    },

    {
        "qid": "q005",
        "query": "What should I do when an OpenShift node has disk pressure?",
        "generation_gt": (
            "Check disk usage, clean unnecessary files and logs, "
            "monitor storage and review PersistentVolumeClaim capacity."
        ),
        "references": [
            corpus[4]["contents"]
        ]
    }
]


corpus_df = pd.DataFrame(corpus)
qa_df = pd.DataFrame(qa)

corpus_path = os.path.join(DATA_DIR, "aiops_corpus.parquet")
qa_path = os.path.join(DATA_DIR, "aiops_qa.parquet")

corpus_df.to_parquet(corpus_path, index=False)
qa_df.to_parquet(qa_path, index=False)

print(f"Created: {corpus_path}")
print(f"Created: {qa_path}")

print("\nCorpus:")
print(corpus_df[["doc_id", "metadata"]])

print("\nQA:")
print(qa_df[["qid", "query", "generation_gt"]])