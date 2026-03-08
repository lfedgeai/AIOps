#!/usr/bin/env python3
"""Patch otel-collector ConfigMap to add ClickHouse exporter."""
import subprocess
import json
import yaml

# Get current configmap
result = subprocess.run(
    ["oc", "get", "configmap", "otel-collector", "-n", "otel-demo", "-o", "json"],
    capture_output=True,
    text=True,
    check=True,
)
cm = json.loads(result.stdout)
relay_yaml = cm["data"]["relay"]
config = yaml.safe_load(relay_yaml)

# Add clickhouse exporter
config["exporters"]["clickhouse"] = {
    "endpoint": "tcp://clickhouse:9000?dial_timeout=10s",
    "database": "otel",
    "create_schema": True,
    "timeout": "10s",
}

# Add clickhouse to pipelines
for pipeline in ["traces", "metrics", "logs"]:
    if pipeline in config["service"]["pipelines"]:
        exporters = config["service"]["pipelines"][pipeline]["exporters"]
        if "clickhouse" not in exporters:
            config["service"]["pipelines"][pipeline]["exporters"] = exporters + ["clickhouse"]

new_relay = yaml.dump(config, default_flow_style=False, sort_keys=False)
cm["data"]["relay"] = new_relay
# Remove managed fields for apply
if "metadata" in cm and "managedFields" in cm["metadata"]:
    del cm["metadata"]["managedFields"]
if "metadata" in cm and "resourceVersion" in cm["metadata"]:
    del cm["metadata"]["resourceVersion"]

with open("/tmp/otel-collector-patched.yaml", "w") as f:
    json.dump(cm, f, indent=2)

subprocess.run(["oc", "apply", "-f", "/tmp/otel-collector-patched.yaml"], check=True)
print("Patched otel-collector ConfigMap with ClickHouse exporter")
