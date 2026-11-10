# Standalone AIOps evaluation harness image.
# Build: podman build -t quay.io/<you>/aiops-harness:latest .
# The image runs as an in-cluster Job/Deployment; it uses the pod service
# account (no oc/kubeconfig required) and talks to ClickHouse/MLflow/model
# endpoints over cluster DNS.
FROM registry.access.redhat.com/ubi9/python-311:latest

USER root
WORKDIR /app

# The OTEL Demo source used by RAG is vendored under context/, so git is not
# required at runtime. If you delete that tree, install git in the image or
# pre-populate CONTEXT_REPO_PATH.

COPY requirements.txt /app/requirements.txt
COPY code/harness/requirements.txt /app/code/harness/requirements.txt
RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -r /app/requirements.txt -r /app/code/harness/requirements.txt

COPY . /app
RUN chmod -R a+rX /app && \
    mkdir -p /app/logs /app/out /app/data /app/mlartifacts && \
    chmod -R a+rwX /app/logs /app/out /app/data /app/mlartifacts && \
    chmod +x /app/scripts/*.sh

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    HARNESS_PYTHON=/opt/app-root/bin/python3 \
    CONTEXT_REPO_PATH=/app/context/opentelemetry-demo

USER 1001
ENTRYPOINT ["/app/scripts/run_harness.sh"]
