# Provider image = harness image + EvalHub entrypoint.
# Build the harness image first (repo-root Dockerfile), push it, then:
#   podman build -f evalhub/lfedge_aiops_provider/Dockerfile \
#     --build-arg HARNESS_IMAGE=<harness-image> -t <provider-image> .
ARG HARNESS_IMAGE
FROM ${HARNESS_IMAGE}

# The harness base image bundles a snapshot of this provider; overwrite it with
# the current source so provider fixes ship without rebuilding the base image.
COPY main.py /app/evalhub/lfedge_aiops_provider/main.py

USER 1001
ENTRYPOINT ["python3", "-u", "/app/evalhub/lfedge_aiops_provider/main.py"]
