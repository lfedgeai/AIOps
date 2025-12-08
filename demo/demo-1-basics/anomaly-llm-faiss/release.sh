#!/usr/bin/env bash
set -euo pipefail

# Log in (make sure to secure your credentials in CI secrets)

# Define your image base and version
IMAGE=quay.io/zagaos/anomaly-llm-gpt2
VERSION=1.19

# Default: versioned tag; also push latest
TAG_VERSIONED=${IMAGE}:${VERSION}
TAG_LATEST=${IMAGE}:latest

# Ensure buildx builder exists
docker buildx ls | grep multiarch || docker buildx create --name multiarch --use

# Build & push both tags for arm64 and amd64
docker buildx build --push \
  --platform linux/amd64 \
  --tag "$TAG_VERSIONED" \
  --tag "$TAG_LATEST" \
  --progress=plain \
  .
