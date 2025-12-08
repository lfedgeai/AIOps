# FROM python:3.11-slim

# # Set working directory
# WORKDIR /app

# # Copy requirements and install
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy all other files (including otel_grpc_server.py)
# COPY ./app/ .

# EXPOSE 50051 8000
# # Run the gRPC server
# CMD ["python", "otel_grpc_server.py"]

FROM python:3.11-slim

# Avoid interactive prompts and speed up builds
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install system deps + Python deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libpq-dev \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy all source files (including otel_grpc_server.py)
COPY ./app/ .

# Expose both gRPC (50051) and FastAPI/HTTP (8000) ports
EXPOSE 50051 8000

# Default command: start the gRPC receiver (and FastAPI if included)
CMD ["python", "otel_grpc_server.py"]

