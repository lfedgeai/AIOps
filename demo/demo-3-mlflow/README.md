# MLflow Tracking Stack (Postgres + MinIO)

Docker Compose setup for **MLflow** using **PostgreSQL** for experiment metadata and **MinIO** for S3-compatible artifact storage.

## Features
- **MLflow Tracking Server**: Centralized dashboard for experiments.
- **PostgreSQL 15**: Relational backend for high-performance metadata tracking.
- **MinIO**: High-performance, S3-compatible object storage for model artifacts.
- **Automated Setup**: Auto-creates the required MinIO bucket on startup.
- **Environment Driven**: Fully configurable via `.env` file.

## Setup Instructions

### 1. Prerequisites
Ensure you have [Docker](https://docs.docker.com) and [Docker Compose](https://docs.docker.com) installed.

### 2. Configure Environment
Create a `.env` file in the root directory (refer to the `.env` template provided).

### 3. Configure Environment
Create a `.env` file in the root directory to store your credentials securely:
```bash
# PostgreSQL Configuration
POSTGRES_USER=mlflow_user
POSTGRES_PASSWORD=postgres123$
POSTGRES_DB=mlflow_db

# MinIO Configuration
MINIO_ROOT_USER=minioAdmin
MINIO_ROOT_PASSWORD=admin123$
MINIO_STORAGE_BUCKET=mlflow

# MLflow Configuration
MLFLOW_PORT=5001

### 4. Build and Start

# Build the custom image and start containers
```bash
docker-compose up -d --build
```bash

# Verify all services are running
```bash
docker-compose ps
```bash

### 4. Access Points
# access MLflow UI	http://localhost:5000	N/A

# access minIo UI console http://localhost:9001	See .env (MINIO_ROOT_USER)

## 🧪 Running the MLflow Client

The `mlflow_client` service automatically runs the `train.py` script upon startup.

### Execute Training
To trigger a new training run from the client container:
```bash
docker-compose up mlflow_client
