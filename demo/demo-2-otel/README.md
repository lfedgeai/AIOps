
# Getting Started  — How to run the OTEL + Storage Stack on your local laptop/desktop

Observability + Lakehouse Storage Stack

This project runs an end-to-end telemetry pipeline using:

OpenTelemetry collectors and aggregator for metrics, logs, and traces

Apache Iceberg REST catalog for lakehouse table management

MinIO as S3-compatible object storage

PostgreSQL for Iceberg metadata storage

Edge exporter service to ingest and forward telemetry data

**Step :1** Clone the repo from Git repository and navigate to demo folder as shown below

```bash
git clone https://github.com/lfedgeai/AIOps.git

 ```

**Step :2** build image using docker compose :

```bash
    cd AIOps/demo/demo-2-otel
    
 ```
**Step :3** Create Docker network 
The stack uses an external Docker network. Create it once before running services.
```bash
    docker network create edge-network
 ```
 verify 
 ```bash
    docker network ls
 ```

 **Step :4**  start Storage Stack (Postgres + MinIO + Iceberg)
Navigate to the folder containing:postgres ,minio ,iceberg-catalog compose file

```bash
   docker compose -f docker-compose.persistance.yml up  --build
 ```

 **Step :4**  Start Edge + OpenTelemetry Stack

Navigate to the folder containing:
edge-exporter
otel-collector
otel-aggregator
otel-testapp compose file

```bash
   docker compose -f docker-compose.yaml up --build
 ```

 **Step :4** To Check Logs for both stoaage and edge 

```bash
  docker compose -f docker-compose.persistance.yaml logs 
  docker compose -f docker-compose.yml logs 
 ```

