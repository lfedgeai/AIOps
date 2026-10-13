This project-implements an Anoamaly Detection system using isolation forest model , with experiment tracking via MLflow and artifact storage using minio 


Build microservice architecture using:
   FastAPI - API layer 
   redis - Background job queue
   PostgresSQL - metadata storage 
   MLflow - Experiment tracking 
   Minio - model & artifact storage 


**Step :1** Clone the repo from Git repository and navigate to demo folder as shown below

```bash
git clone https://github.com/lfedgeai/AIOps.git

 ```

**Step :2** build image using docker compose :

```bash
    cd AIOps/demo/demo-1-basics/anomaly_isolation_forest
   
 ```
**Step :3** Create Docker network 
The stack uses an external Docker network. Create it once before running services.
```bash
    docker network create anomaly-network
 ```

 **Step :4** Build images
To build the service 
```bash
    docker compose build --no-cache
 ```
 **Step :5**  Run the services 

```bash
   docker compose up 
 ```
 **Step : 6**  Verify services.
 Check logs to confirm everything is running correctly.

```bash
   docker compose logs 
```
**Step : 7**  check the list of containers 

```bash
   docker container list  
```
CONTAINER ID  IMAGE                             COMMAND               CREATED         STATUS         PORTS                             NAMES
2a749afb8ddb  docker.io/library/redis:7-alpine  redis-server          10 minutes ago  Up 10 minutes  0.0.0.0:6379->6379/tcp            anomaly-redis
747597baf8d0  docker.io/library/postgres:15     postgres              10 minutes ago  Up 10 minutes                                    mlflow_db
b7c7fe141144  docker.io/minio/minio:latest      server /data --co...  10 minutes ago  Up 9 minutes   0.0.0.0:9000-9001->9000-9001/tcp  mlflow_minio
a92845aa7b1a  ghcr.io/mlflow/mlflow:latest      bash -c  pip inst...  10 minutes ago  Up 9 minutes   0.0.0.0:5000->5000/tcp            mlflow_server

 **Step : 6**  Once all services are running successfully , you can access the MLflow UI to monitor your experiments 
 https://localhost:5000/

  In mlflow ui  can see the tracking of experiment named  anomaly-detction 
  Models are stored in MinIO