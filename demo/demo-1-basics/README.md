# Getting Started and how to run the demo-1-basics  on your local laptop/desktop
Anomaly Detection + LLM FAISS Stack

This project runs an end-to-end anomaly detection pipeline using:

Isolation Forest anomaly detection service

LLM + FAISS inference service

Redis message queue

PostgreSQL database

The stack is orchestrated using Docker Compose and connected through a shared network.

**Step :1** Clone the repo from Git repository and navigate to demo folder as shown below

```bash
git clone https://github.com/lfedgeai/AIOps.git

 ```

**Step :2** build image using docker compose :

```bash
    cd AIOps/demo/demo-1-basics
    docker compose build --no-cache
    
 ```
**Step :3** Create Docker network 
The stack uses an external Docker network. Create it once before running services.
```bash
    docker network create anomaly-network
 ```
 verify 
 ```bash
    docker network ls
 ```
**Step :4** Build images
TBuild all services (LLM + Isolation Forest + dependencies).
```bash
    docker compose build --no-cache
 ```
 **Step :5**  Start infrastructure services (Database + Redis) and application services (LLM + Anomaly Detection)
Start both application components together.

```bash
   docker compose up 
 ```
 **Step : 6**  Verify services.
 Check logs to confirm everything is running correctly.

```bash
   docker compose logs 
```