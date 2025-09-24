# Bootstrap

In this phase bootstrapping is about creating a simple test harness for all the chosen permutations across 
- Models including model architecture and model modality
- Enhancement 

## bootstrapping architecture

The idea is to have a simple podman based container harness consisting of the following:
- 3 containers based on container images on quay.io that simulate the distributed environments
- A single central log and metrics aggregation container to trigger model training and inference
- Secure OTEL based communication between all containers
- Benchmarking against selected benchmarks 

