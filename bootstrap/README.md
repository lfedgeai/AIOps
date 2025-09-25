# Bootstrap

In this phase bootstrapping is about creating a simple test harness for all the chosen permutations across 
- Models including model architecture and model modality and a combination thereof 
- Model enhancement (RAG, finetuning, sparsity (MoA/MoE, teacher/student, self-learning)

The aim is to test the approaches for performance across:
- Failure Perception
-- Failure Prevention
-- Failure Prediction
-- Anomaly detection

- Root Cause Analysis (RCA)
-- Failure Localization
-- Failure Category Classification
-- Root Cause Report Generation

- Assisted Remediation
-- Assisted Questioning
-- Mitigation Solution Gneration
-- Command Recommendation
-- Script Generation
-- Automatic Execution

## bootstrapping architecture

The idea is to have a simple podman based container harness consisting of the following:
- 3 containers based on container images on quay.io that simulate the distributed environments
- A single central log and metrics aggregation container to trigger model training and inference
- Secure OTEL based communication between all containers
- Benchmarking against selected benchmarks 

