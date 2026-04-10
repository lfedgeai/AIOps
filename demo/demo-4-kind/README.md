# Kind Cluster + OpenTelemetry Operator Setup Guide

# Kind Cluster:

kind is a tool for running local Kubernetes cluster using Docker container.
kind was primarily designed for testing Kubernetes itself, but can be used for local development.
Kind is ligt weight Kubernetes platform consume less reosuce so that developer can run most of the Data Mesh compoenrts locally and alos it is very close to target kubernetes platform like OpenShift . 

# Prerequisites - Mac & Linux 
    
    1.Docker
    2.Helm

**Mac & Linux**
All prerequisties are included in the kind install script. kind install script. If any errors are occured during installtion, it could be local environment specific issue that need to addressed based on environment variation. 

# Prerequisites - Windows
    
    1.Docker
    2.Helm

For Windows environment **Docker and Helm need** need to be installed before running install-kind.sh. Kind install script only supports kind installation. For Windows, run **./install-kind.sh install kind**.

# Kind Installation

**Step 1 :** To Install Kind, run the script with the desired action: You need to run this scrit only first time.  If you want to destroy the deployment, follow the instruction at the bottom of this README file. 

# Install Kind ( Mac , Linux & windows)
   
```bash
./install-kind.sh install kind
 ```

 # Install Kind , helm and Docker (Mac &Linux ). 

```bash
chmod +x install-kind.sh
```
**Windows only** To kind installation take effect in Windows environment , restart Git Bash or source your ~/.bashrc file."

```bash
source ~/.bashrc
 ```

**Step 2 :** Create a Kind Cluster: You need to run this script only first time. 

Note : Modify the host path and airflow dags folder specific to your enviroment in **kind-cluster.sh** .
The kind-config.yaml will be generated dynamically when you run kind-cluster.sh.

"- hostPath: **<home>/kind-development/dags**   # Replace with your local directory
        containerPath: /dags  # Path inside the Kind container"

```bash
chmod +x kind-cluster.sh
```
To create a cluster
```bash
./kind-cluster.sh aiops-cluster create
```
   
**Step 3:** once Step 2 completed, verify kind cluster is created successfully by runing the following script.

```bash
kubectl cluster-info --context kind-aiops-cluster
 ```

# Installing otel through helm 

OpenTelemetry Operator Setup Guide

This guide helps you install:


* Helm
* OpenTelemetry Operator



# Step 1:  Install Helm

```bash
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

helm version
```



#  Step 2: Install OpenTelemetry Operator

## Add Helm repo

```bash
helm repo add open-telemetry https://open-telemetry.github.io/opentelemetry-helm-charts
helm repo update
```

---

## Install WITHOUT cert-manager (recommended for dev)

```bash
helm install otel-operator open-telemetry/opentelemetry-operator \
  -n opentelemetry-operator \
  --create-namespace \
  --set admissionWebhooks.certManager.enabled=false
```

---

# step 3: Verify Installation

```bash
kubectl get pods -n opentelemetry-operator
```

Expected:

```text
otel-operator-xxxxx   Running
```





# Summary

* Kind cluster created
* OpenTelemetry Operator installed






