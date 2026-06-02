# Kube Edge platform setup

## K3s + KubeEdge (RHEL)+ OpenTelemetry + Prometheus + Grafana

Control Plane K3 (Raspberry Pi)
 ├─ OTEL Gateway Collector
 ├─ Prometheus
 ├─ Grafana
 ├─ KubeEdge CloudCore
 └─ Kubernetes Dashboard
Edge Node
 └─ OTEL Collector Agent
      ↓ OTLP (4317)


# Environment
## Control Plane
    Raspberry Pi
    Raspberry Pi OS 32-bit (armv7l)
    K3s v1.31.6+k3s1
    IP: 192.168.1.76
## Edge Node
    RHEL / Rocky Linux / AlmaLinux
    amd64/x86_64
    KubeEdge EdgeCore v1.19.0

## Install K3s

```bash
    chmod +x install-control-plane.sh
    ./install-control-plane.sh

 ```
## Verify CloudCore Before Generating Token
    ```bash
    kubectl get pods -n kubeedge

    ```


## Get KubeEdge Join Token on Raspberry Pi Control Plane

```bash

    export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

    sudo keadm gettoken --kube-config=/etc/rancher/k3s/k3s.yaml

 ```
## or run the following script to get token from control plane

    ```bash
    chmod +x get-kubeedge-token.sh
    ./get-kubeedge-token.sh

    ```


 ## Install edge node
 ```bash
    chmod +x install-edge-node.sh
    ./install-edge-node.sh '<PASTE_KUBEEDGE_TOKEN_HERE>'

 ```



## Verify Edge Joined On Raspberry Pi:

    ```bash
    kubectl get nodes

    ```
## Expected:

    NAME                    STATUS   ROLES
    raspberrypi             Ready    control-plane,master
    localhost.localdomain   Ready    agent