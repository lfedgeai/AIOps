#!/bin/bash

CLUSTER_NAME=${1:-kind-cluster} # Default cluster name if not provided
ACTION=${2:-create} # Action: create, delete, or status
CONFIG_FILE="kind-config.yaml" # Default Kind config file
CURRENT_DIR=$(pwd)
NAMESPACE="osclimate"
# Function to create a Kind cluster
create_cluster() {
    echo "Creating Kind cluster: $CLUSTER_NAME..."
    cat <<EOF > $CONFIG_FILE
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4

nodes:
  - role: control-plane
    extraMounts:
      - hostPath: $CURRENT_DIR/dags   # Replace with your local directory
        containerPath: /dags  # Path inside the Kind container
  - role: worker

EOF

    kind create cluster --name "$CLUSTER_NAME" --config "$CONFIG_FILE"
    echo "Cluster $CLUSTER_NAME created successfully."
    echo "Setting resource limits for Kind containers..."
    docker update --cpus 4 --memory 12g --memory-swap 12g "$CLUSTER_NAME-control-plane"
    docker update --cpus 4 --memory 12g --memory-swap 12g "$CLUSTER_NAME-worker"
  
  echo "Cluster created and resource limits applied."
}

# Function to delete a Kind cluster
delete_cluster() {
    
    echo "Deleting Kind cluster: $CLUSTER_NAME..."

    # Delete PVCs
    for pvc in $(kubectl get pvc -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}'); do
      echo "Deleting PVC $pvc in namespace $NAMESPACE"
      kubectl delete pvc $pvc -n $NAMESPACE
    done

  
    kubectl delete pv --grace-period=0 --force
    
    echo "PVC and PV cleanup complete."
    kind delete cluster --name "$CLUSTER_NAME"
        # kubectl config delete-context "$CLUSTER_NAME"

    echo "Cluster $CLUSTER_NAME deleted successfully."
}

# Function to get cluster status
status_cluster() {
    echo "Fetching status for Kind cluster: $CLUSTER_NAME..."
    kubectl cluster-info --context "kind-$CLUSTER_NAME"
}
# Main logic
case "$ACTION" in
    create)
        create_cluster
        ;;
    delete)
        delete_cluster
        ;;
    status)
        status_cluster
        ;;
    *)
        echo "Usage: $0 <cluster-name> <create|delete|status>"
        exit 1
        ;;
esac
