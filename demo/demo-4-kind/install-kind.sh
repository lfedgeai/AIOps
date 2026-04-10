#!/bin/bash

# Variables
# KIND_VERSION=${1:-v0.20.0}  # Default Kind version, can be overridden
KIND_VERSION=${KIND_VERSION:-v0.25.0}

INSTALL_PATH="/usr/local/bin/kind"

# Function to check if a command exists
check_command() {
  command -v "$1" >/dev/null 2>&1
}

# Install Docker
install_docker() {
  if check_command docker; then
    echo "Docker is already installed."
  else
    echo "Installing Docker..."
    # Installing Docker for Ubuntu/Debian systems
    sudo apt-get update
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    sudo systemctl enable docker
    sudo systemctl start docker
    echo "Docker installed successfully."
  fi
}




install_kind() {
  if command -v kind >/dev/null 2>&1; then
    echo "Kind is already installed."
  else
    KIND_VERSION=${KIND_VERSION:-v0.20.0} # Default to a specific version if not set
    echo "Installing Kind version $KIND_VERSION..."

    # Detect OS
    OS=$(uname | tr '[:upper:]' '[:lower:]')
    if [[ "$OS" == "mingw"* || "$OS" == "cygwin"* ]]; then
      OS="windows"
    fi

    # Detect architecture
    ARCH=$(uname -m)
    if [[ "$ARCH" == "x86_64" || "$ARCH" == "amd64" ]]; then
      ARCH="amd64"
    elif [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
      ARCH="arm64"
    else
      echo "Unsupported architecture: $ARCH"
      exit 1
    fi

    # Set the Kind download URL
    KIND_URL="https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-${OS}-${ARCH}"
    echo "Downloading Kind binary from $KIND_URL..."

    # Download the Kind binary
    curl -Lo kind "$KIND_URL"

    # For Windows, add .exe extension
    if [[ "$OS" == "windows" ]]; then
      mv kind kind.exe
      chmod +x kind.exe
      # Add Kind to PATH (assuming Git Bash uses ~/.bashrc or similar)
      export PATH="$PATH:$(pwd)"
      echo 'export PATH="$PATH:$(pwd)"' >> ~/.bashrc
      echo "Kind version $KIND_VERSION installed successfully. Please restart Git Bash or source your ~/.bashrc file."
    else
      # Make it executable and move to /usr/local/bin for Unix-like systems
      chmod +x ./kind
      sudo mv ./kind /usr/local/bin/kind
      echo "Kind version $KIND_VERSION installed successfully."
    fi
  fi
}

# Install Helm
install_helm() {
  if check_command helm; then
    echo "Helm is already installed."
  else
    echo "Installing Helm..."
    curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
    echo "Helm installed successfully."
  fi
}
# Delete Kind, Docker, and Helm installations
delete_kind_install() {
  echo "Deleting Kind, Docker, and Helm installations..."
  
  # Delete Kind
  if check_command kind; then
    sudo rm -f /usr/local/bin/kind
    echo "Kind deleted."
  else
    echo "Kind is not installed."
  fi

  # Delete Helm
  if check_command helm; then
    sudo rm -f /usr/local/bin/helm
    echo "Helm deleted."
  else
    echo "Helm is not installed."
  fi

  # Delete Docker
  if check_command docker; then
    echo "Removing Docker and its dependencies..."
    sudo apt-get purge -y docker-ce docker-ce-cli containerd.io
    sudo apt-get autoremove -y
    sudo rm -rf /var/lib/docker
    sudo rm -rf /var/lib/containerd
    echo "Docker deleted."
  else
    echo "Docker is not installed."
  fi

  echo "All installations removed."
}

# Check and install dependencies
check_dependencies() {
  echo "Checking dependencies..."
  install_docker
  install_kind
  install_helm
}
# Delete Docker
delete_docker() {
  if check_command docker; then
    echo "Removing Docker and its dependencies..."
    sudo apt-get purge -y docker-ce docker-ce-cli containerd.io
    sudo apt-get autoremove -y
    sudo rm -rf /var/lib/docker
    sudo rm -rf /var/lib/containerd
    echo "Docker deleted."
  else
    echo "Docker is not installed."
  fi
}

# Delete Kind
delete_kind() {
  if check_command kind; then
    sudo rm -f /usr/local/bin/kind
    echo "Kind deleted."
  else
    echo "Kind is not installed."
  fi
}

# Delete Helm
delete_helm() {
  if check_command helm; then
    sudo rm -f /usr/local/bin/helm
    echo "Helm deleted."
  else
    echo "Helm is not installed."
  fi
}

install_all() {
  echo "Installing all components: Kind, Helm, and Docker..."
  install_kind
  install_helm
  # install_docker
  echo "All components installed successfully."
}

# Delete All
delete_all() {
  echo "Deleting all installations..."
  delete_kind
  delete_helm
  # delete_docker
  echo "All installations removed."
}

# Function to verify installation
verify_installation() {
    echo "Verifying Kind installation..."
    if command -v kind &> /dev/null; then
        echo "Kind version: $(kind version)"
    else
        echo "Kind installation failed. Please check the logs."
        exit 1
    fi
}

# Main execution logic
case "$1" in
  install)
    if [ -z "$2" ]; then
      install_all
    else
      case "$2" in
        kind)
          install_kind
          ;;
        helm)
          install_helm
          ;;
        docker)
          install_docker
          ;;
        *)
          echo "Invalid option for install. Use: kind, helm, docker, or leave empty to install all."
          exit 1
          ;;
      esac
    fi
    ;;
  delete)
    if [ -z "$2" ]; then
      delete_all
    else
      case "$2" in
        kind)
          delete_kind
          ;;
        helm)
          delete_helm
          ;;
        docker)
          delete_docker
          ;;
        *)
          echo "Invalid option for delete. Use: kind, helm, docker, or leave empty to delete all."
          exit 1
          ;;
      esac
    fi
    ;;
  *)
    echo "Usage: $0 {install|delete} [kind|helm|docker]"
    echo "Example:"
    echo "  $0 install           # Install all components"
    echo "  $0 install kind      # Install Kind only"
    echo "  $0 delete            # Delete all components"
    echo "  $0 delete helm       # Delete Helm only"
    exit 1
    ;;
esac
