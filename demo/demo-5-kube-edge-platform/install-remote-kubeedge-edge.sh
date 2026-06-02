#!/bin/bash
set -e

CLOUDCORE_ENDPOINT="${CLOUDCORE_ENDPOINT:-kubeedge.jplabusa.com:10000}"
KUBEEDGE_VERSION="${KUBEEDGE_VERSION:-v1.19.0}"
OTEL_VERSION="${OTEL_VERSION:-0.102.1}"
CONTROL_PLANE_IP="${CONTROL_PLANE_IP:-69.129.195.230}"

if [ -z "$1" ]; then
  echo "Usage:"
  echo "  ./install-kubeedge-edge.sh '<KUBEEDGE_TOKEN>'"
  echo ""
  echo "Example:"
  echo "  ./install-kubeedge-edge.sh 'your-token-here'"
  exit 1
fi

TOKEN="$1"

echo "Installing required packages..."
sudo dnf install -y wget tar conntrack-tools socat iproute iptables nc || true

echo "Installing keadm ${KUBEEDGE_VERSION}..."
cd /tmp
rm -rf keadm-${KUBEEDGE_VERSION}-linux-amd64*
wget -q https://github.com/kubeedge/kubeedge/releases/download/${KUBEEDGE_VERSION}/keadm-${KUBEEDGE_VERSION}-linux-amd64.tar.gz
tar -xzf keadm-${KUBEEDGE_VERSION}-linux-amd64.tar.gz
sudo cp keadm-${KUBEEDGE_VERSION}-linux-amd64/keadm/keadm /usr/local/bin/keadm
sudo chmod +x /usr/local/bin/keadm

echo "keadm version:"
/usr/local/bin/keadm version

echo "Testing CloudCore connectivity..."
nc -zv "$(echo $CLOUDCORE_ENDPOINT | cut -d: -f1)" "$(echo $CLOUDCORE_ENDPOINT | cut -d: -f2)"

echo "Cleaning previous EdgeCore if present..."
sudo systemctl stop edgecore 2>/dev/null || true
sudo systemctl disable edgecore 2>/dev/null || true
sudo pkill edgecore 2>/dev/null || true
sudo rm -rf /etc/kubeedge
sudo rm -rf /var/lib/kubeedge
sudo rm -rf /var/log/kubeedge
sudo rm -f /etc/systemd/system/edgecore.service
sudo systemctl daemon-reload
sudo systemctl reset-failed

echo "Joining KubeEdge..."
sudo /usr/local/bin/keadm join \
  --cloudcore-ipport="${CLOUDCORE_ENDPOINT}" \
  --token="${TOKEN}"

echo "Installing OpenTelemetry Collector Edge Agent..."
cd /tmp
rm -f otelcol-contrib*
wget -q https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v${OTEL_VERSION}/otelcol-contrib_${OTEL_VERSION}_linux_amd64.tar.gz
tar -xzf otelcol-contrib_${OTEL_VERSION}_linux_amd64.tar.gz
sudo cp otelcol-contrib /usr/local/bin/otelcol-contrib
sudo chmod 755 /usr/local/bin/otelcol-contrib
sudo chown root:root /usr/local/bin/otelcol-contrib

if command -v restorecon >/dev/null 2>&1; then
  sudo restorecon -v /usr/local/bin/otelcol-contrib || true
fi

sudo mkdir -p /etc/otelcol-contrib

cat <<EOF | sudo tee /etc/otelcol-contrib/config.yaml
receivers:
  hostmetrics:
    collection_interval: 10s
    scrapers:
      cpu:
      memory:
      disk:
      filesystem:
      network:
      load:

processors:
  batch:
  resource:
    attributes:
      - key: edge_node
        value: $(hostname)
        action: upsert

exporters:
  otlp:
    endpoint: 192.168.1.76:4317
    tls:
      insecure: true

service:
  pipelines:
    metrics:
      receivers: [hostmetrics]
      processors: [resource, batch]
      exporters: [otlp]
EOF

cat <<EOF | sudo tee /etc/systemd/system/otelcol-contrib.service
[Unit]
Description=OpenTelemetry Collector Contrib Edge Agent
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/local/bin/otelcol-contrib --config=/etc/otelcol-contrib/config.yaml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable otelcol-contrib
sudo systemctl restart otelcol-contrib

echo ""
echo "DONE."
echo ""
echo "Check EdgeCore:"
echo "  sudo systemctl status edgecore --no-pager"
echo ""
echo "Check OTEL agent:"
echo "  sudo systemctl status otelcol-contrib --no-pager"
echo ""
echo "On Raspberry Pi control plane:"
echo "  kubectl get nodes"
echo "  kubectl get pods -A -o wide"