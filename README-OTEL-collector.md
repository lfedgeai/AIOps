# OpenTelemetry Collector – Host Metrics on Podman/Edge device

OTEL collector collects host-level metrics (CPU, memory, filesystem, disk, load, and network) from edge device or any host machine and exports them to back application or storage. As or Prometheus, OTLP, and the debug console.

## high level component overview

### Receiver: Host Metrics

Collects CPU, memory, filesystem, disk, load, and network metrics.

root_path: /host is important when running inside a container on macOS or Podman because the host’s file system is mounted under /host.

### Processors:

resource: adds or updates a host attribute with a fixed value (e.g., Edge-GreatJey-macbook-pro).

attributes: inserts host attribute if not already present.

batch: batches metrics before exporting.

### Exporters:

prometheus: exposes metrics on 0.0.0.0:9464 for scraping by Prometheus.

otlp: sends metrics to an OTLP endpoint (replace <aggregator-endpoint-url>:<port>).

debug: prints detailed telemetry data to the console for troubleshooting.

### Service Pipeline:

Single metrics pipeline combining all of the above.

# Running with Docker/Podman

## Docker :

```bash

docker compose up --build

```
