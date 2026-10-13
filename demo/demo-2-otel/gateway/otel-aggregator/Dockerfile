# Stage 1: Build / download OTEL Collector
FROM alpine:3.18 AS builder

RUN apk add --no-cache curl ca-certificates
WORKDIR /tmp

ARG OTELCOL_VERSION=0.136.0

RUN curl -L --fail "https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v${OTELCOL_VERSION}/otelcol-contrib_${OTELCOL_VERSION}_linux_amd64.tar.gz" \
    -o otelcol.tar.gz && \
    tar -xzf otelcol.tar.gz

# Stage 2: Final minimal image
FROM alpine:3.18

# Copy OTEL binary and CA certificates
COPY --from=builder /tmp/otelcol-contrib /usr/local/bin/otelcol
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt

# Set working directory for config
WORKDIR /etc/otelcol

# Copy config file (ensure this exists in build context)
COPY ./etc/otel-config.yaml ./otel-config.yaml

ENTRYPOINT ["/usr/local/bin/otelcol", "--config", "/etc/otelcol/otel-config.yaml"]
