# baseline/Dockerfile
FROM alpine:3.18 AS builder

ARG OTELCOL_VERSION=0.136.0
ARG ARCH=amd64
RUN apk add --no-cache curl ca-certificates

WORKDIR /tmp

RUN curl -L --fail "https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v${OTELCOL_VERSION}/otelcol-contrib_${OTELCOL_VERSION}_linux_${ARCH}.tar.gz" \
    -o otelcol.tar.gz && \
    tar -xzf otelcol.tar.gz && \
    chmod +x otelcol-contrib

FROM scratch

COPY --from=builder /tmp/otelcol-contrib /otelcol

COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt

WORKDIR /etc/otelcol

EXPOSE 4317 4318

ENTRYPOINT ["/otelcol", "--config", "/etc/otelcol/otel-config.yaml"]
