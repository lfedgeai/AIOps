CREATE DATABASE IF NOT EXISTS telemetry;

CREATE TABLE IF NOT EXISTS telemetry.logs (
  ts DateTime64(6),
  service String,
  level String,
  message String,
  trace_id String,
  span_id String,
  attrs String
) ENGINE = MergeTree()
ORDER BY (ts, service);

CREATE TABLE IF NOT EXISTS telemetry.spans (
  ts_start DateTime64(6),
  trace_id String,
  ts_end DateTime64(6),
  span_id String,
  parent_span_id String,
  service String,
  name String,
  duration_ms Int64,
  attributes String
) ENGINE = MergeTree()
ORDER BY (ts_start, trace_id);

CREATE TABLE IF NOT EXISTS telemetry.metrics (
  ts DateTime64(6),
  metric_name String,
  value Float64,
  labels String
) ENGINE = MergeTree()
ORDER BY (ts, metric_name);
