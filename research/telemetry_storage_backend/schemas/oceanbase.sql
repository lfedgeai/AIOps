-- OceanBase MySQL-compatible schema for telemetry
-- Same column layout as Doris; OceanBase uses MySQL protocol

CREATE DATABASE IF NOT EXISTS telemetry;
USE telemetry;

-- Logs
CREATE TABLE IF NOT EXISTS logs (
  ts DATETIME(6),
  service VARCHAR(256),
  level VARCHAR(64),
  message MEDIUMTEXT,
  trace_id VARCHAR(64),
  span_id VARCHAR(64),
  attrs JSON
);

-- Spans
CREATE TABLE IF NOT EXISTS spans (
  ts_start DATETIME(6),
  trace_id VARCHAR(64),
  ts_end DATETIME(6),
  span_id VARCHAR(64),
  parent_span_id VARCHAR(64),
  service VARCHAR(256),
  name VARCHAR(256),
  duration_ms BIGINT,
  attributes JSON
);

-- Metrics
CREATE TABLE IF NOT EXISTS metrics (
  ts DATETIME(6),
  metric_name VARCHAR(256),
  value DOUBLE,
  labels JSON
);
