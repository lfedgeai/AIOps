-- Create database and tables for logs, spans, metrics
CREATE DATABASE IF NOT EXISTS telemetry;
USE telemetry;

-- Logs: duplicate key (no aggregation), partition by day, distributed by hash(service)
CREATE TABLE IF NOT EXISTS logs (
  ts DATETIME(6),
  service VARCHAR(256),
  level VARCHAR(64),
  message TEXT,
  trace_id VARCHAR(64),
  span_id VARCHAR(64),
  attrs JSON
) ENGINE=OLAP
DUPLICATE KEY(`ts`, `service`)
DISTRIBUTED BY HASH(`service`) BUCKETS 8
PROPERTIES (
  "replication_num" = "1"
);

CREATE TABLE IF NOT EXISTS spans (
  ts_start DATETIME(6),
  trace_id VARCHAR(64),
  ts_end   DATETIME(6),
  span_id VARCHAR(64),
  parent_span_id VARCHAR(64),
  service VARCHAR(256),
  name VARCHAR(256),
  duration_ms BIGINT,
  attributes JSON
) ENGINE=OLAP
DUPLICATE KEY(`ts_start`, `trace_id`)
DISTRIBUTED BY HASH(`trace_id`) BUCKETS 8
PROPERTIES (
  "replication_num" = "1"
);

-- Metrics: duplicate key (we do rollups via queries)
CREATE TABLE IF NOT EXISTS metrics (
  ts DATETIME(6),
  metric_name VARCHAR(256),
  value DOUBLE,
  labels JSON
) ENGINE=OLAP
DUPLICATE KEY(`ts`, `metric_name`)
DISTRIBUTED BY HASH(`metric_name`) BUCKETS 8
PROPERTIES (
  "replication_num" = "1"
);

