-- Create database and tables for logs, spans, metrics
--
-- Every canonical query in queries/doris/ is served by one of:
--   * sort key prefix + partition pruning (time-range scans, ORDER BY ts DESC LIMIT n)
--   * inverted index (equality / range on trace_id, service, level, duration_ms, http_status_code)
--   * full-text inverted index (MATCH_ANY on logs.message)
-- See docs/DORIS_TUNING.md for the query -> access path mapping and the reasoning.
CREATE DATABASE IF NOT EXISTS telemetry;
USE telemetry;

-- Logs: time-ordered, daily auto partitions, random bucketing (no hot service skew).
CREATE TABLE IF NOT EXISTS logs (
  ts DATETIME(6) NOT NULL,
  service VARCHAR(256),
  level VARCHAR(64),
  message TEXT,
  trace_id VARCHAR(64),
  span_id VARCHAR(64),
  attrs JSON,
  INDEX idx_service (service) USING INVERTED,
  INDEX idx_level (level) USING INVERTED,
  INDEX idx_trace_id (trace_id) USING INVERTED,
  INDEX idx_message (message) USING INVERTED PROPERTIES("parser" = "unicode", "lower_case" = "true", "support_phrase" = "true")
) ENGINE=OLAP
DUPLICATE KEY(`ts`, `service`)
AUTO PARTITION BY RANGE (date_trunc(`ts`, 'day')) ()
DISTRIBUTED BY RANDOM BUCKETS 8
PROPERTIES (
  "replication_num" = "1",
  "compaction_policy" = "time_series"
);

-- Spans: time-ordered, hash by trace_id so a single-trace lookup touches one bucket.
-- http_status_code is promoted out of the attributes JSON at load time so it can be indexed.
CREATE TABLE IF NOT EXISTS spans (
  ts_start DATETIME(6) NOT NULL,
  trace_id VARCHAR(64),
  ts_end   DATETIME(6),
  span_id VARCHAR(64),
  parent_span_id VARCHAR(64),
  service VARCHAR(256),
  name VARCHAR(256),
  duration_ms BIGINT,
  http_status_code INT,
  attributes JSON,
  INDEX idx_trace_id (trace_id) USING INVERTED,
  INDEX idx_service (service) USING INVERTED,
  INDEX idx_duration_ms (duration_ms) USING INVERTED,
  INDEX idx_http_status_code (http_status_code) USING INVERTED
) ENGINE=OLAP
DUPLICATE KEY(`ts_start`, `trace_id`)
AUTO PARTITION BY RANGE (date_trunc(`ts_start`, 'day')) ()
DISTRIBUTED BY HASH(`trace_id`) BUCKETS 8
PROPERTIES (
  "replication_num" = "1",
  "compaction_policy" = "time_series"
);

-- Metrics: time-ordered; trace_id promoted out of labels JSON for the trace correlation join.
CREATE TABLE IF NOT EXISTS metrics (
  ts DATETIME(6) NOT NULL,
  metric_name VARCHAR(256),
  value DOUBLE,
  trace_id VARCHAR(64),
  labels JSON,
  INDEX idx_metric_name (metric_name) USING INVERTED,
  INDEX idx_trace_id (trace_id) USING INVERTED
) ENGINE=OLAP
DUPLICATE KEY(`ts`, `metric_name`)
AUTO PARTITION BY RANGE (date_trunc(`ts`, 'day')) ()
DISTRIBUTED BY HASH(`metric_name`) BUCKETS 8
PROPERTIES (
  "replication_num" = "1"
);
