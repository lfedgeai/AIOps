USE telemetry;
WITH span_buckets AS (
  SELECT service, DATE_FORMAT(ts_start, '%Y-%m-%d %H:%i:00') AS time_bucket, COUNT(DISTINCT span_id) AS span_count
  FROM spans
  WHERE ts_start >= NOW() - INTERVAL 30 DAY
  GROUP BY service, DATE_FORMAT(ts_start, '%Y-%m-%d %H:%i:00')
),
log_buckets AS (
  SELECT service, DATE_FORMAT(ts, '%Y-%m-%d %H:%i:00') AS time_bucket, COUNT(*) AS log_count
  FROM logs
  WHERE ts >= NOW() - INTERVAL 30 DAY
  GROUP BY service, DATE_FORMAT(ts, '%Y-%m-%d %H:%i:00')
),
metric_buckets AS (
  SELECT DATE_FORMAT(ts, '%Y-%m-%d %H:%i:00') AS time_bucket, COUNT(*) AS metric_count
  FROM metrics
  WHERE ts >= NOW() - INTERVAL 30 DAY
  GROUP BY DATE_FORMAT(ts, '%Y-%m-%d %H:%i:00')
)
SELECT
  s.service,
  s.time_bucket,
  s.span_count,
  COALESCE(l.log_count, 0) AS log_count,
  COALESCE(m.metric_count, 0) AS metric_count
FROM span_buckets s
LEFT JOIN log_buckets l ON s.service = l.service AND s.time_bucket = l.time_bucket
LEFT JOIN metric_buckets m ON s.time_bucket = m.time_bucket
ORDER BY (s.span_count + COALESCE(l.log_count, 0) + COALESCE(m.metric_count, 0)) DESC
LIMIT 20;
