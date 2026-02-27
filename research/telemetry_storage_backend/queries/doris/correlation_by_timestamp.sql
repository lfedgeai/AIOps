-- Cross-correlation by service + 1-minute time bucket (works without trace_id)
-- Correlates logs, spans, metrics in the same service and time window
WITH span_buckets AS (
  SELECT service, date_trunc(ts_start, 'minute') AS time_bucket, COUNT(DISTINCT span_id) AS span_count
  FROM spans
  WHERE ts_start >= NOW() - INTERVAL 30 DAY
  GROUP BY service, date_trunc(ts_start, 'minute')
),
log_buckets AS (
  SELECT service, date_trunc(ts, 'minute') AS time_bucket, COUNT(*) AS log_count
  FROM logs
  WHERE ts >= NOW() - INTERVAL 30 DAY
  GROUP BY service, date_trunc(ts, 'minute')
),
metric_buckets AS (
  SELECT date_trunc(ts, 'minute') AS time_bucket, COUNT(*) AS metric_count
  FROM metrics
  WHERE ts >= NOW() - INTERVAL 30 DAY
  GROUP BY date_trunc(ts, 'minute')
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
