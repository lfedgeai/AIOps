WITH span_buckets AS (
  SELECT service, TIME_FLOOR(__time, 'PT1M') AS time_bucket, COUNT(DISTINCT span_id) AS span_count
  FROM "telemetry_spans"
  WHERE __time >= CURRENT_TIMESTAMP - INTERVAL '30' DAY
  GROUP BY service, TIME_FLOOR(__time, 'PT1M')
),
log_buckets AS (
  SELECT service, TIME_FLOOR(__time, 'PT1M') AS time_bucket, COUNT(*) AS log_count
  FROM "telemetry_logs"
  WHERE __time >= CURRENT_TIMESTAMP - INTERVAL '30' DAY
  GROUP BY service, TIME_FLOOR(__time, 'PT1M')
),
metric_buckets AS (
  SELECT TIME_FLOOR(__time, 'PT1M') AS time_bucket, COUNT(*) AS metric_count
  FROM "telemetry_metrics"
  WHERE __time >= CURRENT_TIMESTAMP - INTERVAL '30' DAY
  GROUP BY TIME_FLOOR(__time, 'PT1M')
)
SELECT s.service, s.time_bucket, s.span_count, COALESCE(l.log_count, 0) AS log_count, COALESCE(m.metric_count, 0) AS metric_count
FROM span_buckets s
LEFT JOIN log_buckets l ON s.service = l.service AND s.time_bucket = l.time_bucket
LEFT JOIN metric_buckets m ON s.time_bucket = m.time_bucket
ORDER BY (s.span_count + COALESCE(l.log_count, 0) + COALESCE(m.metric_count, 0)) DESC
LIMIT 20
