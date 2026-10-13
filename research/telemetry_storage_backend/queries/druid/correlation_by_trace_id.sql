-- Cross-correlation by trace_id (spans + logs; metric_count=0 until metrics has trace_id)
-- Druid join requires equality only; empty trace_id filtered in WHERE
SELECT
  s.trace_id,
  s.service,
  COUNT(DISTINCT s.span_id) AS span_count,
  COUNT(DISTINCT l.__time) AS log_count,
  0 AS metric_count
FROM "telemetry_spans" s
LEFT JOIN "telemetry_logs" l ON l.trace_id = s.trace_id
WHERE s.__time >= CURRENT_TIMESTAMP - INTERVAL '30' DAY AND s.trace_id <> ''
GROUP BY s.trace_id, s.service
ORDER BY (span_count + log_count) DESC
LIMIT 20
