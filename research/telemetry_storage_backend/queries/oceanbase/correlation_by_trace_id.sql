USE telemetry;
SELECT
  s.trace_id,
  s.service,
  COUNT(DISTINCT s.span_id) AS span_count,
  COUNT(DISTINCT l.ts) AS log_count,
  COUNT(DISTINCT m.ts) AS metric_count
FROM spans s
LEFT JOIN logs l ON l.trace_id = s.trace_id AND l.trace_id != ''
LEFT JOIN metrics m ON JSON_UNQUOTE(JSON_EXTRACT(m.labels, '$.trace_id')) = s.trace_id
WHERE s.ts_start >= NOW() - INTERVAL 30 DAY AND s.trace_id != ''
GROUP BY s.trace_id, s.service
ORDER BY (span_count + log_count + metric_count) DESC
LIMIT 20;
