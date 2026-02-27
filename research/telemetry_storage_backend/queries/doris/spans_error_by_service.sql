-- Spans with 5xx status (from attributes) or high duration - error trace analysis
SELECT service, COUNT(*) AS error_span_count
FROM spans
WHERE ts_start >= NOW() - INTERVAL 30 DAY
  AND (
    json_extract_string(attributes, '$.http.status_code') >= '500'
    OR duration_ms > 5000
  )
GROUP BY service
ORDER BY error_span_count DESC
LIMIT 20;
