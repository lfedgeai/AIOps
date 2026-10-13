-- Spans with 5xx status (from attributes) or high duration - error trace analysis
-- Key "http.status_code" needs quoting in JSON path: $."http.status_code"
SELECT service, COUNT(*) AS error_span_count
FROM spans
WHERE ts_start >= NOW() - INTERVAL 30 DAY
  AND (
    COALESCE(CAST(json_extract_string(attributes, '$."http.status_code"') AS INT), 0) >= 500
    OR duration_ms > 5000
  )
GROUP BY service
ORDER BY error_span_count DESC
LIMIT 20;
