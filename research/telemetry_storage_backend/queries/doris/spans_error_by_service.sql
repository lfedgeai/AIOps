-- Spans with 5xx status or high duration - error trace analysis.
-- http_status_code is promoted from attributes["http.status_code"] at load time, so both
-- predicates are served by inverted indexes instead of parsing JSON per row.
SELECT service, COUNT(*) AS error_span_count
FROM spans
WHERE ts_start >= NOW() - INTERVAL 30 DAY
  AND (http_status_code >= 500 OR duration_ms > 5000)
GROUP BY service
ORDER BY error_span_count DESC
LIMIT 20;
