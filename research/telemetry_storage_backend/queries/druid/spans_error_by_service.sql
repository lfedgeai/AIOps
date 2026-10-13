SELECT service, COUNT(*) AS error_span_count
FROM "telemetry_spans"
WHERE __time >= CURRENT_TIMESTAMP - INTERVAL '30' DAY
  AND (duration_ms > 5000)
GROUP BY service
ORDER BY error_span_count DESC
LIMIT 20
