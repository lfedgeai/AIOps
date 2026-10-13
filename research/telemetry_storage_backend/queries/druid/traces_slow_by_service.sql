SELECT service, COUNT(*) AS slow_cnt
FROM "telemetry_spans"
WHERE __time >= CURRENT_TIMESTAMP - INTERVAL '1' DAY
  AND duration_ms > 500
GROUP BY service
ORDER BY slow_cnt DESC
LIMIT 20
