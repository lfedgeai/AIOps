SELECT
  COUNT(CASE WHEN duration_ms < 500 THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0) AS pct_under_500ms,
  COUNT(*) AS total_spans
FROM "telemetry_spans"
WHERE __time >= CURRENT_TIMESTAMP - INTERVAL '30' DAY
