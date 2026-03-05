USE telemetry;
SELECT
  SUM(CASE WHEN duration_ms < 500 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0) AS pct_under_500ms,
  COUNT(*) AS total_spans
FROM spans
WHERE ts_start >= NOW() - INTERVAL 30 DAY;
