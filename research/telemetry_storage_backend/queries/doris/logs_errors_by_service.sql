-- Error logs per service. level uses the idx_level inverted index; message uses the
-- idx_message full-text index (unicode parser, lower_case) instead of a LIKE '%error%' scan.
SELECT service, COUNT(*) AS err_count
FROM logs
WHERE ts >= NOW() - INTERVAL 1 DAY
  AND (level = 'error' OR message MATCH_ANY 'error')
GROUP BY service
ORDER BY err_count DESC
LIMIT 20;
