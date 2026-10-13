SELECT metric_name, quantile(0.95)(value) AS p95 FROM telemetry.metrics WHERE ts>=now()-INTERVAL 365 DAY GROUP BY metric_name ORDER BY p95 DESC LIMIT 20
