SELECT service, count() AS slow_cnt FROM telemetry.spans WHERE ts_start>=now()-INTERVAL 1 DAY AND duration_ms>500 GROUP BY service ORDER BY slow_cnt DESC LIMIT 20
