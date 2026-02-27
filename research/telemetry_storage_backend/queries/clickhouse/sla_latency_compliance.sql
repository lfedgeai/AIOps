SELECT countIf(duration_ms<500)*100.0/nullIf(count(),0) AS pct_under_500ms, count() AS total_spans FROM telemetry.spans WHERE ts_start>=now()-INTERVAL 30 DAY
