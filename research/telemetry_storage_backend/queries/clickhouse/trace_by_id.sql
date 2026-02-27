SELECT trace_id, span_id, service, name, ts_start, duration_ms FROM telemetry.spans WHERE trace_id=(SELECT trace_id FROM telemetry.spans LIMIT 1) ORDER BY ts_start
