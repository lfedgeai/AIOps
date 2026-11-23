"""Siloed correlation in Python. correlate_go/main.go uses the same algorithm.

Input: the raw response bodies, by name. Output: the canonical task answer.
"""
import base64, json
MIN_MS = 60_000


def _res(attrs, key):
    for a in attrs:
        if a["key"] == key: return a["value"].get("stringValue", "")
    return ""


def _hex(b64): return base64.b64decode(b64).hex()


def _loki(raw):
    for body in raw:
        for s in json.loads(body)["data"]["result"]:
            for ts, line in s["values"]: yield ts, line, s["stream"]


def worst_service(raw): return json.loads(raw["thanos"][0])["data"]["result"][0]["metric"]["service_name"]
def search_ids(raw): return [t["traceID"].rjust(32, "0") for t in json.loads(raw["tempo"][0]).get("traces") or []]


def search_services(raw):
    return {_res(s["attributes"], "service.name") for t in json.loads(raw["tempo"][0]).get("traces") or [] for ss in t["spanSets"] for s in ss["spans"]}


def trace_scope(raw):
    """Services and time range of one trace, to limit the Loki query."""
    svcs, t = set(), []
    for rs in json.loads(raw["tempo"][0])["trace"]["resourceSpans"]:
        svcs.add(_res(rs["resource"]["attributes"], "service.name"))
        for ss in rs["scopeSpans"]:
            for s in ss["spans"]: t += [int(s["startTimeUnixNano"]), int(s["endTimeUnixNano"])]
    return svcs, min(t), max(t)


def trace_context(raw, f0_ms):
    spans = []
    for rs in json.loads(raw["tempo"][0])["trace"]["resourceSpans"]:
        svc = _res(rs["resource"]["attributes"], "service.name")
        for ss in rs["scopeSpans"]:
            for s in ss["spans"]: spans.append([_hex(s["spanId"]), svc])
    logs = [[l["span_id"], l["service_name"], l["severity_text"], line] for _, line, l in _loki(raw["loki"])]
    return {"spans": sorted(spans), "logs": sorted(logs)}


def service_window(raw, f0_ms):
    out = {}
    # Every backend labels a bucket with its end.
    def cell(svc, ts_ms): return out.setdefault(f"{svc}|{(ts_ms - f0_ms) // MIN_MS - 1}", [0, 0, 0, 0.0])
    for s in json.loads(raw["tempo"][0])["series"]:
        lab = {l["key"]: l["value"].get("stringValue", "") for l in s["labels"]}
        for p in s["samples"]:
            ts, v = int(p["timestampMs"]), int(p.get("value", 0))
            if not v or ts <= f0_ms: continue
            c = cell(lab["resource.service.name"], ts)
            c[0] += v
            if lab["status"] == "error": c[1] += v
    for s in json.loads(raw["loki"][0])["data"]["result"]:
        for ts, v in s["values"]: cell(s["metric"]["service_name"], int(ts) * 1000)[2] += int(v)
    for s in json.loads(raw["thanos"][0])["data"]["result"]:
        for ts, v in s["values"]: cell(s["metric"]["service_name"], int(ts) * 1000)[3] = float(v)
    return out


def error_drilldown(raw, f0_ms):
    counts = dict.fromkeys(search_ids(raw), 0)
    for _, _, l in _loki(raw["loki"]):
        if l["trace_id"] in counts: counts[l["trace_id"]] += 1
    return {"service": worst_service(raw), "traces": counts}


def bulk_join(raw, f0_ms):
    by_trace = {}
    seen = set()
    for ts, line, l in _loki(raw["loki"]):
        k = (ts, l["span_id"], line)
        if k in seen: continue  # pages overlap by one timestamp
        seen.add(k)
        by_trace.setdefault(l["trace_id"], []).append(l["service_name"])
    out = {}
    for t in json.loads(raw["tempo"][0]).get("traces") or []:
        logs = by_trace.get(t["traceID"].rjust(32, "0"), [])
        for ss in t["spanSets"]:
            for s in ss["spans"]:
                svc = _res(s["attributes"], "service.name")
                for lsvc in logs:
                    k = f"{svc}|{lsvc}"
                    out[k] = out.get(k, 0) + 1
    return out
