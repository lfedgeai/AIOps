// Siloed correlation in Go. Same algorithm as correlate.py: full JSON decode, hash join.
// Usage: correlate <task> <dir> <f0_ms> <iterations>. Prints {"result":..., "ns":[...]}.
// Files are read before the timer starts; the timer covers decode and join only.
package main

import (
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

type attr struct {
	Key   string `json:"key"`
	Value struct {
		S string `json:"stringValue"`
	} `json:"value"`
}
type lokiResp struct {
	Data struct {
		Result []struct {
			Stream map[string]string   `json:"stream"`
			Metric map[string]string   `json:"metric"`
			Values [][]json.RawMessage `json:"values"`
		} `json:"result"`
	} `json:"data"`
}
type promResp struct {
	Data struct {
		Result []struct {
			Metric map[string]string   `json:"metric"`
			Values [][]json.RawMessage `json:"values"`
		} `json:"result"`
	} `json:"data"`
}
type searchResp struct {
	Traces []struct {
		TraceID  string `json:"traceID"`
		SpanSets []struct {
			Spans []struct {
				Attributes []attr `json:"attributes"`
			} `json:"spans"`
		} `json:"spanSets"`
	} `json:"traces"`
}
type input struct{ tempo, loki, thanos [][]byte }

func must(err error) {
	if err != nil {
		panic(err)
	}
}
func dec(b []byte, v any) { must(json.Unmarshal(b, v)) }
func get(as []attr, k string) string {
	for _, a := range as {
		if a.Key == k {
			return a.Value.S
		}
	}
	return ""
}
func pad(s string) string {
	if len(s) < 32 {
		return strings.Repeat("0", 32-len(s)) + s
	}
	return s
}
func str(r json.RawMessage) string { var s string; dec(r, &s); return s }

func eachLog(pages [][]byte, f func(ts, line string, l map[string]string)) {
	for _, b := range pages {
		var r lokiResp
		dec(b, &r)
		for _, s := range r.Data.Result {
			for _, v := range s.Values {
				f(str(v[0]), str(v[1]), s.Stream)
			}
		}
	}
}

func traceContext(in input) any {
	var t struct {
		Trace struct {
			ResourceSpans []struct {
				Resource struct {
					Attributes []attr `json:"attributes"`
				} `json:"resource"`
				ScopeSpans []struct {
					Spans []struct {
						SpanID string `json:"spanId"`
					} `json:"spans"`
				} `json:"scopeSpans"`
			} `json:"resourceSpans"`
		} `json:"trace"`
	}
	dec(in.tempo[0], &t)
	spans, logs := [][]string{}, [][]string{}
	for _, rs := range t.Trace.ResourceSpans {
		svc := get(rs.Resource.Attributes, "service.name")
		for _, ss := range rs.ScopeSpans {
			for _, s := range ss.Spans {
				b, err := base64.StdEncoding.DecodeString(s.SpanID)
				must(err)
				spans = append(spans, []string{hex.EncodeToString(b), svc})
			}
		}
	}
	eachLog(in.loki, func(_, line string, l map[string]string) {
		logs = append(logs, []string{l["span_id"], l["service_name"], l["severity_text"], line})
	})
	for _, x := range [][][]string{spans, logs} {
		sort.Slice(x, func(i, j int) bool {
			for k := range x[i] {
				if x[i][k] != x[j][k] {
					return x[i][k] < x[j][k]
				}
			}
			return false
		})
	}
	return map[string]any{"spans": spans, "logs": logs}
}

func serviceWindow(in input, f0 int64) any {
	out := map[string]*[4]float64{}
	cell := func(svc string, ts int64) *[4]float64 {
		k := svc + "|" + strconv.FormatInt((ts-f0)/60000-1, 10)
		if out[k] == nil {
			out[k] = &[4]float64{}
		}
		return out[k]
	}
	var t struct {
		Series []struct {
			Labels  []attr `json:"labels"`
			Samples []struct {
				Ts string  `json:"timestampMs"`
				V  float64 `json:"value"`
			} `json:"samples"`
		} `json:"series"`
	}
	dec(in.tempo[0], &t)
	for _, s := range t.Series {
		svc, st := get(s.Labels, "resource.service.name"), get(s.Labels, "status")
		for _, p := range s.Samples {
			ts, err := strconv.ParseInt(p.Ts, 10, 64)
			must(err)
			if p.V == 0 || ts <= f0 {
				continue
			}
			c := cell(svc, ts)
			c[0] += p.V
			if st == "error" {
				c[1] += p.V
			}
		}
	}
	for i, b := range [][]byte{in.loki[0], in.thanos[0]} {
		var r promResp
		dec(b, &r)
		for _, s := range r.Data.Result {
			for _, v := range s.Values {
				var ts int64
				dec(v[0], &ts)
				f, err := strconv.ParseFloat(str(v[1]), 64)
				must(err)
				c := cell(s.Metric["service_name"], ts*1000)
				if i == 0 {
					c[2] += f
				} else {
					c[3] = f
				}
			}
		}
	}
	return out
}

func errorDrilldown(in input) any {
	var p promResp
	var s searchResp
	dec(in.thanos[0], &p)
	dec(in.tempo[0], &s)
	counts := map[string]int{}
	for _, t := range s.Traces {
		counts[pad(t.TraceID)] = 0
	}
	eachLog(in.loki, func(_, _ string, l map[string]string) {
		if _, ok := counts[l["trace_id"]]; ok {
			counts[l["trace_id"]]++
		}
	})
	return map[string]any{"service": p.Data.Result[0].Metric["service_name"], "traces": counts}
}

func bulkJoin(in input) any {
	byTrace := map[string][]string{}
	type key struct{ ts, sid, line string }
	seen := map[key]bool{}
	eachLog(in.loki, func(ts, line string, l map[string]string) {
		k := key{ts, l["span_id"], line}
		if seen[k] {
			return
		}
		seen[k] = true
		byTrace[l["trace_id"]] = append(byTrace[l["trace_id"]], l["service_name"])
	})
	var s searchResp
	dec(in.tempo[0], &s)
	out := map[string]int{}
	for _, t := range s.Traces {
		logs := byTrace[pad(t.TraceID)]
		for _, ss := range t.SpanSets {
			for _, sp := range ss.Spans {
				svc := get(sp.Attributes, "service.name")
				for _, l := range logs {
					out[svc+"|"+l]++
				}
			}
		}
	}
	return out
}

func main() {
	task, dir := os.Args[1], os.Args[2]
	f0, err := strconv.ParseInt(os.Args[3], 10, 64)
	must(err)
	n, err := strconv.Atoi(os.Args[4])
	must(err)
	read := func(name string) (out [][]byte) {
		fs, _ := filepath.Glob(filepath.Join(dir, name+"_*.json"))
		sort.Strings(fs)
		for _, f := range fs {
			b, err := os.ReadFile(f)
			must(err)
			out = append(out, b)
		}
		return
	}
	in := input{read("tempo"), read("loki"), read("thanos")}
	var res any
	ns := []int64{}
	for i := 0; i < n; i++ {
		t := time.Now()
		switch task {
		case "trace_context":
			res = traceContext(in)
		case "service_window":
			res = serviceWindow(in, f0)
		case "error_drilldown":
			res = errorDrilldown(in)
		case "bulk_join":
			res = bulkJoin(in)
		}
		ns = append(ns, time.Since(t).Nanoseconds())
	}
	b, err := json.Marshal(map[string]any{"result": res, "ns": ns})
	must(err)
	fmt.Println(string(b))
}
