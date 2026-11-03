# Dev notes: what the raw Tier 3 observation makes Jev work out (2026-11-02)

These notes are for `openspec/changes/archive/2026-11-02-observation-facts` task 2.1, written before any `facts` code. They use only the dev set, `results/tier3/10.3-combined` (40 runs, 351 polls, `jev-1.13.0`). We did not read the confirm set (`results/tier4/6.2-combined`) for these notes.

## Where Jev goes wrong on the dev set

| Scenario | True component | Jev's scored answer (3 runs) |
|---|---|---|
| K001 cart scaled to zero | cart | frontend, cart, frontend |
| K002 cart pods deleted once (recovered; no incident) | cart | frontend ×3, all reported as incidents (p 0.81-0.88 on the first detecting poll) |
| K007 valkey-cart scaled to zero | valkey-cart | frontend-proxy, frontend-proxy, valkey-cart |
| K008 cart to zero under load | cart | frontend ×3 |

## What the raw layout asks Jev to do

1. **Join sections.** A deployment's readiness (`deployments`: `desired 0, ready 0, baseline_available 1`) sits in a different list from the same service's request metrics (`services`). Its callers' errors are in a third (`dependencies`, `failing_traces`). To name `cart` and not `frontend`, Jev has to connect "frontend's calls fail" with "frontend's callee cart has no ready replicas" across three lists. Jev's docs say this is a weak spot: "multiple hops of reasoning costs accuracy".
2. **Compare numbers.** Each service row gives the current and baseline values side by side (`error_rate`, `baseline_error_rate`, `latency_p95_ms`, `baseline_latency_p95_ms`), plus `error_rate_change` and `latency_p95_change_ms`. Ratios are not given. The docs say "Jev is not a calculator".
3. **Order times.** The window and baseline are ISO timestamps. Error logs carry their own `ts`, and events carry `last_seen`. To tell whether a failure is still happening, Jev must order these strings against the window end. The docs say jev-1.13 "reads dates as text, not as ordered quantities".

## What the data cannot support

- **Early polls show no metric change.** On 52 of 351 dev polls, every service's current metrics equal its baseline metrics. The 180 s window then lies mostly before the injection, and span metrics arrive every 60 s. A layout cannot add a change that is not in the data. It can only state that the values are equal.
- **Some edges are missing.** `valkey-cart` emits no spans, so `dependencies` has no `cart -> valkey-cart` edge. A layout that added the edge would add knowledge of the topology, which is new information. `facts` therefore puts a deployment's readiness next to a service row only when the names match, and lists callees only from the `dependencies` edges that are present.
- **Restarts have no time.** Pod rows give `restarts_in_window` and `new_since_baseline`, but no timestamp. "How long ago" can be computed only for error logs and events.

## What `facts` will do (and only this)

- **On every deployment row:** "ready R of D (baseline available A of B)". Also "desired replicas D, baseline B" when they differ, and "rollout X" when the rollout is not complete.
- **On every service row:**
  - error rate and p95 latency against the baseline, as plain statements, with a ratio only when the baseline is above zero;
  - request rate against the baseline;
  - the same-named deployment's readiness statement, if there is one;
  - "same as baseline" where the values are equal.
- **On every service row, a `callees` list** built from `dependencies` edges, giving each callee's own call and error counts, readiness and error statements.
- **For time:** `seconds_before_window_end` replaces `ts` and `last_seen`. The window is shown as `length_s`, and the baseline as `length_s` plus `ends_s_before_window_end`.

Every service gets the same treatment, row order is unchanged, and no statement compares one service with another or names a cause.
