# 05 – Tools & Simulated Environment

## 1. Tool result envelope (common to all tools)

```python
class ToolResult(BaseModel):
    tool: str
    status: Literal["ok", "partial", "stale", "empty", "error"]
    data: dict | list | None
    completeness: float            # 0..1
    truncated: bool = False
    next_cursor: str | None = None # pagination
    warnings: list[str] = []       # "data older than 10 min", "3/5 shards responded"
    error: ToolError | None = None # {code: TIMEOUT|UNAVAILABLE|RATE_LIMITED|BAD_INPUT, message, retryable}
    latency_ms: int
```

Rules:
- Tools **never raise** to the agent. Failures come back as a `ToolResult` with `status="error"` and are sent back to the model as a `ToolMessage` with the error payload, so the model can adapt.
- Inputs are validated with Pydantic, and bad input returns `BAD_INPUT` with a helpful message.
- Every successful call is written to the blackboard as an `Evidence` record. The ID is assigned by code.

## 2. Tool catalogue

| Tool | Input (key params) | Output (aggregated) |
|---|---|---|
| `query_logs` | `service`, `start`, `end`, `level?`, `pattern?`, `max_clusters=10` | Error-signature clusters: `{template, count, first_seen, last_seen, sample_lines[≤3], trace_ids[≤3]}` plus totals per level |
| `query_metrics` | `service`, `metric` (latency_p50/p99, error_rate, rps, cpu, mem, db_connections, queue_depth…), `start`, `end`, `step` | `{baseline_stats, window_stats, change_points[], anomaly_windows[], downsampled_series[≤30 points]}` |
| `get_deployments` | `service?`, `start`, `end` | `[ {id, service, version, author, time, status, rollback_of?, diff_summary, files_changed[]} ]` |
| `get_config_changes` | `service?`, `start`, `end` | `[ {id, service, key, old, new, time, actor, source(flag/env/infra)} ]` |
| `get_service_dependencies` | `service`, `direction=up|down|both`, `depth=2` | `{nodes[], edges[{from,to,type(sync/async/db/cache), criticality}]}` |
| `search_incident_history` | `query`, `services?`, `top_k=3` | `[ {incident_id, title, root_cause, resolution, similarity} ]` (BM25 or simple TF-IDF, with no external dependency) |

Analytics inside the tools are deterministic and cheap: log templating via simple masking of numbers, UUIDs and IPs, and change-point detection via rolling z-score or CUSUM.

## 3. Fault injection

`FaultInjector` wraps every tool. It is configured **per scenario** and seeded for reproducibility.

| Fault | Behaviour | What it tests |
|---|---|---|
| `timeout` | Sleeps past the tool timeout and returns `TIMEOUT` (retryable) | Retry + backoff, degraded mode |
| `flaky_n` | Fails the first *n* calls, then succeeds | Bounded retry success path |
| `unavailable` | Always fails (non-retryable after k attempts) | Recording gaps, inconclusive outcome |
| `partial` | Returns a subset of shards or time windows, `completeness < 1` | Confidence calibration, Critic re-query |
| `stale` | Returns data that lags by N minutes, with a warning | Critic challenge on temporal evidence |
| `truncated` | Paginates and requires a cursor | Agents following pagination |
| `rate_limited` | Returns `RATE_LIMITED` with `retry_after` | Respecting backoff |
| `empty` | Returns no data | Distinguishing "no errors" from "no data" |

Fault scope can target a tool, a service, a metric or the Nth call, for example: *"`query_metrics` for `payments-db` times out on the first 2 calls"*.

## 4. Simulated environment & data generator

```
data/
  topology.yaml            # ~12 services: api-gateway, checkout, payments, payments-db, inventory, cache, auth, notifications, queue, search, user-svc, cdn
  history/incidents.yaml   # ~25 historical incidents (some deliberately similar-but-different)
  scenarios/
    S01_bad_deploy.yaml
    ...
```

A **scenario file** declares the incident and the *injected ground truth*, which the generator synthesises into telemetry:

```yaml
id: S04
incident_report:
  title: "Checkout latency p99 > 5s, elevated 5xx"
  reported_at: "2026-03-10T14:20:00Z"
  symptoms: ["checkout p99 latency 5.2s", "5xx rate 8%"]
  affected_services: [checkout]
ground_truth:                     # visible ONLY to the generator + grader
  root_cause_component: payments-db
  root_cause_category: resource_exhaustion
  key_evidence: ["payments-db db_connections saturated before 14:03", "pool timeout errors in payments"]
  acceptable_aliases: ["payments database", "db connection pool"]
  expected_outcome: ROOT_CAUSE_IDENTIFIED
signals:                          # what the generator injects
  - {type: metric_shift, service: payments-db, metric: db_connections, at: "14:02", to: 100, pct_of_max: true}
  - {type: log_burst, service: payments, template: "ConnectionPoolTimeout after {ms}ms", at: "14:03", rate: 80/min}
  - {type: latency_propagation, path: [payments-db, payments, checkout], lag_s: 30}
red_herrings:
  - {type: deployment, service: checkout, at: "14:06", diff_summary: "update button copy"}
  - {type: log_burst, service: notifications, template: "SMTP retry", at: "13:00", rate: 5/min}  # chronic noise
faults:
  - {tool: query_metrics, match: {service: payments-db}, fault: timeout, times: 2}
noise: {seed: 42, baseline_error_rate: 0.002}
```

The generator writes deterministic telemetry stores (in memory, or Parquet/JSON files) that the tools query. **Agents only ever see tool outputs.** They never see scenario files, IDs or ground truth.

## 5. Why this design is defensible
- Telemetry is synthesised **from causal signals**, so the data is internally consistent and realistic (propagation lag, baseline noise, chronic background errors).
- Red herrings and faults are *declared*, which makes the adversarial traits of each test explicit and documentable.
- Adding a new incident is a short YAML file, and no code changes are needed.
