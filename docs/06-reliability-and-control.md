# 06 – Reliability, Budgets & Termination

## 1. Layered failure handling

| Layer | Failure | Handling |
|---|---|---|
| Tool | Timeout, 5xx, rate limit | `RetryPolicy(max_attempts=3, backoff=exp(0.2s)·2^n + jitter, honour retry_after)` applied only to `retryable` errors. Per-call timeout via `asyncio.wait_for` |
| Tool | Non-retryable / exhausted | Return an `error` ToolResult to the agent and record a **data gap** on the blackboard |
| Tool | Partial / stale | Pass through with `completeness` and warnings. The Evidence inherits a lower weight. The Critic is prompted to treat it cautiously |
| LLM call | 429 / 5xx / connection | `ChatOpenAI(max_retries=...)` built-in retries plus the LangGraph node `RetryPolicy` (max 3 attempts, exponential backoff + jitter) |
| LLM call | Schema-invalid output | Structured outputs make this rare. On a citation-validation failure, re-prompt with the validator error (at most 2 times) |
| LLM call | Refusal / truncated output | Detected by the structured-output parser. Treated as an agent failure (next row) |
| Agent task | Exception / timeout / repeated invalid output | Retry the task once, then mark it `DEGRADED` with its reason. Dependent tasks still run with whatever evidence exists |
| Orchestrator | Plan invalid (unknown agent, cycle) | Validate the DAG in code, re-prompt once, then fall back to a **default static plan** (all specialists on the affected services) |
| Whole run | Budget exhausted | Graceful stop: Reasoning runs with current evidence, the Reporting Agent is forced to produce output, and the outcome is likely `PROBABLE`/`INCONCLUSIVE` with the gap noted |

The principle: **failure becomes information, not an exception.** Every gap shows up in `report.data_gaps` and lowers confidence.

## 2. Budgets (enforced by code, visible to agents)

```python
class Budget(BaseModel):
    max_rounds: int = 3              # critic feedback loops
    max_tool_calls: int = 40
    max_tool_turns_per_agent: int = 6
    max_total_tokens: int = 400_000  # input+output across all agents
    max_wall_clock_s: int = 240
    max_reasoning_retries: int = 2
```

- The remaining budget is passed to the Orchestrator and Critic so they can prioritise.
- The scheduler checks the budget before each task. Specialists check it before each tool turn.
- The **same budget** applies to the single-agent baseline.

## 3. Termination conditions (checked after every Critic verdict)

Stop and report when **any** of these holds:
1. `decision == ACCEPT` and `top.confidence ≥ 0.75` and `top.confidence − second.confidence ≥ 0.2` → `ROOT_CAUSE_IDENTIFIED`
2. `decision == ACCEPT` but the thresholds are not met → `PROBABLE`
3. `decision == INCONCLUSIVE` → `INCONCLUSIVE`
4. `round == max_rounds` → `PROBABLE` if top ≥ 0.5, else `INCONCLUSIVE`
5. Any hard budget is exhausted → same mapping as 4, with a `budget_exhausted` flag
6. **No progress**: a round added no new evidence (every request failed or duplicated earlier queries) → stop early

Duplicate-query suppression: tool calls are memoised by `(tool, normalised params)` within a run, which avoids loops that repeatedly ask for the same thing.

## 4. Inconclusive outcome is a first-class result
The Reporting Agent's template has a dedicated path for `INCONCLUSIVE`:
- What is known (facts)
- The candidate hypotheses and why they can't be distinguished
- Which data is missing and why (tool failures)
- **Next investigative steps** for a human, such as specific queries or dashboards

Some test scenarios *expect* `INCONCLUSIVE` (for example, when the critical data source is down). A confident wrong answer counts as a failure in grading.

## 5. Concurrency details
- The LangGraph `Send` fan-out runs the specialist branches in parallel within one superstep. `max_concurrency` in the run config caps parallel LLM calls to respect rate limits.
- Inside an agent, multiple tool calls from one model turn run concurrently (`asyncio.gather`). All results go back together.
- Shared-state writes go through LangGraph **reducers** (`operator.add` on the evidence, findings and gaps lists), so parallel branches never overwrite each other. Evidence IDs (`E-n`) are assigned sequentially by the single shared `ToolExecutor` on the event loop, so they never collide; a custom `merge_evidence` reducer drops duplicates from memoised calls.
- `recursion_limit` (e.g. 40) is a final safety net on top of our own budget.

## 6. Observability of the system itself
Every event is written to `runs/<run_id>/trace.jsonl`:
`{ts, run_id, round, agent, event: llm_call|tool_call|tool_error|retry|state_update|verdict|terminate, latency_ms, input_tokens, output_tokens, cache_read_tokens, …}`

The evaluation harness computes all metrics from these traces, so the numbers are auditable.
