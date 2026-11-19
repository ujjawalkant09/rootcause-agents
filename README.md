# RootCause Agents 🔍 — Multi-Agent AI for Production Incident Root-Cause Analysis

[![CI](https://github.com/ujjawalkant09/rootcause-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/ujjawalkant09/rootcause-agents/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-purple)

**Highlights**
- 6 cooperating agents (Orchestrator, Observability, Dependency, Reasoning, Critic, Reporter) on a LangGraph state machine with parallel fan-out and a bounded critic feedback loop
- An evidence ledger where every claim cites a tool-result ID, enforced by code, so facts are separated from assumptions
- Fault-tolerant tooling: retries with backoff, memoisation, budgets, graceful degradation, and fault injection (timeouts, partial or stale data)
- LLM resilience and cost control: a model and provider fallback chain with a circuit breaker, OpenAI prompt caching with cached-token cost accounting, and compact tool payloads (49% smaller)
- An evaluation harness that compares the multi-agent system with a single-agent baseline on accuracy, grounding, tool calls, tokens, cost and latency
- Built test-first: 100+ offline tests using a scripted LLM, plus CI

A LangGraph + OpenAI multi-agent system that investigates production incidents in a simulated environment (logs, metrics, deployments, config changes, service dependencies, incident history), compares competing root-cause hypotheses, challenges them with a critic, and produces an **evidence-backed report**: ranked hypotheses, supporting and contradicting evidence, confidence, facts vs inferences vs assumptions, data gaps, and remediation. An honest **INCONCLUSIVE** is a valid outcome.

The design docs are in [`docs/`](docs/README.md) (architecture diagrams in [`docs/03-architecture.md`](docs/03-architecture.md)). Contributor and agent instructions are in [`CLAUDE.md`](CLAUDE.md).

## Quick start

```bash
# prerequisites: Python ≥3.11, uv (https://docs.astral.sh/uv/)
make setup                 # uv sync
cp .env.example .env       # then put your OPENAI_API_KEY in .env
make test                  # 100+ offline tests, no API key needed

uv run rootcause-agents list                              # the 10 synthetic incidents
uv run rootcause-agents investigate S03 --system mas      # live trace + report (runs/<id>/report.md)
uv run rootcause-agents investigate S03 --system single   # single-agent baseline
make eval                                            # MAS vs MAS-cheap vs single → eval/results/summary.md
```

Systems:

| System | Description |
|---|---|
| `mas` | Full multi-agent graph. All agents use `MAIN_MODEL` (default `gpt-6.2-sol`) |
| `mas-cheap` | Same graph, but the Observability and Dependency workers use `WORKER_MODEL` (default `gpt-6-luna`) |
| `single` | One ReAct agent with the **same 6 tools, budget, report schema and outcome policy** |

If your key can't access the default models, set `MAIN_MODEL` / `WORKER_MODEL` in `.env` (for example `gpt-6.1-terra`). For models without `json_schema` structured output, set `STRUCTURED_METHOD=function_calling`.

## How it works

```
START → Orchestrator ─Send()→ ┌ Observability (logs, metrics)        ┐
          ↑                   ├ Dependency (deploys, config, graph)   ├→ Reasoning → Critic ─┬→ Reporter → END
          │                   └ History (deterministic search)       ┘                      │
          └──────── NEED_MORE_EVIDENCE (targeted requests, ≤3 rounds, must add new evidence) ┘
```

- **Shared state (blackboard)**: a LangGraph `StateGraph` whose reducers merge an append-only **evidence ledger**, findings, data gaps and critic verdicts from parallel branches.
- **Evidence ledger**: every successful tool call becomes `Evidence` with a code-assigned ID (`E-n`). Hypotheses and claims must cite IDs. A validator rejects unknown IDs, re-prompts the agent (bounded), then sanitises the output. Uncited "facts" are demoted to assumptions.
- **Parallelism**: the `Send` API fans out the specialist tasks, and agents issue parallel tool calls in a single turn.
- **Feedback loop**: the Critic returns `ACCEPT | NEED_MORE_EVIDENCE | INCONCLUSIVE` with structured evidence requests, which the Orchestrator turns into new targeted tasks.
- **Outcome by policy, not by the LLM**: ACCEPT + top confidence ≥0.75 + margin ≥0.2 → `ROOT_CAUSE_IDENTIFIED`; otherwise `PROBABLE`/`INCONCLUSIVE` (`src/rootcause_agents/budget.py`).
- **Reliability**:
  - Tool retries with exponential backoff and jitter for retryable faults.
  - Repeated identical calls are memoised.
  - Budgets on rounds, tool calls, tool turns per agent, tokens and wall-clock time.
  - The loop stops early when a round adds no new evidence.
  - Node-level retry, then graceful degradation (a failed agent becomes a recorded data gap).
  - Deterministic fallbacks for the plan and the report.
  - `recursion_limit` as a final safety net.
- **LLM fallback chain** (`llm.py`): primary model → `FALLBACK_MODELS` → an optional second OpenAI-compatible provider (`FALLBACK_BASE_URL`).
  - It switches on provider errors, timeouts or unparseable structured output.
  - A circuit breaker skips an endpoint after repeated failures, then tries it again after a cooldown.
  - Every fallback is traced and counted in the report.
- **Prompt caching**:
  - Each agent's system message is a stable prefix (role instructions plus the static environment), and all incident-specific content comes after it.
  - Tool loops only ever append messages, so later turns reuse the cached prefix.
  - Each role sends a `prompt_cache_key`.
  - Cached tokens are read from the API response and priced at the cached rate (5% of the normal input price for `gpt-6.2-sol`).
- **Token diet**:
  - Tools return error clusters and change points, never raw logs.
  - The model receives a compact version of each tool result (no metric series, one sample line per log cluster). That is 49% fewer characters across all scenarios, while the full data stays in the evidence ledger.
  - Per-agent token and cost breakdown appears in every report and in the evaluation summary.
- **Smart tools**: logs come back as error-signature clusters and metrics as change-point onsets and magnitudes, never as raw dumps. This keeps tokens low and signal high.
- **Fault injection** (declared per scenario, seeded): timeouts, unavailable backends, rate limits, partial shards, stale metrics, empty results.

## Dataset & evaluation

`data/scenarios/S01–S10` cover: a control case, misleading coincident deploys, misleading history priors, conflicting observations, noisy chronic errors, tool failures (timeouts, partial, stale, unavailable) and an expected-inconclusive case. Ground truth is read **only** by the simulator and the grader. `tests/test_no_hardcoding.py` enforces that agent code and prompts never reference scenarios, and `tests/test_dataset.py` proves every scenario's key signals are discoverable through the tools.

Metrics (computed from reports and traces):
- top-1 and top-3 root-cause accuracy, outcome correctness, confident-wrong rate
- citation validity, claim coverage, key-evidence recall
- Brier score
- rounds, tool calls, LLM calls, tokens (including cache hit rate and per-agent breakdown), cost (USD, cached-token aware) and latency

Results go to `eval/results/summary.md`, plus per-run reports and traces under `eval/results/runs/`.

## Project layout

```
src/rootcause_agents/   config, schemas, llm, budget, validation, graph, runner, evaluation, cli
  agents/          orchestrator, specialists, reasoning, critic, reporter, single_agent, tool_loop
  tools/           impl (6 tools), executor (retry/memo/evidence/budget), faults, analytics
  sim/             scenario models, generator, store
  prompts/         system prompts (generic)
data/              topology, incident history, scenarios
tests/             offline pytest suite (scripted LLM); `-m live` for real API smoke tests
eval/              run_eval.py wrapper + results
docs/              design docs
```

## External dependencies
- **OpenAI API** (via `langchain-openai`) for live runs and evaluation. Needs `OPENAI_API_KEY` and network access.
- Python packages: `langgraph`, `langchain-openai`, `langchain-core`, `pydantic`, `numpy`, `pyyaml`, `typer`, `rich`, `python-dotenv`. Dev: `pytest`, `pytest-asyncio`, `ruff`.
- No databases, Docker or cloud services.

## Limitations / incomplete functionality
- The telemetry is synthetic, generated from declared causal signals, so it is cleaner than real production data.
- 10 scenarios × 1 run by default (`--repeats N` is supported), so results are indicative rather than statistically significant.
- Grounding is measured by code (citation validity, claim coverage, key-evidence recall). There is no LLM-judge "semantic support" score yet.
- Confidence comes from the LLM and is uncalibrated.
- There is no persistence or checkpoint-resume (a LangGraph checkpointer could be added).
- Pricing for cost estimates is hard-coded in `config.py` (checked 2026-11-19).
