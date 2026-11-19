# CLAUDE.md – RootCause Agents (rootcause-agents)

Multi-agent **production incident investigation** system. It receives an incident report, investigates simulated telemetry (logs, metrics, deployments, config changes, dependencies, incident history) with six cooperating agents, and produces an evidence-backed report with ranked hypotheses, or an honest INCONCLUSIVE. Design docs are in `docs/` (start at `docs/README.md`).

## Decisions (fixed, do not re-litigate)
- **Python ≥3.11**, managed with **uv**. Package lives in `src/rootcause_agents/` (src layout). Tests are in `tests/`, data in `data/`, evaluation in `eval/`.
- **LangGraph** (`StateGraph`, `Send` fan-out, conditional Critic loop) for orchestration.
- **OpenAI** via `langchain-openai`. Models are set in `.env`: `MAIN_MODEL` (default `gpt-6.2-sol`) and `WORKER_MODEL` (default `gpt-6-luna`, used by the `mas-cheap` system).
- Keep scope tight. Roadmap items live in `docs/10-open-questions.md` (ADR log).

## Workflow rules (MUST follow)
1. **TDD for every feature**: write the test first → run it and **see it fail** → implement → run until green → refactor. Never write implementation before its failing test.
2. All tests must run **offline** (no API key). Use `rootcause_agents.llm.ScriptedLLM` for agent and graph tests. Real-API tests go under `@pytest.mark.live` (excluded by default).
3. **No hardcoded answers**: agent code and prompts must never reference scenario IDs, ground truth or scenario-specific strings. `tests/test_no_hardcoding.py` enforces this. Ground truth is read only by `sim/` (generator) and `eval/grader.py`.
4. LLM-facing Pydantic schemas must stay **OpenAI-structured-output friendly**: no free-form `dict` fields. Use lists of typed items or JSON strings.
5. Tools **never raise** to agents. They always return a `ToolResult` (status ok/partial/stale/empty/error).
6. Keep the docs in `docs/` in sync when the design changes.

## Commands
```bash
uv sync                                   # install
uv run pytest -q                          # offline test suite
uv run pytest -m live                     # live API smoke tests (needs OPENAI_API_KEY)
uv run rootcause-agents list                   # list scenarios
uv run rootcause-agents investigate S03 --system mas        # mas | mas-cheap | single
uv run rootcause-agents eval --systems mas,single,mas-cheap --max-cost 2 # → eval/results/summary.md
uv run rootcause-agents ui                                 # Streamlit UI (or double-click start-ui.command)
docker compose up -d                                       # local Postgres for persistent sessions (port 5433)
uv run ruff check src tests
```

## Architecture map (where things live)
| Path | Responsibility |
|---|---|
| `src/rootcause_agents/config.py` | Settings from env: models per role, budgets, pricing, paths |
| `src/rootcause_agents/schemas.py` | All Pydantic contracts (Evidence, Hypothesis, CriticVerdict, ToolResult, InvestigationReport, …) |
| `src/rootcause_agents/sim/` | `scenario.py` (YAML models), `generator.py` (scenario → `TelemetryStore`), `store.py` |
| `src/rootcause_agents/tools/` | `analytics.py` (log clustering, change points), `faults.py` (FaultInjector), `impl.py` (6 tools), `executor.py` (retry, memoisation, evidence IDs, budget) |
| `src/rootcause_agents/llm.py` | `LLM` protocol, `OpenAILLM` (fallback chain, circuit breaker, `prompt_cache_key`, cached-token usage), `ScriptedLLM` (tests) |
| `src/rootcause_agents/context.py` | `RunContext`: store, executor, llm, budget, tracer for one investigation |
| `src/rootcause_agents/budget.py` | Budget limits, termination and outcome decisions |
| `src/rootcause_agents/validation.py` | Citation validator (facts must cite existing evidence) |
| `src/rootcause_agents/agents/` | `tool_loop.py` (bounded tool loop), orchestrator, specialists (observability/dependency/history), reasoning, critic, reporter, `single_agent.py` (baseline) |
| `src/rootcause_agents/prompts/*.md` | System prompts (generic, no scenario knowledge) |
| `src/rootcause_agents/graph.py` | LangGraph wiring for the multi-agent system |
| `src/rootcause_agents/runner.py` | `investigate(scenario, system)` entry point → report + trace files |
| `src/rootcause_agents/report.py` | Report → Markdown |
| `src/rootcause_agents/tracing.py` | JSONL tracer + usage accounting |
| `src/rootcause_agents/cli.py` | Typer CLI (`list`, `investigate`, `eval --max-cost`, `ui`) |
| `src/rootcause_agents/telemetry.py` | Optional Langfuse tracing: `NoopTelemetry` / `LangfuseTelemetry`, `observe()` typed observations, scores |
| `src/rootcause_agents/persistence.py` | LangGraph checkpointer (Postgres via `DATABASE_URL`, else in-memory), checkpoint serde allowlist, session registry |
| `src/rootcause_agents/ui/` | Streamlit UI: `app.py` (page), `view.py` (pure, tested helpers) |
| `eval/grader.py`, `eval/run_eval.py` | Metrics + comparison of MAS vs MAS-cheap vs single-agent |
| `data/topology.yaml`, `data/history/incidents.yaml`, `data/scenarios/*.yaml` | Synthetic environment + test incidents |

## Key design invariants
- **Human-in-the-loop**: `human_review` node uses `interrupt()`; no side effects before it (the node re-runs on resume). Human hints are leads, never evidence. `resume_session` must rebuild everything from the checkpoint (`meta` + `snapshot` in state). Max 2 questions per run; `investigate()`/eval never pause.
- **Observability never breaks a run**: all Langfuse calls go through `LangfuseTelemetry._safe`. Callbacks and the agent name are bound on the chat model (not the outer runnable) so each LLM call is one generation without LangChain noise. Observation names are stable (no IDs or rounds in names; those go in metadata).
- **Per-task tool quota** (`Budget.max_tool_calls_per_task`) keeps budget for critic-driven follow-up rounds.
- **Prompt-cache friendly prompts**: system message = `agents.common.system_prompt(ctx, role)` (instructions + static environment only). Never put incident-specific or time-varying content in it; `tests/test_prompt_cache.py` enforces this.
- **Token diet**: tool results go to the model through `tools.executor.compact_data`. The full data stays in the ledger.
- **Evidence ledger**: every successful tool call → `Evidence` with a code-assigned ID (`E-<n>`). Hypotheses and claims cite IDs, and `validation.py` rejects unknown IDs. Claims without evidence must be `kind="assumption"`.
- **The outcome is decided by code** (`budget.decide_outcome`), not by the LLM: ACCEPT + top ≥0.75 + margin ≥0.2 → ROOT_CAUSE_IDENTIFIED; ACCEPT otherwise → PROBABLE; INCONCLUSIVE verdict, or limits reached with top <0.5 → INCONCLUSIVE.
- **Critic loop**: NEED_MORE_EVIDENCE loops back to the orchestrator only if round < max_rounds, the budget is OK and the last round added new evidence.
- **Failures become data**: tool failures → `data_gaps`. A failed agent node → retried, then degraded with a gap recorded; the run continues.
- **Time** in scenarios is in minutes relative to `reported_at` (negative = before the report).

## Scenario authoring (data/scenarios/*.yaml)
Fields: `id`, `incident` (title, description, reported_at, affected_services, symptoms), `ground_truth` (root_cause_component, root_cause_category, aliases, expected_outcomes, key_signals), `signals` (metric_shift / log_burst), `deployments`, `config_changes`, `faults`, `seed`. Red herrings are ordinary signals/deployments that are not causal. See an existing scenario for the format.
