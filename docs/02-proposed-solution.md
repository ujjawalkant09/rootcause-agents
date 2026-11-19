# 02 – Proposed Solution & Technology Decisions

> **Key decisions:** Python · OpenAI models · LangGraph · cheaper-model variant · CLI + Markdown reports (no UI). See [ADR log](10-open-questions.md).

## 1. Stack

| Concern | Choice | Why |
|---|---|---|
| Language | **Python 3.12** | Confirmed |
| LLM provider | **OpenAI** via `langchain-openai` (`ChatOpenAI`) | Confirmed (you have the key). It plugs straight into LangGraph |
| Main model | **`gpt-6.2-sol`** ($2 / $10 per 1M tokens) | OpenAI's "balance of capability and cost" flagship. It's set via `.env`, so switching to `gpt-6.2-astra` is one line |
| Cheaper model (ablation) | **`gpt-6-luna`** ($0.10 / $0.50) for worker agents | Confirmed cheaper-model test. Used for the Observability and Dependency agents in one evaluation variant |
| Orchestration | **LangGraph** (`StateGraph`) | Confirmed. It provides typed shared state, parallel fan-out, conditional edges, per-node retry policies and recursion limits |
| Schemas | **Pydantic v2** | Agent contracts, tool I/O and structured LLM outputs (`.with_structured_output(Model)`) |
| Tools | LangChain `@tool` functions with Pydantic args | Structured inputs and outputs, and the same registry for both systems |
| Simulation data | YAML scenarios plus a seeded generator | Deterministic and readable |
| Interface | Streamlit web UI | Simple, testable, no separate frontend build |
| Tests | `pytest` + a **fake chat model** (scripted responses) | Offline, free and deterministic |
| Packaging | `uv` + `pyproject.toml` + `Makefile` | One-command setup |

> Model IDs come from OpenAI's model page (checked 2026-11-19). Your key must have access to them. If it doesn't, just change `MAIN_MODEL` and `WORKER_MODEL` in `.env`.

## 2. How LangGraph maps to the design

| Requirement | LangGraph feature |
|---|---|
| Shared investigation state | `StateGraph(InvestigationState)`, where `InvestigationState` is a TypedDict with **reducers** |
| Agents write evidence in parallel without races | `evidence: Annotated[list[Evidence], operator.add]` merges the outputs of parallel branches |
| Parallel execution | **`Send` API**: the Orchestrator's router returns one `Send("observability", task)` / `Send("dependency", task)` per planned task, and LangGraph runs them concurrently |
| Task dependencies | Graph edges (gather → reason → critic → …), plus `depends_on` inside the plan for follow-up tasks |
| Feedback loop | A **conditional edge** after the Critic: `NEED_MORE_EVIDENCE` → back to the Orchestrator (re-plan), `ACCEPT`/`INCONCLUSIVE` → Reporter |
| Bounded retries | `add_node(..., retry_policy=RetryPolicy(max_attempts=3, ...))` per node, plus our own tool-level retries |
| Execution limits / termination | `recursion_limit` in the run config as a hard stop, plus our own `Budget` (rounds, tool calls, tokens, time) checked in the routers |
| Recovery from failures | Nodes catch errors and write `failures`/`data_gaps` into the state instead of crashing. The graph continues |
| Observability | A per-node trace written by our callback handler to JSONL (tokens from `AIMessage.usage_metadata`) |

## 3. Core design ideas (unchanged from the original proposal)

1. **Evidence ledger.** Every tool result becomes an `Evidence` record with a code-assigned ID. Hypotheses must cite IDs, and a code validator rejects unknown ones. This separates facts from assumptions and makes grounding measurable.
2. **Critic loop with structured evidence requests.** `ACCEPT | NEED_MORE_EVIDENCE | INCONCLUSIVE`. Requests become new targeted tasks, with at most 3 rounds.
3. **Competing hypotheses.** At least 2, each with supporting **and** contradicting evidence, which counters "a deploy happened, so the deploy did it".
4. **Smart tools.** Logs come back as error clusters and metrics as change points or anomaly windows, never raw dumps. This is the biggest saving on tokens and noise.
5. **Inconclusive is allowed** and is expected in some test cases.

## 4. Optimisation levers

| Lever | How |
|---|---|
| Parallelism | `Send` fan-out across specialists. Agents may emit several tool calls per turn, which run concurrently |
| Pre-aggregating tools | Bounded, summarised outputs |
| Prompt caching | OpenAI caches long identical prompt prefixes automatically, so the system prompts and tool schemas are kept stable and placed first |
| Structured outputs | `ChatOpenAI(...).with_structured_output(PydanticModel)` gives no parse-retry loops |
| Context isolation | Each specialist gets only its task and the relevant evidence summaries |
| Early stop | The Critic accepts plus confidence thresholds, and a "no new evidence" rule |
| Hard budgets | Rounds, tool calls, tokens and wall-clock, all enforced in the routers |
| Cheaper workers | `gpt-6-luna` for the Observability and Dependency agents, measured as an ablation |

## 5. MVP scope

**Must have** (covers every core requirement in doc 01):
- 6 agents in LangGraph, 6 tools with fault injection, evidence ledger, Critic loop, budgets and recovery
- 10 test incidents (S01–S10 from doc 07), with S12 added if time allows
- Single-agent baseline on the same tools
- Evaluation: 10 incidents × 2 systems × **1 run**, plus a cheaper-worker ablation on the same 10
- Offline pytest suite, README, architecture doc and technical note

**Cut or simplified for time:**
- 3 repeated runs per incident → 1 run (noted as a limitation)
- LLM-judge grounding score → replaced by code-only grounding metrics (citation validity + key-evidence recall)
- Optional HTML viewer → dropped

## 6. Honest trade-off framing
Research shows multi-agent systems often use **4–15× more tokens** and don't always beat a strong single agent at equal budget. The evaluation reports cost per correct diagnosis and shows *where* multi-agent helps (misleading or conflicting evidence) and where it doesn't (easy cases).
