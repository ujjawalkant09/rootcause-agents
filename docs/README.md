# RootCause Agents – Design Documentation

> **RootCause Agents** is a multi-agent AI system that investigates production incidents, identifies probable root causes and recommends evidence-backed corrective actions. It works over logs, metrics, deployment history, configuration changes, service dependencies and past incidents, and it is built to cope with incomplete, contradictory or misleading data.
>
> Status: implemented. 100+ offline tests pass and lint is clean. Live runs need `OPENAI_API_KEY` (see the root [README](../README.md)).

| # | Document | What it covers |
|---|----------|----------------|
| 01 | [Requirements & traceability](01-requirements-analysis.md) | Functional and non-functional requirements, mapped to design elements |
| 02 | [Solution & tech decisions](02-proposed-solution.md) | Stack, framework choice, model strategy, optimisations, alternatives considered |
| 03 | [Architecture](03-architecture.md) | Diagrams, shared state, investigation lifecycle, agent coordination |
| 04 | [Agents & data contracts](04-agents-and-contracts.md) | Each agent's role, inputs, outputs, Pydantic schemas and prompting strategy |
| 05 | [Tools & simulated environment](05-tools-and-simulation.md) | The 6 tools, their I/O schemas, fault injection and the synthetic data generator |
| 06 | [Reliability & control](06-reliability-and-control.md) | Retries, budgets, termination, failure recovery, the inconclusive outcome |
| 07 | [Dataset & evaluation](07-dataset-and-evaluation.md) | Test incidents, the single-agent baseline, metrics and the grading method |
| 08 | [Project structure & roadmap](08-project-structure-and-roadmap.md) | Repo layout, setup, roadmap |
| 09 | [Technical note](09-technical-note-draft.md) | Key decisions, trade-offs, limitations, improvements (≤500 words) |
| 10 | [Design decisions (ADR log)](10-open-questions.md) | The decisions taken and why |
| — | [References](references.md) | The research the design is based on |

## TL;DR

1. **Python + LangGraph + OpenAI** (`gpt-6.2-sol` as the main model, `gpt-6-luna` as the cheaper worker model). One shared typed state with reducers, `Send` fan-out for parallel agents, and a conditional Critic loop.
2. **Evidence ledger as the backbone.** Every tool result becomes an `Evidence` record with a code-assigned ID. Hypotheses must cite IDs, and claims without evidence are *assumptions*. Grounding is checked by code.
3. **A real feedback loop.** The Critic returns `ACCEPT | NEED_MORE_EVIDENCE | INCONCLUSIVE` with machine-readable evidence requests, which become new tasks (at most 3 rounds).
4. **Optimised for cost and latency.** Tools pre-aggregate data so the LLM never reads raw dumps. Agents and tool calls run in parallel. Tool calls are memoised. All outputs are structured.
5. **Deterministic, seeded simulation** with fault injection. Ground truth lives only in the grader, never in agent code.
6. **Evaluation harness** that compares the multi-agent system, a cheaper-worker variant and a single-agent baseline on the same tools and budget.
