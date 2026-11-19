# 10 – Design Decisions (ADR log)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| ADR-1 | Interface | CLI + Markdown/JSON reports | The value is in the investigation engine. A UI is optional and out of scope |
| ADR-2 | Language | Python ≥3.11 | Best ecosystem for LLM apps (Pydantic, asyncio, LangGraph, pytest) |
| ADR-3 | LLM provider | OpenAI via `langchain-openai` | Structured outputs, parallel tool calls and token usage on every response. Models are configurable via `.env` |
| ADR-4 | Orchestration | LangGraph `StateGraph` | Typed shared state with reducers, `Send` fan-out, conditional edges, recursion limits |
| ADR-5 | Model tiering | `mas-cheap` variant with a cheaper worker model for the data-gathering agents | Measures the cost/accuracy trade-off instead of assuming it |
| ADR-6 | MVP scope | 10 scenarios, 1 run each, code-only grounding metrics | Fast, reproducible evaluation. Repeats and an LLM-judge are roadmap items |
| ADR-7 | Outcome decision | Deterministic policy in code, not by the LLM | Prevents over-confident conclusions and makes outcomes auditable |
| ADR-8 | Testing | Test-driven development with a scripted LLM | The whole pipeline is testable offline and for free |
| ADR-9 | LLM resilience | Fallback chain (models → second provider) + circuit breaker, implemented in `llm.py` | Provider outages and rate limits must degrade cost or quality, not fail the investigation |
| ADR-10 | Cost control | Prompt-cache-friendly prefixes + `prompt_cache_key`, cached-token pricing, compact tool payloads | Cached input costs 5% of the normal price, and compact payloads cut tool-result size by 49% |

## Roadmap items
- An optional on-disk LLM response cache for development reruns (accounting must exclude cache hits).
- Prompt deltas in later rounds (send only new evidence + current hypotheses).
- Repeated evaluation runs and an LLM-judge for semantic grounding.
