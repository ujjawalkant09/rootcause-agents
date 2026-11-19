# 01 – Requirements & Traceability

This document lists the system's requirements and maps each one to the part of the design that satisfies it. The matrix doubles as an acceptance checklist.

## 1. What makes this problem hard

Incident data is often incomplete, contradictory or misleading, so the system must *investigate* rather than generate a plausible-sounding answer, and simply chaining LLM calls with different prompts is not enough. The design therefore targets:

- **Genuine collaboration**: agents read and write a shared state and react to each other's outputs, including loops back.
- **Evidence discipline**: conclusions are traceable to tool outputs, facts are separated from assumptions, and an *inconclusive* outcome is allowed.
- **Engineering rigour**: bounded loops, retries, failure handling, parallelism and measurement.
- **Honest evaluation**: a fair baseline on the same tools, a scenario set that includes adversarial cases, and real numbers including where multi-agent loses.

## 2. Traceability matrix

| ID | Requirement | Design element | Doc |
|----|-------------------|----------------|-----|
| A1 | Orchestrator plans & coordinates | `OrchestratorAgent`: builds an `InvestigationPlan` (task DAG) and re-plans after each Critic verdict | 03, 04 |
| A2 | Observability Agent: logs & metrics | `ObservabilityAgent` with `query_logs` and `query_metrics` tools | 04, 05 |
| A3 | Dependency Agent: deploys, config, deps | `DependencyAgent` with `get_deployments`, `get_config_changes` and `get_service_dependencies` | 04, 05 |
| A4 | Reasoning Agent: competing hypotheses | `ReasoningAgent` produces at least 2 ranked `Hypothesis` objects with supporting and contradicting evidence IDs | 04 |
| A5 | Critic Agent: challenges, requests more | `CriticAgent` returns a `CriticVerdict` with `evidence_requests[]` that become new tasks | 04, 06 |
| A6 | Reporting Agent: final report + remediation | `ReportingAgent` produces `InvestigationReport` (JSON + Markdown) | 04 |
| A7 | Structured exchange, genuine collaboration | Pydantic contracts, shared `InvestigationState` (blackboard) and a Critic→Orchestrator feedback loop | 03 |
| B1 | Tools: logs, metrics, deploys, deps, history | 6 tools (config changes split out from deploys) plus `search_incident_history` | 05 |
| B2 | Structured I/O + error handling | Pydantic `ToolInput`/`ToolResult` envelope with `status`, `error`, `completeness` and `warnings` | 05 |
| B3 | Simulated failures (timeouts, incomplete) | Seeded `FaultInjector`: timeout, 5xx, partial page, stale data, empty result, rate limit | 05 |
| C1 | Parallel execution | LangGraph `Send` fan-out across specialists, plus parallel tool calls inside agents | 03, 06 |
| C2 | Shared state + task dependencies | LangGraph `InvestigationState` with reducers, graph edges and `Task.depends_on[]` | 03 |
| C3 | Feedback loops for more evidence | The Critic's `NEED_MORE_EVIDENCE` triggers targeted follow-up tasks, at most 3 rounds | 06 |
| C4 | Bounded retries, limits, termination | `Budget` (tokens, tool calls, wall-clock, rounds), per-tool retry with backoff and jitter, and explicit stop conditions | 06 |
| C5 | Recovery from agent/tool failures | Tool errors are returned as data. Failed agent tasks are retried and then marked `degraded`, and the pipeline continues with the evidence it has | 06 |
| D1 | Ranked hypotheses, supporting and contradicting evidence, confidence, actions | `InvestigationReport` schema | 04 |
| D2 | Facts vs assumptions | `Evidence.kind = FACT` (tool-sourced) vs `Claim.kind = INFERENCE / ASSUMPTION`, plus a code-level citation validator | 03, 04 |
| D3 | Inconclusive outcome | `Report.outcome ∈ {ROOT_CAUSE_IDENTIFIED, PROBABLE, INCONCLUSIVE}`, driven by confidence thresholds and the Critic | 06 |
| E1 | ≥10 test incidents incl. misleading, conflicting, tool failures | 12 scenarios (see 07), each tagged with its adversarial traits | 07 |
| E2 | Single-agent baseline, same tools | `SingleAgentInvestigator`: a ReAct loop with an identical tool registry and the same budget | 07 |
| E3 | Measure accuracy, grounding, tool calls, latency, tokens | `eval/run_eval.py` writes a per-run JSONL plus summary tables in CSV and Markdown | 07 |
| F1 | Runnable implementation + setup | `uv` project, `make setup`, `make demo` and `make eval` | 08 |
| F2 | Architecture diagram + coordination | `docs/03-architecture.md` (Mermaid) | 03 |
| F3 | Synthetic dataset + automated tests | `data/scenarios/*.yaml`, a generator and a `pytest` suite (offline via FakeLLM) | 05, 07, 08 |
| F4 | Eval results comparing both approaches | `eval/results/summary.md` | 07 |
| F5 | Technical note (≤500 words) | `docs/TECHNICAL_NOTE.md` (draft in 09) | 09 |
| G1 | Any language, provider or framework | Python + OpenAI (`langchain-openai`) + LangGraph | 02 |
| G3 | No hardcoded answers | Agents never see scenario IDs or ground truth. A test asserts that agent code contains no scenario identifiers | 07 |
| G4 | No polished frontend | CLI (`rich`) plus Markdown reports. An optional tiny HTML trace viewer is a stretch goal | 08 |
| G5 | Document incomplete functionality and dependencies | A "Limitations & external dependencies" section in the README | 08 |

## 3. Implicit requirements and good-to-haves

- **Reproducibility**: a seeded simulation, recorded LLM responses for tests, and a pinned model ID.
- **Observability of the agent system itself**: a JSONL trace of every LLM call, tool call, state change and verdict, which is useful for the demo and for the evaluation.
- **Cost transparency**: a `$` column in the evaluation computed from `usage` with real per-token pricing.
