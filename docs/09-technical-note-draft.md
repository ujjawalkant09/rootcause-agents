# 09 – Technical Note (DRAFT, ≤500 words)

> This is a pre-implementation draft. The numbers in [brackets] will be filled in from the evaluation results, and the final version will be saved as `docs/TECHNICAL_NOTE.md`.

---

**Key decisions.**
The system is a LangGraph `StateGraph` with six agents powered by OpenAI models. All agents share one typed investigation state: an evidence ledger, ranked hypotheses, the plan, critic verdicts and usage. Parallel branches merge into it through reducers. Every tool result becomes an `Evidence` record with a code-assigned ID. Hypotheses must cite these IDs, and a validator rejects citations to non-existent evidence. As a result, "facts vs assumptions" and evidence grounding are enforced structurally rather than requested in prompts.

Coordination combines *orchestrator–workers* with an *evaluator–optimizer* loop. The Orchestrator emits a plan, and LangGraph's `Send` API runs the Observability, Dependency and history tasks in parallel. The Reasoning Agent must produce at least two competing hypotheses, each with supporting and contradicting evidence. The Critic returns `ACCEPT`, `NEED_MORE_EVIDENCE` (with structured evidence requests that the Orchestrator turns into new tasks) or `INCONCLUSIVE`. A conditional edge bounds the loop by rounds, tool calls, tokens and time, with `recursion_limit` as a final safety net.

Tools pre-aggregate telemetry: log-signature clusters and metric change points instead of raw data. This was the biggest lever on both token cost and accuracy. Tool failures (timeouts, partial pages, stale data) are returned as data, recorded as gaps and reflected in confidence.

**Trade-offs.**
Compared with a single-agent baseline using the same tools and budget, the multi-agent system used [X×] more tokens and [Y%] more latency. It changed top-1 accuracy from [a%] to [b%], mainly on misleading and conflicting scenarios. Running the worker agents on the cheaper `gpt-6-luna` cut cost by [Z%] at [Δ] accuracy. LangGraph gave parallelism, retries and routing for free, at the cost of some abstraction.

**Limitations.**
- The telemetry is synthetic and cleaner than production data.
- There are 10 scenarios with one run each, so the results are not statistically significant.
- Grounding metrics are code-based (citation validity and key-evidence recall) and do not judge semantic support.
- Confidence comes from the LLM and is uncalibrated.
- There are no real observability integrations or persistence.

**Improvements.**
- Real backends (Loki, Prometheus, Argo) behind the same tool interfaces.
- Repeated runs and an LLM-judge support score.
- Confidence calibration.
- A larger adversarial set drawn from public RCA benchmarks such as OpenRCA.
- A LangGraph checkpointer for resumable investigations.
- Human-in-the-loop approval of remediation actions.

*(~400 words)*
