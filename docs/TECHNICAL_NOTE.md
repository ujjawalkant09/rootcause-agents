# Technical Note: RootCause Agents

**Key decisions.**
Six agents (Orchestrator, Observability, Dependency, Reasoning, Critic, Reporter) run on a LangGraph `StateGraph` over one typed state. Every successful tool call becomes an `Evidence` record with a code-assigned ID. Hypotheses and report claims must cite these IDs, a validator rejects unknown IDs, and uncited claims must be labelled assumptions. Facts versus assumptions is therefore enforced structurally. The final outcome (identified, probable or inconclusive) is decided by a code policy, never by the LLM.

Coordination combines orchestrator–workers with an evaluator loop. The Orchestrator emits typed tasks, `Send` runs specialists in parallel, and the Critic can return specific evidence requests that become the next round's tasks. The loop is bounded by rounds, a tool budget with a per-task quota, tokens and wall-clock time. Tools pre-aggregate telemetry (log clusters, metric change points) and never raise. Injected timeouts, partial or stale data and outages are retried with backoff, then recorded as data gaps. Failed nodes degrade instead of crashing.

For LLM resilience and cost: a per-role model fallback chain with a circuit breaker, a cache-friendly prompt prefix, compact tool payloads (49% smaller), output-length limits and a hard evaluation spend cap. Langfuse traces each run as typed agent, tool and evaluator observations.

**Results** (10 incidents, 1 run each, $2.02 total).
The single-agent baseline matched the multi-agent system on top-1 accuracy (10/10 each) at about one third of the cost ($0.036 vs $0.109 per incident) and latency (51 s vs 147 s). The multi-agent system reached the correct outcome on all 10 incidents. The baseline was correct about the cause but declared "inconclusive" on two misleading cases, where the critic's second round supplied the missing evidence. All systems had 100% valid citations. Using a cheaper model for the data-gathering agents cut cost by 48% but missed one conflicting case.

**Trade-offs.**
Collaboration costs 6.5× more LLM calls. In return it gives better outcome calibration on hard cases, auditable intermediate artefacts and isolated failure domains. Pre-aggregating tools helped both systems the most, which narrows the gap. LangGraph gave fan-out and routing cheaply, at the price of abstraction. A real-API pilot exposed two defects that offline tests missed: the budget was drained in round 1, and the critic demanded proof the tools cannot provide. Fixing them raised pilot accuracy from 0.5 to 1.0.

**Limitations.**
- The telemetry is synthetic and cleaner than production data, and the scenarios are near the ceiling for both systems.
- n=10 with one run each is not statistically significant, and two grading aliases are lenient (documented).
- Confidence comes from the LLM and is uncalibrated; grounding metrics are code-based, not semantic.
- Custom tickets run against simulated environments only.

**Improvements.**
- A harder, noisier dataset (e.g. from OpenRCA) with repeated runs.
- An LLM-judge for semantic grounding, and confidence calibration.
- Real observability backends behind the same tool interfaces.
- Human approval gates for remediation actions (sessions already persist in Postgres and can pause for hints).
- Prompt deltas across rounds to cut tokens.
