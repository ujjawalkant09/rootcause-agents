# 03 – Architecture & Agent Coordination

> Implementation: **LangGraph `StateGraph`** (`src/rootcause_agents/graph.py`) with OpenAI models via `langchain-openai`. §0 is the graph exactly as built, §0.1 the runtime around it, §1–3 the logical design.

## 0. LangGraph graph (as implemented)

Image: [`diagrams/01-langgraph-graph.png`](diagrams/01-langgraph-graph.png) (all diagrams are also exported as PNG in [`docs/diagrams/`](diagrams/), with `.mmd` sources).

```mermaid
flowchart LR
    START((START)) --> plan["orchestrator<br/>plan / re-plan<br/>(≤4 tasks per round)"]
    plan -- "Send() one branch per task<br/>(parallel, max_concurrency 4)" --> obs["observability<br/>logs + metrics"]
    plan -- "Send() per task" --> dep["dependency<br/>deploys · config · topology"]
    plan -- "Send() round 1 only" --> hist["history<br/>(deterministic, no LLM)"]
    plan -- "no tasks" --> reason
    obs --> reason["reasoning<br/>≥2 competing hypotheses"]
    dep --> reason
    hist --> reason
    reason -. "invalid evidence IDs<br/>re-prompt ≤2×" .-> reason
    reason --> critic["critic<br/>ACCEPT · NEED_MORE_EVIDENCE · INCONCLUSIVE"]
    critic -- "NEED_MORE_EVIDENCE + requests<br/>& round < max & budget ok<br/>& last round added evidence" --> plan
    critic -- "stuck (or review mode)<br/>& asked < 2 times" --> human["human_review<br/>interrupt() → pause<br/>state saved in Postgres"]
    human -- "continue + hint<br/>(one extra round)" --> plan
    human -- "write report" --> report
    critic -- "otherwise" --> report["reporter<br/>outcome decided by code"]
    report --> END((END))
```

**Persistence and human-in-the-loop.** The graph is compiled with a LangGraph checkpointer (`AsyncPostgresSaver` on the local Docker Postgres, or in-memory without `DATABASE_URL`), and each investigation is a thread (`thread_id` = run id) checkpointed after every super-step. In `stuck` mode, when the critic cannot conclude, `human_review` calls `interrupt()`: the run pauses and the question (top hypotheses, critic issues) is shown in the UI. `resume_session` rebuilds the run context from the checkpoint (seeded simulator, restored budget and evidence counters, so it works from a new process) and continues with `Command(resume=answer)`. A human hint is passed to later agents as a *lead to verify with tools*. It is never citable evidence, and it appears in the report under "Human input". Each run asks at most 2 times, and evaluation runs never pause. Checkpoint deserialization uses an explicit allowlist of our schema classes.

**Shared state** (`InvestigationState`, a `TypedDict` with reducers so parallel branches merge safely):

| Key | Reducer | Written by |
|---|---|---|
| `evidence: list[Evidence]` | `merge_evidence` (dedupe by code-assigned `E-n` id) | specialists, history |
| `findings: list[SpecialistFindings]` | append | specialists |
| `data_gaps: list[str]` | append | any node, tool failures |
| `plan: list[Task]` (`agent, objective, services, rationale`) | replace | orchestrator |
| `hypotheses: list[Hypothesis]` | replace | reasoning |
| `verdicts: list[CriticVerdict]` | append | critic |
| `round`, `evidence_at_round_start` | replace | orchestrator |
| `report: InvestigationReport` | replace | reporter |

**Task dependencies** are structural: specialists depend only on the plan, so they run in parallel; reasoning waits for *all* branches of the round (LangGraph superstep barrier); the critic depends on reasoning; the next round depends on the critic's evidence requests, which the orchestrator turns into new tasks.

**Reliability inside the graph:**
- every node runs in `_resilient`: 2 attempts, then it degrades (a gap is recorded and the run continues);
- the tool loop is bounded: ≤6 LLM turns and ≤8 tool calls per task (`max_tool_calls_per_task`, which keeps budget for follow-up rounds);
- global budget: ≤40 tool calls, 400k tokens, 420 s and `max_rounds` (default 2 in `.env`), with `recursion_limit=60` as a final net;
- the outcome comes from `budget.decide_outcome`: ACCEPT + top ≥0.75 + margin ≥0.2 → ROOT_CAUSE_IDENTIFIED; ACCEPT → PROBABLE; else INCONCLUSIVE.

## 0.1 Runtime view (what surrounds the graph)

Image: [`diagrams/02-runtime.png`](diagrams/02-runtime.png)

```mermaid
flowchart TB
    UI["Streamlit UI<br/>test incident or custom ticket"] --> RUN
    CLI["CLI<br/>investigate · eval · ui"] --> RUN
    RUN["runner.investigate()<br/>builds RunContext"] --> G["LangGraph StateGraph<br/>(§0)"]
    G --> LLM["OpenAILLM<br/>per-role model · semaphore"]
    LLM --> CH{"fallback chain<br/>+ circuit breaker"}
    CH --> M1["MAIN_MODEL<br/>gpt-6.1-terra"]
    CH --> M2["FALLBACK_MODELS<br/>gpt-6-luna …"]
    CH -.optional.-> M3["second provider<br/>FALLBACK_BASE_URL"]
    G --> EX["ToolExecutor<br/>validation · retries+backoff · memo<br/>evidence IDs · budget"]
    EX --> FI["FaultInjector<br/>timeout · rate limit · partial · stale · empty"]
    FI --> SIM[("TelemetryStore<br/>seeded synthetic logs, metrics,<br/>deploys, config, topology, history")]
    RUN --> OUT["runs/&lt;id&gt;/report.json · report.md · trace.jsonl"]
    G -.callbacks + typed observations.-> LF[("Langfuse<br/>optional")]
    EX -.tool observations.-> LF
```

- **Prompt caching:** system prompt = instructions + static environment only, plus `prompt_cache_key` per role. Cached tokens are priced at the cached rate.
- **Token diet:** tool results reach the model compacted (`compact_data`), while the ledger keeps the full data.
- **Observability:** one Langfuse trace per run (`investigate-incident`), with `agent` / `tool` / `retriever` / `evaluator` observations, a generation per LLM call, and scores for outcome, confidence, cost, tool calls and latency. See the README.

## 1. High-level component diagram (logical)

```mermaid
flowchart TB
    IR[Incident Report<br/>JSON/YAML] --> ORCH

    subgraph Core["Orchestration Core"]
        ORCH[Orchestrator Agent<br/>plan / re-plan]
        SCHED[LangGraph runtime<br/>Send fan-out]
        BUD[Budget & Termination<br/>Controller]
        BB[(Blackboard<br/>InvestigationState<br/>evidence ledger · hypotheses · tasks)]
    end

    subgraph Specialists["Specialist Agents (parallel)"]
        OBS[Observability Agent]
        DEP[Dependency Agent]
        HIST[History lookup<br/>tool task]
    end

    subgraph Analysis["Analysis Agents"]
        REA[Reasoning Agent<br/>competing hypotheses]
        CRI[Critic Agent<br/>challenge / request evidence]
        REP[Reporting Agent]
    end

    subgraph Tools["Tool Layer (structured I/O + FaultInjector)"]
        T1[query_logs]
        T2[query_metrics]
        T3[get_deployments]
        T4[get_config_changes]
        T5[get_service_dependencies]
        T6[search_incident_history]
    end

    SIM[(Simulated Environment<br/>seeded scenario data)]

    ORCH --> SCHED
    SCHED --> OBS & DEP & HIST
    OBS --> T1 & T2
    DEP --> T3 & T4 & T5
    HIST --> T6
    T1 & T2 & T3 & T4 & T5 & T6 --> SIM
    OBS & DEP & HIST -->|Evidence| BB
    BB --> REA -->|Hypotheses| BB
    BB --> CRI -->|Verdict + EvidenceRequests| ORCH
    CRI -->|ACCEPT / INCONCLUSIVE| REP
    BB --> REP --> OUT[InvestigationReport<br/>JSON + Markdown]
    BUD -.enforces.- SCHED
    BUD -.enforces.- ORCH
    TRACE[(JSONL Trace)] -.-> Core
```

## 2. Investigation lifecycle (state machine)

```mermaid
stateDiagram-v2
    [*] --> PLANNING
    PLANNING --> GATHERING: plan (task DAG)
    GATHERING --> REASONING: all ready tasks done / degraded
    REASONING --> CRITIQUE: >=2 hypotheses with valid citations
    REASONING --> REASONING: citation validation failed (retry ≤2)
    CRITIQUE --> GATHERING: NEED_MORE_EVIDENCE and round < 3 and budget ok
    CRITIQUE --> REPORTING: ACCEPT
    CRITIQUE --> REPORTING: INCONCLUSIVE or round limit or budget exhausted
    REPORTING --> [*]
```

## 3. Sequence: one investigation including the feedback loop

```mermaid
sequenceDiagram
    autonumber
    participant U as UI / CLI
    participant O as Orchestrator
    participant S as Scheduler
    participant OB as Observability
    participant DP as Dependency
    participant B as Blackboard
    participant R as Reasoning
    participant C as Critic
    participant RP as Reporting

    U->>O: IncidentReport
    O->>B: InvestigationPlan (tasks T1..Tn, deps)
    O->>S: schedule ready tasks
    par parallel
        S->>OB: T1 logs+metrics for checkout
        OB->>B: Evidence E1..E6 (1 tool call timed out → retried → partial)
    and
        S->>DP: T2 deploys/config/deps
        DP->>B: Evidence E7..E10
    and
        S->>B: T3 history search → E11
    end
    S->>R: all gathering done
    R->>B: H1 (deploy D-42, conf .55), H2 (DB saturation, conf .40)
    B->>C: hypotheses + evidence
    C->>O: NEED_MORE_EVIDENCE: "DB p99 before D-42?", "D-42 diff touches pool size?"
    O->>S: follow-up tasks T4 (OBS), T5 (DEP)
    par
        S->>OB: T4
        OB->>B: E12: DB p99 spiked 6 min BEFORE D-42
    and
        S->>DP: T5
        DP->>B: E13: D-42 only changes UI copy
    end
    S->>R: re-reason (round 2)
    R->>B: H2 (DB saturation, .78) > H1 (.12)
    C->>O: ACCEPT
    O->>RP: generate report
    RP->>U: InvestigationReport (ranked H, facts vs assumptions, actions)
```

This sequence also shows a typical *misleading evidence* scenario: a deploy coincides with the incident but is not the cause.

## 4. Coordination model, explained

**Pattern:** a hybrid of *orchestrator–workers* (dynamic task decomposition) and *evaluator–optimizer* (Critic feedback loop) over a shared *blackboard*.

1. **The Orchestrator plans once and re-plans after every Critic verdict.** Its output is a typed `InvestigationPlan`: a list of `Task{agent, objective, services, rationale}` (at most 4 per round). It never calls tools directly.
2. **LangGraph executes the plan.** The `Send` API fans out one branch per task, and the branches run concurrently in the same superstep. Each node gets 2 attempts; a final failure is recorded as a data gap without crashing the run.
3. **Specialists are tool-using agents.** Each one runs a short bounded tool loop (at most 6 LLM turns and 8 tool calls per task) against *its own* subset of tools. It emits `Evidence` records plus a short `Findings` summary with explicit *gaps* ("metrics for `inventory` unavailable – timeout ×3").
4. **The Reasoning Agent sees only the blackboard**: evidence summaries, gaps and prior critiques. It does not see raw transcripts. It outputs ranked hypotheses that cite evidence IDs.
5. **Code validates citations** before the Critic sees anything (invalid IDs trigger a re-prompt).
6. **The Critic** checks for anchoring, temporal ordering (cause must precede effect), unexplained symptoms, conflicting evidence and missing data. It returns a verdict plus *specific* evidence requests.
7. **The termination controller** (`after_critic` + `should_continue`) loops back only if the Critic asked for specific evidence, rounds and budget remain, and the last round added new evidence; otherwise it reports (see 06).
8. **The Reporting Agent** turns the final blackboard into the report. It is not allowed to introduce new claims; a code check verifies that every cited ID exists.

### Why this is "genuine collaboration" and not a prompt chain
- The control flow is **data-dependent**. The Critic's output changes which agents run next and with what objectives.
- Agents **communicate through typed artifacts on a shared state**, not by passing free-text transcripts.
- Specialists report **gaps and conflicts**, and these drive the Reasoning and Critic behaviour.
- The number of rounds, tasks and tool calls varies by incident. Easy incidents finish in 1 round; adversarial ones take 2–3.

## 5. Single-agent baseline (for comparison)

```mermaid
flowchart LR
    IR[Incident] --> SA[Single ReAct Agent<br/>same 6 tools, same budget]
    SA <--> TOOLS[Tool Layer + FaultInjector]
    SA --> REP2[Same InvestigationReport schema]
```

The baseline uses the same tool registry, fault injection seed, model, budget caps and output schema. Only the architecture differs.
