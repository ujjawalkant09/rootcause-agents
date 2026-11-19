# 03 – Architecture & Agent Coordination

> Implementation: **LangGraph `StateGraph`** with OpenAI models. The diagrams below show the logical design, and §0 shows the actual graph.

## 0. LangGraph graph (what will be implemented)

```mermaid
flowchart LR
    START((START)) --> plan[orchestrator<br/>plan / re-plan]
    plan -- "Send() per task<br/>(parallel)" --> obs[observability]
    plan -- "Send() per task" --> dep[dependency]
    plan -- "Send() round 1" --> hist[history<br/>no LLM]
    obs --> reason[reasoning]
    dep --> reason
    hist --> reason
    reason -- citations invalid<br/>(≤2 retries) --> reason
    reason --> critic[critic]
    critic -- NEED_MORE_EVIDENCE<br/>& round<3 & budget ok --> plan
    critic -- ACCEPT / INCONCLUSIVE<br/>/ limit reached --> report[reporter]
    report --> END((END))
```

```python
# sketch – src/rootcause_agents/graph.py
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, RetryPolicy

class InvestigationState(TypedDict):
    incident: IncidentReport
    plan: list[Task]
    evidence: Annotated[list[Evidence], operator.add]       # parallel branches merge
    findings: Annotated[list[SpecialistFindings], operator.add]
    data_gaps: Annotated[list[str], operator.add]
    hypotheses: list[Hypothesis]
    verdicts: Annotated[list[CriticVerdict], operator.add]
    round: int
    usage: Annotated[list[UsageRecord], operator.add]      # tokens / tool calls per node
    report: InvestigationReport | None

def fan_out(state):        # router after orchestrator
    return [Send(t.agent, {"task": t, "incident": state["incident"]}) for t in state["plan"]]

def after_critic(state):   # termination controller
    v = state["verdicts"][-1]
    if v.decision == "NEED_MORE_EVIDENCE" and state["round"] < MAX_ROUNDS and budget_ok(state):
        return "orchestrator"
    return "reporter"

retry = RetryPolicy(max_attempts=3)
g = StateGraph(InvestigationState)
g.add_node("orchestrator", orchestrator_node, retry_policy=retry)
g.add_node("observability", observability_node, retry_policy=retry)
g.add_node("dependency", dependency_node, retry_policy=retry)
g.add_node("history", history_node)
g.add_node("reasoning", reasoning_node, retry_policy=retry)
g.add_node("critic", critic_node, retry_policy=retry)
g.add_node("reporter", reporter_node, retry_policy=retry)
g.add_edge(START, "orchestrator")
g.add_conditional_edges("orchestrator", fan_out, ["observability", "dependency", "history"])
for n in ("observability", "dependency", "history"):
    g.add_edge(n, "reasoning")
g.add_edge("reasoning", "critic")
g.add_conditional_edges("critic", after_critic, ["orchestrator", "reporter"])
g.add_edge("reporter", END)
graph = g.compile()
# run: graph.invoke({"incident": inc, "round": 0}, {"recursion_limit": 40})
```

> Exact LangGraph signatures (e.g. `RetryPolicy` fields) will be checked against the installed version during implementation.

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
    participant U as CLI
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

1. **The Orchestrator plans once and re-plans after every Critic verdict.** Its output is a typed `InvestigationPlan`: a list of `Task{id, agent, objective, params_hint, depends_on, priority}`. It never calls tools directly.
2. **LangGraph executes the plan.** The `Send` API fans out one branch per task, and the branches run concurrently in the same superstep. Each task has a timeout and a retry policy, and a failure is recorded on the blackboard without crashing the run.
3. **Specialists are tool-using agents.** Each one runs a short bounded tool loop (at most 6 tool turns) against *its own* subset of tools. It emits `Evidence` records plus a short `Findings` summary with explicit *gaps* ("metrics for `inventory` unavailable – timeout ×3").
4. **The Reasoning Agent sees only the blackboard**: evidence summaries, gaps and prior critiques. It does not see raw transcripts. It outputs ranked hypotheses that cite evidence IDs.
5. **Code validates citations** before the Critic sees anything (invalid IDs trigger a re-prompt).
6. **The Critic** checks for anchoring, temporal ordering (cause must precede effect), unexplained symptoms, conflicting evidence and missing data. It returns a verdict plus *specific* evidence requests.
7. **The termination controller** decides whether to loop again or report (see 06).
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
