# 04 – Agents & Data Contracts

All contracts are Pydantic v2 models in `src/rootcause_agents/schemas.py`. LLM agents return them through OpenAI structured outputs via LangChain (`ChatOpenAI(model=...).with_structured_output(Model)`), so the outputs are always schema-valid. Tool-using agents use `.bind_tools([...])` in a bounded loop.

**Model per agent:** every agent uses `MAIN_MODEL` (`gpt-6.2-sol`) by default. In the *cheaper-worker* evaluation variant, Observability and Dependency switch to `WORKER_MODEL` (`gpt-6-luna`). The "Effort" rows below translate to prompt and temperature settings (and `reasoning_effort` if the chosen model supports it).

## 1. Shared core schemas (sketch)

```python
from enum import StrEnum
from pydantic import BaseModel, Field

class EvidenceKind(StrEnum):
    FACT = "fact"                 # directly from a tool result

class ClaimKind(StrEnum):
    INFERENCE = "inference"       # reasoning over cited facts
    ASSUMPTION = "assumption"     # not backed by evidence

class Evidence(BaseModel):
    id: str                       # "E-12" (assigned by code, not the LLM)
    source_tool: str              # "query_metrics"
    params: dict                  # exact tool input
    summary: str                  # 1–2 sentence factual summary (code- or agent-generated)
    data: dict                    # bounded structured payload
    completeness: float           # 0..1 (partial pages, missing windows)
    tool_status: str              # ok | partial | stale | error
    observed_window: tuple[str, str] | None
    collected_by: str             # agent name
    round: int

class Claim(BaseModel):
    text: str
    kind: ClaimKind
    evidence_ids: list[str] = []

class Hypothesis(BaseModel):
    id: str
    title: str                                   # "DB connection-pool exhaustion in payments-db"
    root_cause_component: str                    # normalised service/component name (graded)
    root_cause_category: str                     # enum: bad_deploy | config_change | dependency_failure | resource_exhaustion | traffic_surge | infra | data | unknown
    mechanism: list[Claim]                       # causal chain
    supporting_evidence_ids: list[str]
    contradicting_evidence_ids: list[str]
    unexplained_symptoms: list[str]
    confidence: float = Field(ge=0, le=1)

class EvidenceRequest(BaseModel):
    target_agent: str                            # "observability" | "dependency"
    question: str                                # what must be answered
    rationale: str                               # which hypothesis it discriminates
    suggested_tool: str | None = None
    params_hint: dict = {}

class CriticVerdict(BaseModel):
    decision: str                                # ACCEPT | NEED_MORE_EVIDENCE | INCONCLUSIVE
    issues: list[str]                            # anchoring, temporal violation, uncited claim…
    evidence_requests: list[EvidenceRequest] = []
    confidence_adjustments: dict[str, float] = {} # hypothesis_id -> adjusted confidence
```

## 2. Agent specifications

### 2.1 Orchestrator
| | |
|---|---|
| Input | `IncidentReport` (title, symptoms, affected services, start time, severity) plus the current blackboard summary, the last `CriticVerdict` and the remaining budget |
| Output | `InvestigationPlan { tasks: list[Task], rationale }` |
| Tools | None. It plans only |
| Effort | `medium` |
| Key prompt rules | Decompose by evidence source. Make tasks independent where possible so they run in parallel. Include history search on round 1. On re-plan, create *only* tasks that answer the Critic's requests. Respect the remaining budget |

### 2.2 Observability Agent
| | |
|---|---|
| Input | `Task` (objective, services, time window hint) |
| Output | `SpecialistFindings { evidence_ids, key_observations: list[Claim], gaps: list[str], conflicts: list[str] }` |
| Tools | `query_logs`, `query_metrics` |
| Loop | At most 6 tool turns. Parallel tool calls are encouraged |
| Effort | `low`/`medium` |
| Key prompt rules | Establish the *timeline* (first anomaly per service). Compare against a baseline window. Report gaps from tool failures explicitly. Never speculate about root cause |

### 2.3 Dependency Agent
| | |
|---|---|
| Input | `Task` |
| Output | `SpecialistFindings` |
| Tools | `get_deployments`, `get_config_changes`, `get_service_dependencies` |
| Key prompt rules | List changes in the window from T−2h to T+30m. Map the blast radius through the dependency graph (upstream/downstream). Flag changes on services *outside* the symptom path as weaker candidates |

### 2.4 History lookup
A deterministic tool task (`search_incident_history`) with no LLM, which saves tokens. Optionally an LLM summariser could be added. It returns similar past incidents with their root causes, which the Reasoning Agent treats as *priors, not evidence of cause*.

### 2.5 Reasoning Agent
| | |
|---|---|
| Input | The blackboard's evidence summaries, specialist findings (with gaps and conflicts), history priors and previous Critic issues |
| Output | `HypothesisSet { hypotheses: list[Hypothesis] (≥2, ranked), timeline: list[Claim], discriminating_questions: list[str] }` |
| Effort | `high` |
| Key prompt rules | Generate competing explanations. Each needs supporting **and** contradicting evidence. Check temporal precedence. Prefer explanations that explain *all* symptoms. Mark uncited claims as `ASSUMPTION`. Confidences across hypotheses should sum to ≤ 1, with any remainder treated as "other/unknown" |
| Post-check (code) | All cited IDs exist. Each hypothesis has ≥1 supporting ID. If either check fails, re-prompt (at most 2 times) |

### 2.6 Critic Agent
| | |
|---|---|
| Input | `HypothesisSet`, the referenced evidence, the round number and the remaining budget |
| Output | `CriticVerdict` |
| Effort | `high` |
| Checklist | (1) Anchoring on a coincident change? (2) Does cause precede effect? (3) Unexplained symptoms? (4) Is conflicting evidence ignored? (5) Does key evidence come from partial or stale tool results? (6) Is the confidence calibrated given the gaps? (7) Is there a cheap query that would discriminate between H1 and H2? |
| Rules | Request evidence only if it would change the ranking. Allow at most 3 requests per round. Return `INCONCLUSIVE` if the critical data is unavailable after retries |

### 2.7 Reporting Agent
| | |
|---|---|
| Input | The final blackboard and the final verdict |
| Output | `InvestigationReport` (below), rendered to Markdown by a Jinja template |
| Effort | `medium` |
| Rule | No new claims. Every statement maps to evidence or is labelled an assumption. A code check enforces this |

## 3. Final report schema

```python
class RecommendedAction(BaseModel):
    action: str
    type: str            # mitigate_now | fix | prevent | investigate_further
    priority: str        # P0..P3
    linked_hypothesis: str
    evidence_ids: list[str]

class InvestigationReport(BaseModel):
    incident_id: str
    outcome: str                             # ROOT_CAUSE_IDENTIFIED | PROBABLE | INCONCLUSIVE
    summary: str
    ranked_hypotheses: list[Hypothesis]
    facts: list[Claim]                       # kind=fact, each with evidence_ids
    inferences: list[Claim]
    assumptions: list[Claim]
    timeline: list[Claim]
    data_gaps: list[str]                     # tool failures, missing data
    recommended_actions: list[RecommendedAction]
    investigation_stats: dict                # rounds, tool calls, tokens, latency, cost
```

## 4. Prompting principles applied to every agent
- A stable system prompt placed first (so OpenAI's automatic prefix caching applies) covering role, rules and the output contract. The dynamic content goes in the user turn.
- Agents receive an **objective, an output format, tool guidance and boundaries**. This follows published multi-agent lessons: vague subagent briefs cause duplicated or incomplete work.
- Tool descriptions are written like docs for a junior engineer: when to use the tool, parameter semantics, and what `partial` means.
- No scenario-specific hints anywhere (see the "no hardcoding" test in 07).
