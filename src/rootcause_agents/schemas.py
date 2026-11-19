"""Pydantic contracts shared by tools, agents, graph, report and evaluation.

Models listed in ``LLM_OUTPUT_MODELS`` are produced by the LLM via OpenAI structured outputs, so they
must not contain free-form ``dict`` fields (enforced by tests/test_schemas.py).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RootCauseCategory = Literal[
    "bad_deploy",
    "config_change",
    "dependency_failure",
    "resource_exhaustion",
    "traffic_surge",
    "infrastructure",
    "data_migration",
    "unknown",
]
Outcome = Literal["ROOT_CAUSE_IDENTIFIED", "PROBABLE", "INCONCLUSIVE"]
AgentName = Literal["observability", "dependency"]
ToolStatus = Literal["ok", "partial", "stale", "empty", "error"]
ToolErrorCode = Literal["TIMEOUT", "UNAVAILABLE", "RATE_LIMITED", "BAD_INPUT", "BUDGET_EXHAUSTED", "INTERNAL"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- incident input
class IncidentReport(_Strict):
    id: str
    title: str
    description: str
    reported_at: datetime
    affected_services: list[str]
    symptoms: list[str]


# --------------------------------------------------------------------------- tools
class ToolError(_Strict):
    code: ToolErrorCode
    message: str
    retryable: bool = False


class ToolResult(_Strict):
    tool: str
    status: ToolStatus
    data: Any = None
    completeness: float = Field(default=1.0, ge=0.0, le=1.0)
    truncated: bool = False
    warnings: list[str] = []
    error: ToolError | None = None
    attempts: int = 1

    @property
    def is_error(self) -> bool:
        return self.status == "error"


class Evidence(_Strict):
    """A FACT: the (summarised) result of exactly one successful tool call."""

    id: str
    source_tool: str
    params: dict[str, Any]
    status: ToolStatus
    completeness: float
    summary: str
    data: Any
    warnings: list[str] = []
    collected_by: str
    round: int


# --------------------------------------------------------------------------- LLM outputs
class Claim(_Strict):
    text: str
    kind: Literal["fact", "inference", "assumption"]
    evidence_ids: list[str] = Field(default_factory=list)


class Task(_Strict):
    agent: AgentName
    objective: str = Field(description="What question this task must answer.")
    services: list[str] = Field(description="Services to focus on.")
    rationale: str = Field(description="Why this task matters (which hypotheses it helps discriminate).")


class InvestigationPlan(_Strict):
    tasks: list[Task]
    rationale: str


class SpecialistFindings(_Strict):
    key_observations: list[Claim] = Field(description="Factual observations, each citing evidence IDs.")
    timeline: list[Claim] = Field(default_factory=list, description="Time-ordered events with evidence IDs.")
    gaps: list[str] = Field(default_factory=list, description="Data that could not be obtained and why.")
    conflicts: list[str] = Field(default_factory=list, description="Observations that contradict each other.")


class Hypothesis(_Strict):
    id: str
    title: str
    root_cause_component: str = Field(description="The service/component where the root cause lives.")
    root_cause_category: RootCauseCategory
    mechanism: list[Claim] = Field(description="Causal chain from root cause to observed symptoms.")
    supporting_evidence_ids: list[str]
    contradicting_evidence_ids: list[str]
    unexplained_symptoms: list[str]
    confidence: float = Field(description="Probability in [0, 1].")

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:  # strict schemas can't carry min/max, so clamp instead of failing
        return min(1.0, max(0.0, float(v)))


class HypothesisSet(_Strict):
    hypotheses: list[Hypothesis] = Field(description="At least two competing hypotheses, ranked best first.")
    timeline: list[Claim] = Field(default_factory=list)
    assumptions: list[Claim] = Field(default_factory=list)


class EvidenceRequest(_Strict):
    target_agent: AgentName
    question: str
    services: list[str]
    discriminates: str = Field(description="Which hypotheses this evidence would separate and how.")


class CriticVerdict(_Strict):
    decision: Literal["ACCEPT", "NEED_MORE_EVIDENCE", "INCONCLUSIVE"]
    issues: list[str]
    evidence_requests: list[EvidenceRequest] = Field(default_factory=list)


class RecommendedAction(_Strict):
    action: str
    type: Literal["mitigate_now", "fix", "prevent", "investigate_further"]
    priority: Literal["P0", "P1", "P2", "P3"]
    evidence_ids: list[str] = Field(default_factory=list)


class ReportDraft(_Strict):
    summary: str
    facts: list[Claim]
    inferences: list[Claim]
    assumptions: list[Claim]
    recommended_actions: list[RecommendedAction]


class BaselineConclusion(_Strict):
    """Final structured output of the single-agent baseline."""

    hypotheses: list[Hypothesis]
    conclusive: bool = Field(description="False if the evidence is insufficient to name a root cause.")
    report: ReportDraft


LLM_OUTPUT_MODELS: list[type[BaseModel]] = [
    InvestigationPlan,
    SpecialistFindings,
    HypothesisSet,
    CriticVerdict,
    ReportDraft,
    BaselineConclusion,
]


# --------------------------------------------------------------------------- final report
class RunStats(_Strict):
    system: str
    rounds: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    budget_exhausted: bool = False
    degraded_nodes: list[str] = []


class InvestigationReport(_Strict):
    incident_id: str
    system: str
    outcome: Outcome
    summary: str
    ranked_hypotheses: list[Hypothesis]
    facts: list[Claim]
    inferences: list[Claim]
    assumptions: list[Claim]
    data_gaps: list[str]
    recommended_actions: list[RecommendedAction]
    critic_issues: list[str] = []
    evidence: list[Evidence]
    stats: RunStats
