"""Per-investigation runtime context shared by all agents (never stored in graph state)."""

from __future__ import annotations

from dataclasses import dataclass, field

from rootcause_agents.budget import BudgetTracker
from rootcause_agents.llm import LLM
from rootcause_agents.schemas import IncidentReport
from rootcause_agents.sim.store import TelemetryStore
from rootcause_agents.tools.executor import ToolExecutor
from rootcause_agents.tracing import Tracer


@dataclass
class RunContext:
    incident: IncidentReport
    store: TelemetryStore
    executor: ToolExecutor
    llm: LLM
    budget: BudgetTracker
    tracer: Tracer
    system: str
    degraded_nodes: list[str] = field(default_factory=list)
