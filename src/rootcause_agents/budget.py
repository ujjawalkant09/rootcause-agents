"""Execution limits, termination policy and the code-level outcome decision."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel

from rootcause_agents.schemas import CriticVerdict, Hypothesis, Outcome, RoleUsage

IDENTIFIED_MIN_CONFIDENCE = 0.75
IDENTIFIED_MIN_MARGIN = 0.20
PROBABLE_MIN_CONFIDENCE = 0.50


class Budget(BaseModel):
    max_rounds: int = 3
    max_tool_calls: int = 40
    max_tool_turns_per_agent: int = 6
    max_total_tokens: int = 400_000
    max_wall_clock_s: float = 420.0
    max_reasoning_retries: int = 2
    max_tasks_per_round: int = 4


class BudgetTracker:
    """Mutable per-run counters checked by the executor, agents and routers."""

    def __init__(self, budget: Budget):
        self.budget = budget
        self.started = time.monotonic()
        self.tool_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_calls = 0
        self.cached_tokens = 0
        self.cost_usd = 0.0
        self.by_role: dict[str, RoleUsage] = {}

    # -- accounting
    def add_usage(self, role: str, usage: Any, cost: float = 0.0) -> None:
        """``usage`` has input_tokens / output_tokens / cached_tokens (see llm.Usage)."""
        self.llm_calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cached_tokens += usage.cached_tokens
        self.cost_usd += cost
        r = self.by_role.setdefault(role, RoleUsage(role=role))
        r.calls += 1
        r.input_tokens += usage.input_tokens
        r.output_tokens += usage.output_tokens
        r.cached_tokens += usage.cached_tokens
        r.cost_usd = round(r.cost_usd + cost, 6)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    # -- checks
    def can_call_tool(self) -> bool:
        return self.tool_calls < self.budget.max_tool_calls and not self.exhausted()

    def exhausted_reason(self) -> str | None:
        if self.total_tokens >= self.budget.max_total_tokens:
            return f"tokens {self.total_tokens}/{self.budget.max_total_tokens}"
        if self.elapsed() >= self.budget.max_wall_clock_s:
            return f"wall clock {self.elapsed():.0f}s/{self.budget.max_wall_clock_s:.0f}s"
        if self.tool_calls >= self.budget.max_tool_calls:
            return f"tool calls {self.tool_calls}/{self.budget.max_tool_calls}"
        return None

    def exhausted(self) -> bool:
        return self.exhausted_reason() is not None


def should_continue(verdict: CriticVerdict, round_: int, budget: BudgetTracker, new_evidence: int) -> bool:
    """Loop back for more evidence only when it is requested, allowed and making progress."""
    return (
        verdict.decision == "NEED_MORE_EVIDENCE"
        and round_ < budget.budget.max_rounds
        and not budget.exhausted()
        and new_evidence > 0
    )


def decide_outcome(verdict: CriticVerdict | None, hypotheses: list[Hypothesis], limits_hit: bool) -> Outcome:
    """Deterministic outcome policy – the LLM never declares the outcome itself."""
    if not hypotheses:
        return "INCONCLUSIVE"
    ranked = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)
    top = ranked[0]
    second = ranked[1].confidence if len(ranked) > 1 else 0.0
    if top.root_cause_category == "unknown":
        return "INCONCLUSIVE"
    if verdict is not None and verdict.decision == "INCONCLUSIVE":
        return "INCONCLUSIVE"
    if verdict is not None and verdict.decision == "ACCEPT" and not limits_hit:
        if top.confidence >= IDENTIFIED_MIN_CONFIDENCE and top.confidence - second >= IDENTIFIED_MIN_MARGIN:
            return "ROOT_CAUSE_IDENTIFIED"
        return "PROBABLE"
    return "PROBABLE" if top.confidence >= PROBABLE_MIN_CONFIDENCE else "INCONCLUSIVE"
