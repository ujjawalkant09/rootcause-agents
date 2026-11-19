"""Critic agent: challenges hypotheses and requests targeted evidence."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from rootcause_agents.agents.common import (
    fmt_evidence,
    fmt_gaps,
    fmt_hypotheses,
    fmt_incident,
    fmt_verdicts,
    system_prompt,
)
from rootcause_agents.context import RunContext
from rootcause_agents.schemas import CriticVerdict, Evidence, Hypothesis

MAX_REQUESTS = 3


async def critique(ctx: RunContext, round_: int, hypotheses: list[Hypothesis], evidence: list[Evidence],
                   gaps: list[str], verdicts: list[CriticVerdict]) -> CriticVerdict:
    b = ctx.budget
    final_round = round_ >= b.budget.max_rounds or b.exhausted()
    user = "\n\n".join([
        fmt_incident(ctx.incident), fmt_hypotheses(hypotheses), fmt_evidence(evidence), fmt_gaps(gaps),
        fmt_verdicts(verdicts),
        f"Round: {round_} of max {b.budget.max_rounds}. Tool calls used {b.tool_calls}/{b.budget.max_tool_calls}."
        + (" THIS IS THE FINAL ROUND: decide ACCEPT or INCONCLUSIVE." if final_round else ""),
    ])
    v = await ctx.llm.structured("critic", [SystemMessage(system_prompt(ctx, "critic")), HumanMessage(user)], CriticVerdict)
    v = v.model_copy(update={"evidence_requests": v.evidence_requests[:MAX_REQUESTS]})
    ctx.tracer.event("verdict", round=round_, decision=v.decision, issues=v.issues,
                     requests=[r.question for r in v.evidence_requests])
    return v
