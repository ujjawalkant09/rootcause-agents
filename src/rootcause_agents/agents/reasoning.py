"""Reasoning agent: competing hypotheses with code-validated citations (retry, then sanitize)."""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from rootcause_agents.agents.common import (
    fmt_evidence,
    fmt_findings,
    fmt_gaps,
    fmt_incident,
    fmt_verdicts,
    system_prompt,
)
from rootcause_agents.context import RunContext
from rootcause_agents.schemas import CriticVerdict, Evidence, Hypothesis, HypothesisSet, SpecialistFindings
from rootcause_agents.validation import sanitize_hypotheses, validate_hypotheses


def renumber(hyps: list[Hypothesis]) -> list[Hypothesis]:
    ranked = sorted(hyps, key=lambda h: h.confidence, reverse=True)
    return [h.model_copy(update={"id": f"H{i}"}) for i, h in enumerate(ranked, 1)]


async def reason(ctx: RunContext, evidence: list[Evidence], findings: list[SpecialistFindings],
                 verdicts: list[CriticVerdict], gaps: list[str]) -> HypothesisSet:
    known = {e.id for e in evidence}
    user = "\n\n".join([
        fmt_incident(ctx.incident), fmt_evidence(evidence), fmt_findings(findings), fmt_gaps(gaps),
        fmt_verdicts(verdicts),
        "Produce the ranked competing hypotheses now. Address every issue raised in previous critiques.",
    ])
    messages: list[BaseMessage] = [SystemMessage(system_prompt(ctx, "reasoning")), HumanMessage(user)]
    retries = ctx.budget.budget.max_reasoning_retries
    result: HypothesisSet | None = None
    for attempt in range(retries + 1):
        result = await ctx.llm.structured("reasoning", messages, HypothesisSet)
        errors = validate_hypotheses(result.hypotheses, known, min_count=2)
        ctx.tracer.event("hypotheses", attempt=attempt + 1, errors=errors,
                         hypotheses=[(h.root_cause_component, h.root_cause_category, h.confidence)
                                     for h in result.hypotheses])
        if not errors:
            break
        messages = [*messages, HumanMessage("Your previous answer failed validation:\n- " + "\n- ".join(errors) + f"\nPrevious answer: {result.model_dump_json()}\nReturn a corrected answer.")]
    assert result is not None
    hyps = sanitize_hypotheses(result.hypotheses, known)
    return result.model_copy(update={"hypotheses": renumber(hyps)})
