"""Reporting agent + deterministic report assembly (outcome and stats are set by code)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from rootcause_agents.agents.common import fmt_evidence, fmt_gaps, fmt_hypotheses, fmt_incident, load_prompt
from rootcause_agents.context import RunContext
from rootcause_agents.schemas import (
    Claim,
    CriticVerdict,
    Evidence,
    Hypothesis,
    InvestigationReport,
    Outcome,
    RecommendedAction,
    ReportDraft,
    RunStats,
)
from rootcause_agents.validation import sanitize_report


def fallback_draft(outcome: Outcome, hyps: list[Hypothesis], evidence: list[Evidence]) -> ReportDraft:
    top = hyps[0] if hyps else None
    summary = (f"{outcome}: most likely root cause is '{top.title}' in {top.root_cause_component} "
               f"({top.root_cause_category}, confidence {top.confidence:.2f})." if top
               else f"{outcome}: no hypothesis could be established from the available evidence.")
    facts = [Claim(text=e.summary, kind="fact", evidence_ids=[e.id]) for e in evidence[:15]]
    actions = [RecommendedAction(action="Review the evidence ledger and continue the investigation manually.",
                                 type="investigate_further", priority="P1")]
    return ReportDraft(summary=summary + " (Report generated deterministically: reporting agent unavailable.)",
                       facts=facts, inferences=[], assumptions=[], recommended_actions=actions)


async def write_report(ctx: RunContext, outcome: Outcome, hyps: list[Hypothesis], evidence: list[Evidence],
                       gaps: list[str], verdict: CriticVerdict | None) -> ReportDraft:
    user = "\n\n".join([
        fmt_incident(ctx.incident), f"OUTCOME (decided by policy, do not change): {outcome}",
        fmt_hypotheses(hyps), fmt_evidence(evidence), fmt_gaps(gaps),
        f"CRITIC: {verdict.decision if verdict else 'n/a'}; issues: {'; '.join(verdict.issues) if verdict else ''}",
    ])
    try:
        draft = await ctx.llm.structured("reporter", [SystemMessage(load_prompt("reporter")), HumanMessage(user)],
                                         ReportDraft)
    except Exception as e:
        ctx.tracer.event("node_error", node="reporter", error=f"{type(e).__name__}: {e}")
        ctx.degraded_nodes.append("reporter")
        draft = fallback_draft(outcome, hyps, evidence)
    return sanitize_report(draft, {e.id for e in evidence})


def assemble(ctx: RunContext, outcome: Outcome, hyps: list[Hypothesis], draft: ReportDraft,
             evidence: list[Evidence], gaps: list[str], verdicts: list[CriticVerdict], rounds: int) -> InvestigationReport:
    b = ctx.budget
    all_gaps = list(dict.fromkeys(list(ctx.executor.data_gaps) + list(gaps)))
    stats = RunStats(
        system=ctx.system, rounds=rounds, tool_calls=b.tool_calls, tool_errors=ctx.executor.errors,
        llm_calls=b.llm_calls, input_tokens=b.input_tokens, output_tokens=b.output_tokens,
        cost_usd=round(b.cost_usd, 6), latency_s=round(b.elapsed(), 2), budget_exhausted=b.exhausted(),
        degraded_nodes=list(ctx.degraded_nodes),
    )
    issues = [i for v in verdicts for i in v.issues]
    return InvestigationReport(
        incident_id=ctx.incident.id, system=ctx.system, outcome=outcome, summary=draft.summary,
        ranked_hypotheses=hyps, facts=draft.facts, inferences=draft.inferences, assumptions=draft.assumptions,
        data_gaps=all_gaps, recommended_actions=draft.recommended_actions, critic_issues=issues,
        evidence=sorted(evidence, key=lambda e: int(e.id.split("-")[1])), stats=stats,
    )
