"""Single-agent baseline: one ReAct loop with ALL tools, same budget, same report schema and outcome policy."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from rootcause_agents.agents.common import fmt_incident, fmt_services, load_prompt
from rootcause_agents.agents.reasoning import renumber
from rootcause_agents.agents.reporter import assemble, fallback_draft
from rootcause_agents.agents.tool_loop import run_tool_loop
from rootcause_agents.budget import decide_outcome
from rootcause_agents.context import RunContext
from rootcause_agents.schemas import BaselineConclusion, CriticVerdict, InvestigationReport
from rootcause_agents.tools.impl import TOOLS
from rootcause_agents.validation import sanitize_hypotheses, sanitize_report, validate_hypotheses


async def run_single(ctx: RunContext) -> InvestigationReport:
    b = ctx.budget.budget
    # Same total tool-turn allowance the MAS specialists get across rounds.
    max_turns = b.max_tool_turns_per_agent * b.max_rounds
    user = "\n\n".join([fmt_incident(ctx.incident), fmt_services(ctx), "Round: 1"])
    loop = await run_tool_loop(ctx, "single", load_prompt("single_agent"), user, list(TOOLS), round_=1,
                               max_turns=max_turns)
    evidence = list(ctx.executor.evidence.values())
    known = {e.id for e in evidence}
    messages = [*loop.messages, HumanMessage(load_prompt("single_conclusion"))]
    conclusion: BaselineConclusion | None = None
    try:
        for attempt in range(b.max_reasoning_retries + 1):
            conclusion = await ctx.llm.structured("single", messages, BaselineConclusion)
            errors = validate_hypotheses(conclusion.hypotheses, known, min_count=2)
            ctx.tracer.event("hypotheses", attempt=attempt + 1, errors=errors)
            if not errors:
                break
            messages = [*messages, HumanMessage("Your answer failed validation:\n- " + "\n- ".join(errors) + "\nReturn a corrected answer.")]
    except Exception as e:
        ctx.tracer.event("node_error", node="single.conclusion", error=f"{type(e).__name__}: {e}")
        ctx.degraded_nodes.append("single.conclusion")

    hyps = renumber(sanitize_hypotheses(conclusion.hypotheses, known)) if conclusion else []
    verdict = CriticVerdict(decision="ACCEPT" if conclusion and conclusion.conclusive else "INCONCLUSIVE",
                            issues=[])
    outcome = decide_outcome(verdict, hyps, limits_hit=False)
    draft = sanitize_report(conclusion.report, known) if conclusion else fallback_draft(outcome, hyps, evidence)
    gaps = [] if loop.stopped_reason == "final" else [f"single agent stopped early ({loop.stopped_reason})"]
    return assemble(ctx, outcome, hyps, draft, evidence, gaps, [], rounds=1)
