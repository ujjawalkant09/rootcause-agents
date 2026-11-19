"""Observability / Dependency specialists (tool-using) and the deterministic history lookup."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

from rootcause_agents.agents.common import fmt_incident, load_prompt, system_prompt
from rootcause_agents.agents.tool_loop import run_tool_loop
from rootcause_agents.context import RunContext
from rootcause_agents.schemas import Claim, Evidence, SpecialistFindings, Task

SPECIALIST_TOOLS: dict[str, list[str]] = {
    "observability": ["query_logs", "query_metrics"],
    "dependency": ["get_deployments", "get_config_changes", "get_service_dependencies"],
}


@dataclass
class SpecialistOutput:
    evidence: list[Evidence] = field(default_factory=list)
    findings: list[SpecialistFindings] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)


def _clean_findings(f: SpecialistFindings, known: set[str]) -> SpecialistFindings:
    def clean(cs: list[Claim]) -> list[Claim]:
        return [c.model_copy(update={"evidence_ids": [e for e in c.evidence_ids if e in known]}) for c in cs]

    return f.model_copy(update={"key_observations": clean(f.key_observations), "timeline": clean(f.timeline)})


async def run_specialist(ctx: RunContext, agent: str, task: Task, round_: int) -> SpecialistOutput:
    quota = ctx.budget.budget.max_tool_calls_per_task
    user = "\n\n".join([
        fmt_incident(ctx.incident),
        f"Round: {round_}",
        f"YOUR TASK: {task.objective}\nFocus services: {', '.join(task.services)}"
        + (f"\nWhy: {task.rationale}" if task.rationale else ""),
        f"Quota: at most {quota} tool calls for this task. Spend them on the highest-value queries "
        "(follow the dependency path from the affected services; do not sweep every service).",
    ])
    loop = await run_tool_loop(ctx, agent, system_prompt(ctx, agent), user, SPECIALIST_TOOLS[agent], round_,
                               max_turns=ctx.budget.budget.max_tool_turns_per_agent, max_calls=quota)
    out = SpecialistOutput(evidence=loop.evidence)
    if loop.stopped_reason not in ("final", "quota"):
        out.gaps.append(f"{agent} stopped early ({loop.stopped_reason}) while working on: {task.objective[:80]}")
    if not loop.evidence:
        out.gaps.append(f"{agent} collected no evidence for: {task.objective[:80]}")
        return out
    known = set(ctx.executor.evidence)
    try:
        f = await ctx.llm.structured(agent, [*loop.messages, HumanMessage(load_prompt("findings"))],
                                     SpecialistFindings)
        out.findings.append(_clean_findings(f, known))
        out.gaps.extend(f.gaps)
    except Exception as e:  # findings are a compression step; the raw evidence still counts
        ctx.tracer.event("node_error", node=f"{agent}.findings", error=f"{type(e).__name__}: {e}")
        out.findings.append(SpecialistFindings(
            key_observations=[Claim(text=ev.summary, kind="fact", evidence_ids=[ev.id]) for ev in loop.evidence]))
    ctx.tracer.event("findings", agent=agent, round=round_, evidence=[e.id for e in loop.evidence],
                     stopped=loop.stopped_reason, turns=loop.turns)
    return out


async def run_history(ctx: RunContext, round_: int) -> SpecialistOutput:
    """Deterministic (no LLM) lookup of similar incidents – cheap prior information."""
    inc = ctx.incident
    query = f"{inc.title}. {inc.description} " + " ".join(inc.symptoms)
    result, ev = await ctx.executor.execute("search_incident_history",
                                            {"query": query, "services": inc.affected_services},
                                            agent="history", round=round_)
    out = SpecialistOutput()
    if ev is not None:
        out.evidence.append(ev)
    else:
        out.gaps.append(f"Incident history unavailable: {result.error.code if result.error else result.status}")
    return out
