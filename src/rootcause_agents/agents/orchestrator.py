"""Orchestrator: plans (round 1) and re-plans from critic evidence requests (later rounds)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from rootcause_agents.agents.common import fmt_evidence, fmt_incident, fmt_verdicts, system_prompt
from rootcause_agents.context import RunContext
from rootcause_agents.schemas import CriticVerdict, Evidence, InvestigationPlan, Task


def default_plan(ctx: RunContext) -> list[Task]:
    svcs = ctx.incident.affected_services
    return [
        Task(agent="observability", objective="Establish the error/latency timeline of the affected services and "
             "their dependencies; find the earliest anomaly.", services=svcs, rationale="default plan"),
        Task(agent="dependency", objective="List all recent deployments and config changes and map the dependency "
             "graph of the affected services.", services=svcs, rationale="default plan"),
    ]


def tasks_from_requests(verdict: CriticVerdict) -> list[Task]:
    return [Task(agent=r.target_agent, objective=r.question, services=r.services, rationale=r.discriminates)
            for r in verdict.evidence_requests]


def _sanitize(ctx: RunContext, tasks: list[Task], limit: int) -> list[Task]:
    known = set(ctx.store.topology.services)
    out, seen = [], set()
    for t in tasks:
        services = [s for s in t.services if s in known] or list(ctx.incident.affected_services)
        key = (t.agent, t.objective.strip().lower(), tuple(sorted(services)))
        if key in seen:
            continue
        seen.add(key)
        out.append(t.model_copy(update={"services": services}))
    return out[:limit]


async def plan(ctx: RunContext, round_: int, evidence: list[Evidence], verdicts: list[CriticVerdict]) -> list[Task]:
    limit = ctx.budget.budget.max_tasks_per_round
    last = verdicts[-1] if verdicts else None
    user = "\n\n".join([
        fmt_incident(ctx.incident),
        f"Round: {round_} of max {ctx.budget.budget.max_rounds}. Max tasks this round: {limit}. "
        f"Tool calls used: {ctx.budget.tool_calls}/{ctx.budget.budget.max_tool_calls}.",
        fmt_evidence(evidence, limit=40) if round_ > 1 else "",
        fmt_verdicts(verdicts) if round_ > 1 else "",
        ("Plan tasks that answer ONLY the critic's latest evidence requests:\n" +
         "\n".join(f"- ({r.target_agent}, services={r.services}) {r.question} [discriminates: {r.discriminates}]"
                   for r in last.evidence_requests)) if last and last.evidence_requests else
        "Create the initial investigation plan.",
    ])
    messages = [SystemMessage(system_prompt(ctx, "orchestrator")), HumanMessage(user)]
    try:
        result = await ctx.llm.structured("orchestrator", messages, InvestigationPlan)
        tasks = _sanitize(ctx, result.tasks, limit)
        ctx.tracer.event("plan", round=round_, rationale=result.rationale,
                         tasks=[t.model_dump() for t in tasks])
    except Exception as e:  # recovery: deterministic fallback plan
        ctx.tracer.event("node_error", node="orchestrator", error=f"{type(e).__name__}: {e}")
        ctx.degraded_nodes.append(f"orchestrator(round {round_})")
        tasks = _sanitize(ctx, tasks_from_requests(last) if last and round_ > 1 else default_plan(ctx), limit)
    if not tasks:
        tasks = _sanitize(ctx, tasks_from_requests(last) if last and round_ > 1 else default_plan(ctx), limit)
    return tasks
