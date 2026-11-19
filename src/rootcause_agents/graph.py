"""LangGraph wiring of the multi-agent investigation.

START → orchestrator ─Send()→ {observability ∥ dependency ∥ history} → reasoning → critic
          ↑                                                                    │
          └────────────── NEED_MORE_EVIDENCE (bounded, must make progress) ────┤
                                                                               └→ reporter → END
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from rootcause_agents.agents.critic import critique
from rootcause_agents.agents.orchestrator import plan
from rootcause_agents.agents.reasoning import reason
from rootcause_agents.agents.reporter import assemble, write_report
from rootcause_agents.agents.specialists import run_history, run_specialist
from rootcause_agents.budget import decide_outcome, should_continue
from rootcause_agents.context import RunContext
from rootcause_agents.schemas import (
    CriticVerdict,
    Evidence,
    Hypothesis,
    InvestigationReport,
    SpecialistFindings,
    Task,
)

NODE_ATTEMPTS = 2


def merge_evidence(left: list[Evidence], right: list[Evidence]) -> list[Evidence]:
    """Reducer: parallel branches append evidence; duplicates (memoised calls) are dropped by ID."""
    seen = {e.id for e in left}
    return left + [e for e in right if not (e.id in seen or seen.add(e.id))]


def add(left: list, right: list) -> list:
    return left + right


class InvestigationState(TypedDict, total=False):
    round: int
    plan: list[Task]
    evidence: Annotated[list[Evidence], merge_evidence]
    findings: Annotated[list[SpecialistFindings], add]
    data_gaps: Annotated[list[str], add]
    hypotheses: list[Hypothesis]
    verdicts: Annotated[list[CriticVerdict], add]
    evidence_at_round_start: int
    report: InvestigationReport


async def _resilient(ctx: RunContext, name: str, fn: Callable[[], Awaitable[dict[str, Any]]],
                     fallback: Callable[[Exception], dict[str, Any]]) -> dict[str, Any]:
    """Retry a node a bounded number of times; on final failure degrade gracefully instead of crashing."""
    err: Exception | None = None
    for attempt in range(1, NODE_ATTEMPTS + 1):
        try:
            return await fn()
        except Exception as e:
            err = e
            ctx.tracer.event("node_error", node=name, attempt=attempt, error=f"{type(e).__name__}: {e}")
            await asyncio.sleep(0)
    ctx.degraded_nodes.append(name)
    assert err is not None
    return fallback(err)


def build_graph(ctx: RunContext):
    async def orchestrator(state: InvestigationState) -> dict[str, Any]:
        r = state.get("round", 0) + 1
        evidence = state.get("evidence", [])
        tasks = await plan(ctx, r, evidence, state.get("verdicts", []))
        ctx.tracer.event("round_start", round=r, tasks=len(tasks))
        return {"round": r, "plan": tasks, "evidence_at_round_start": len(evidence)}

    def fan_out(state: InvestigationState) -> list[Send] | str:
        r = state["round"]
        sends = [Send(t.agent, {"task": t, "round": r}) for t in state.get("plan", [])]
        if r == 1:
            sends.append(Send("history", {"round": r}))
        return sends or "reasoning"

    def specialist(agent: str):
        async def node(payload: dict[str, Any]) -> dict[str, Any]:
            task: Task = payload["task"]
            r: int = payload["round"]

            async def go() -> dict[str, Any]:
                out = await run_specialist(ctx, agent, task, r)
                return {"evidence": out.evidence, "findings": out.findings, "data_gaps": out.gaps}

            return await _resilient(ctx, f"{agent}(round {r})", go, lambda e: {
                "data_gaps": [f"{agent} agent failed on '{task.objective[:60]}': {type(e).__name__}"]})

        return node

    async def history(payload: dict[str, Any]) -> dict[str, Any]:
        async def go() -> dict[str, Any]:
            out = await run_history(ctx, payload["round"])
            return {"evidence": out.evidence, "data_gaps": out.gaps}

        return await _resilient(ctx, "history", go, lambda e: {"data_gaps": ["history lookup failed"]})

    async def reasoning(state: InvestigationState) -> dict[str, Any]:
        async def go() -> dict[str, Any]:
            hs = await reason(ctx, state.get("evidence", []), state.get("findings", []), state.get("verdicts", []),
                              state.get("data_gaps", []) + ctx.executor.data_gaps)
            return {"hypotheses": hs.hypotheses}

        return await _resilient(ctx, f"reasoning(round {state['round']})", go, lambda e: {
            "hypotheses": state.get("hypotheses", []), "data_gaps": ["reasoning agent failed; kept previous hypotheses"]})

    async def critic(state: InvestigationState) -> dict[str, Any]:
        async def go() -> dict[str, Any]:
            v = await critique(ctx, state["round"], state.get("hypotheses", []), state.get("evidence", []),
                               state.get("data_gaps", []) + ctx.executor.data_gaps, state.get("verdicts", []))
            return {"verdicts": [v]}

        return await _resilient(ctx, f"critic(round {state['round']})", go, lambda e: {"verdicts": [CriticVerdict(
            decision="NEED_MORE_EVIDENCE", issues=["critic unavailable: hypotheses were not reviewed"])]})

    def after_critic(state: InvestigationState) -> str:
        v = state["verdicts"][-1]
        new = len(state.get("evidence", [])) - state.get("evidence_at_round_start", 0)
        go_on = bool(v.evidence_requests) and should_continue(v, state["round"], ctx.budget, new)
        ctx.tracer.event("route", after="critic", decision=v.decision, new_evidence=new, loop=go_on)
        return "orchestrator" if go_on else "reporter"

    async def reporter(state: InvestigationState) -> dict[str, Any]:
        verdicts = state.get("verdicts", [])
        v = verdicts[-1] if verdicts else None
        hyps = state.get("hypotheses", [])
        evidence = state.get("evidence", [])
        gaps = state.get("data_gaps", [])
        outcome = decide_outcome(v, hyps, limits_hit=v is None or v.decision == "NEED_MORE_EVIDENCE")
        draft = await write_report(ctx, outcome, hyps, evidence, gaps + ctx.executor.data_gaps, v)
        report = assemble(ctx, outcome, hyps, draft, evidence, gaps, verdicts, rounds=state["round"])
        ctx.tracer.event("report", outcome=outcome, top=hyps[0].root_cause_component if hyps else None)
        return {"report": report}

    g = StateGraph(InvestigationState)
    g.add_node("orchestrator", orchestrator)
    g.add_node("observability", specialist("observability"))
    g.add_node("dependency", specialist("dependency"))
    g.add_node("history", history)
    g.add_node("reasoning", reasoning)
    g.add_node("critic", critic)
    g.add_node("reporter", reporter)
    g.add_edge(START, "orchestrator")
    g.add_conditional_edges("orchestrator", fan_out, ["observability", "dependency", "history", "reasoning"])
    for n in ("observability", "dependency", "history"):
        g.add_edge(n, "reasoning")
    g.add_edge("reasoning", "critic")
    g.add_conditional_edges("critic", after_critic, ["orchestrator", "reporter"])
    g.add_edge("reporter", END)
    return g.compile()
