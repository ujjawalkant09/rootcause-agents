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
from langgraph.types import Send, interrupt
from pydantic import BaseModel

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
MAX_HUMAN_ROUNDS = 2  # a session asks the human at most this many times
HITL_MODES = ("off", "stuck", "review")


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
    # persistence + human-in-the-loop
    meta: dict[str, Any]  # how to rebuild the run context on resume (scenario, system, ticket, run id, hitl)
    snapshot: dict[str, Any]  # cumulative budget/executor counters at the last critic step
    human_inputs: Annotated[list[str], add]
    human_rounds: int
    human_action: str


def _brief(v: Any) -> Any:
    """JSON-friendly, compact view of node output for observability (evidence → id/tool/summary only)."""
    if isinstance(v, Evidence):
        return {"id": v.id, "tool": v.source_tool, "status": v.status, "summary": v.summary}
    if isinstance(v, BaseModel):
        return v.model_dump(mode="json")
    if isinstance(v, dict):
        return {k: _brief(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_brief(x) for x in v]
    return v


async def _traced(ctx: RunContext, name: str, as_type: str, round_: int, fn: Callable[[], Awaitable[dict[str, Any]]],
                  input: Any = None, output: Callable[[dict[str, Any]], Any] = _brief) -> dict[str, Any]:
    """Run a node inside its own typed Langfuse observation (stable name; round in metadata)."""
    with ctx.telemetry.observe(name, as_type, input=_brief(input), metadata={"round": round_}) as obs:
        out = await fn()
        obs.update(output=output(out))
        return out


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


def run_snapshot(ctx: RunContext) -> dict[str, Any]:
    ex = ctx.executor
    return {"budget": ctx.budget.snapshot(), "next_evidence_id": ex._next_id, "data_gaps": list(ex.data_gaps),
            "tool_errors": ex.errors, "degraded_nodes": list(ctx.degraded_nodes)}


def build_graph(ctx: RunContext, hitl: str = "off", checkpointer: Any = None):
    async def orchestrator(state: InvestigationState) -> dict[str, Any]:
        r = state.get("round", 0) + 1
        evidence = state.get("evidence", [])
        verdicts = state.get("verdicts", [])

        async def go() -> dict[str, Any]:
            tasks = await plan(ctx, r, evidence, verdicts)
            ctx.tracer.event("round_start", round=r, tasks=len(tasks))
            return {"round": r, "plan": tasks, "evidence_at_round_start": len(evidence)}

        return await _traced(ctx, "plan-investigation", "agent", r, go,
                             input={"evidence_so_far": len(evidence),
                                    "critic_requests": verdicts[-1].evidence_requests if verdicts else []},
                             output=lambda o: {"tasks": _brief(o["plan"])})

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

            return await _traced(ctx, f"{agent}-agent", "agent", r, lambda: _resilient(
                ctx, f"{agent}(round {r})", go, lambda e: {
                    "data_gaps": [f"{agent} agent failed on '{task.objective[:60]}': {type(e).__name__}"]}),
                input=task, output=lambda o: {"findings": _brief(o.get("findings", [])),
                                              "data_gaps": o.get("data_gaps", [])})

        return node

    async def history(payload: dict[str, Any]) -> dict[str, Any]:
        async def go() -> dict[str, Any]:
            out = await run_history(ctx, payload["round"])
            return {"evidence": out.evidence, "data_gaps": out.gaps}

        # the executor already records the lookup as a `retriever` observation; no extra dispatch node
        return await _resilient(ctx, "history", go, lambda e: {"data_gaps": ["history lookup failed"]})

    async def reasoning(state: InvestigationState) -> dict[str, Any]:
        async def go() -> dict[str, Any]:
            hs = await reason(ctx, state.get("evidence", []), state.get("findings", []), state.get("verdicts", []),
                              state.get("data_gaps", []) + ctx.executor.data_gaps)
            return {"hypotheses": hs.hypotheses}

        return await _traced(ctx, "generate-hypotheses", "agent", state["round"], lambda: _resilient(
            ctx, f"reasoning(round {state['round']})", go, lambda e: {
                "hypotheses": state.get("hypotheses", []),
                "data_gaps": ["reasoning agent failed; kept previous hypotheses"]}),
            input={"evidence": len(state.get("evidence", [])), "findings": len(state.get("findings", []))})

    async def critic(state: InvestigationState) -> dict[str, Any]:
        async def go() -> dict[str, Any]:
            v = await critique(ctx, state["round"], state.get("hypotheses", []), state.get("evidence", []),
                               state.get("data_gaps", []) + ctx.executor.data_gaps, state.get("verdicts", []))
            return {"verdicts": [v], "snapshot": run_snapshot(ctx)}

        return await _traced(ctx, "critique-hypotheses", "evaluator", state["round"], lambda: _resilient(
            ctx, f"critic(round {state['round']})", go, lambda e: {"verdicts": [CriticVerdict(
                decision="NEED_MORE_EVIDENCE", issues=["critic unavailable: hypotheses were not reviewed"])]}),
            input={"hypotheses": state.get("hypotheses", [])}, output=lambda o: _brief(o["verdicts"][-1]))

    def after_critic(state: InvestigationState) -> str:
        v = state["verdicts"][-1]
        new = len(state.get("evidence", [])) - state.get("evidence_at_round_start", 0)
        go_on = bool(v.evidence_requests) and should_continue(v, state["round"], ctx.budget, new)
        ctx.tracer.event("route", after="critic", decision=v.decision, new_evidence=new, loop=go_on)
        if go_on:
            return "orchestrator"
        asks = state.get("human_rounds", 0)
        if asks < MAX_HUMAN_ROUNDS and (hitl == "review" or (hitl == "stuck" and v.decision != "ACCEPT")):
            ctx.tracer.event("human_question", reason="review" if v.decision == "ACCEPT" else "stuck",
                             round=state["round"])
            return "human_review"
        return "reporter"

    async def human_review(state: InvestigationState) -> dict[str, Any]:
        """Pause for the on-call engineer (LangGraph interrupt). Re-runs from the top on resume, so no side
        effects before ``interrupt``."""
        v = state["verdicts"][-1]
        hyps = state.get("hypotheses", [])
        reason = "review" if v.decision == "ACCEPT" else "stuck"
        question = ("Please review the conclusion before the report is written. Approve it, or add context and "
                    "ask for another round." if reason == "review" else
                    f"The investigation is stuck (critic: {v.decision}). Do you know anything that could help, e.g. a "
                    "recent change or a suspicious service? Add a hint for another round, or write the report now.")
        payload = {"reason": reason, "question": question, "round": state["round"], "critic_decision": v.decision,
                   "critic_issues": list(v.issues),
                   "evidence_requests": [r.question for r in v.evidence_requests],
                   "hypotheses": [{"component": h.root_cause_component, "category": h.root_cause_category,
                                   "confidence": round(h.confidence, 2), "title": h.title} for h in hyps[:3]],
                   "human_rounds_left": MAX_HUMAN_ROUNDS - state.get("human_rounds", 0)}
        answer = interrupt(payload) or {}
        action = "continue" if answer.get("action") == "continue" else "report"
        hint = str(answer.get("hint") or "").strip()[:1000]
        ctx.tracer.event("human_answer", action=action, hint=hint, round=state["round"])
        with ctx.telemetry.observe("ask-human", "span", input=payload, metadata={"round": state["round"]}) as obs:
            obs.update(output={"action": action, "hint": hint})
        update: dict[str, Any] = {"human_rounds": state.get("human_rounds", 0) + 1, "human_action": action}
        if hint:
            update["human_inputs"] = [hint]
            ctx.human_inputs.append(hint)
        if action == "continue":
            grant_human_round(ctx, state["round"])
        return update

    def after_human(state: InvestigationState) -> str:
        return "orchestrator" if state.get("human_action") == "continue" else "reporter"

    async def reporter(state: InvestigationState) -> dict[str, Any]:
        verdicts = state.get("verdicts", [])
        v = verdicts[-1] if verdicts else None
        hyps = state.get("hypotheses", [])
        evidence = state.get("evidence", [])
        gaps = state.get("data_gaps", [])
        outcome = decide_outcome(v, hyps, limits_hit=v is None or v.decision == "NEED_MORE_EVIDENCE")

        async def go() -> dict[str, Any]:
            draft = await write_report(ctx, outcome, hyps, evidence, gaps + ctx.executor.data_gaps, v)
            report = assemble(ctx, outcome, hyps, draft, evidence, gaps, verdicts, rounds=state["round"])
            ctx.tracer.event("report", outcome=outcome, top=hyps[0].root_cause_component if hyps else None)
            return {"report": report}

        return await _traced(ctx, "write-report", "agent", state["round"], go,
                             input={"outcome_decided_by_policy": outcome, "hypotheses": len(hyps)},
                             output=lambda o: {"outcome": o["report"].outcome, "summary": o["report"].summary})

    g = StateGraph(InvestigationState)
    g.add_node("orchestrator", orchestrator)
    g.add_node("observability", specialist("observability"))
    g.add_node("dependency", specialist("dependency"))
    g.add_node("history", history)
    g.add_node("reasoning", reasoning)
    g.add_node("critic", critic)
    g.add_node("reporter", reporter)
    g.add_node("human_review", human_review)
    g.add_edge(START, "orchestrator")
    g.add_conditional_edges("orchestrator", fan_out, ["observability", "dependency", "history", "reasoning"])
    for n in ("observability", "dependency", "history"):
        g.add_edge(n, "reasoning")
    g.add_edge("reasoning", "critic")
    g.add_conditional_edges("critic", after_critic, ["orchestrator", "reporter", "human_review"])
    g.add_conditional_edges("human_review", after_human, ["orchestrator", "reporter"])
    g.add_edge("reporter", END)
    return g.compile(checkpointer=checkpointer)


def grant_human_round(ctx: RunContext, current_round: int) -> None:
    """A human 'continue' buys exactly one more round and enough tool budget for it."""
    b = ctx.budget.budget
    b.max_rounds = max(b.max_rounds, current_round + 1)
    b.max_tool_calls = max(b.max_tool_calls, ctx.budget.tool_calls + 2 * b.max_tool_calls_per_task)
