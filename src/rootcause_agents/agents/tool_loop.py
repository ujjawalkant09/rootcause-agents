"""Bounded tool-calling loop shared by the specialist agents and the single-agent baseline."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from rootcause_agents.context import RunContext
from rootcause_agents.schemas import Evidence
from rootcause_agents.tools.executor import result_for_llm


@dataclass
class ToolLoopResult:
    messages: list[BaseMessage]
    evidence: list[Evidence] = field(default_factory=list)
    turns: int = 0
    stopped_reason: str = "final"
    final_text: str = ""


async def run_tool_loop(ctx: RunContext, role: str, system_prompt: str, user_prompt: str, allowed_tools: list[str],
                        round_: int, max_turns: int, max_calls: int | None = None) -> ToolLoopResult:
    """Run tool turns until the model stops, ``max_turns``/``max_calls`` is reached, or the budget runs out.

    ``max_calls`` is a per-task quota: calls beyond it are refused (no global budget is spent) so one
    agent cannot drain the run's tool budget before the critic can ask for follow-up evidence.
    """
    schemas = ctx.executor.tool_schemas(allowed_tools)
    messages: list[BaseMessage] = [SystemMessage(system_prompt), HumanMessage(user_prompt)]
    res = ToolLoopResult(messages=messages)
    seen_ids: set[str] = set()
    used = 0

    while True:
        if res.turns >= max_turns:
            res.stopped_reason = "max_turns"
            break
        if ctx.budget.exhausted():
            res.stopped_reason = "budget"
            break
        ai: AIMessage = await ctx.llm.tool_step(role, messages, schemas)
        res.turns += 1
        messages.append(ai)
        if not ai.tool_calls:
            res.final_text = ai.content if isinstance(ai.content, str) else json.dumps(ai.content)
            res.stopped_reason = "final"
            break

        async def run_one(call: dict, allowed_by_quota: bool) -> ToolMessage:
            name = call["name"]
            if not allowed_by_quota:
                content = json.dumps({"status": "error", "error": {
                    "code": "QUOTA_REACHED", "message": f"Tool-call quota for this task ({max_calls}) is used up. "
                                                        "Summarise what you found."}})
                return ToolMessage(content=content, tool_call_id=call["id"], name=name)
            if name not in allowed_tools:
                content = json.dumps({"status": "error", "error": {
                    "code": "BAD_INPUT", "message": f"Tool {name!r} is not available to this agent. "
                                                    f"Allowed: {allowed_tools}"}})
                return ToolMessage(content=content, tool_call_id=call["id"], name=name)
            result, ev = await ctx.executor.execute(name, call.get("args") or {}, agent=role, round=round_)
            if ev is not None and ev.id not in seen_ids:
                seen_ids.add(ev.id)
                res.evidence.append(ev)
            return ToolMessage(content=result_for_llm(result, ev), tool_call_id=call["id"], name=name)

        grants = []
        for c in ai.tool_calls:
            ok = max_calls is None or used < max_calls
            used += ok and c["name"] in allowed_tools
            grants.append(ok)
        tool_messages = await asyncio.gather(*(run_one(c, g) for c, g in zip(ai.tool_calls, grants, strict=True)))
        messages.extend(tool_messages)  # all results of one turn are returned together
        if max_calls is not None and used >= max_calls:
            res.stopped_reason = "quota"
            break
        if ctx.budget.exhausted():
            res.stopped_reason = "budget"
            break
    return res
