from langchain_core.messages import AIMessage, ToolMessage

from rootcause_agents.agents.tool_loop import run_tool_loop
from rootcause_agents.budget import Budget, BudgetTracker
from rootcause_agents.context import RunContext
from rootcause_agents.llm import ScriptedLLM
from rootcause_agents.tools.executor import ToolExecutor
from rootcause_agents.tools.faults import FaultInjector
from rootcause_agents.tracing import Tracer


def tc(name, args, id):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


def make_ctx(store, scenario, llm, budget=None):
    b = BudgetTracker(budget or Budget())
    tracer = Tracer()
    ex = ToolExecutor(store, FaultInjector(scenario.faults), b, sleep_scale=0.0, on_event=tracer.event)
    return RunContext(incident=scenario.incident_report(), store=store, executor=ex, llm=llm, budget=b,
                      tracer=tracer, system="mas")


async def test_parallel_tool_calls_then_final(store, scenario):
    llm = ScriptedLLM(tool_steps={"observability": [
        AIMessage(content="", tool_calls=[tc("query_logs", {"service": "payments", "level": "ERROR"}, "1"),
                                          tc("query_metrics", {"service": "payments", "metric": "latency_p99"}, "2")]),
        AIMessage(content="done"),
    ]})
    ctx = make_ctx(store, scenario, llm)
    res = await run_tool_loop(ctx, "observability", "sys", "task", ["query_logs", "query_metrics"], round_=1,
                              max_turns=5)
    assert len(res.evidence) == 2
    assert res.turns == 2
    tool_msgs = [m for m in res.messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 2 and "evidence_id" in tool_msgs[0].content


async def test_max_turns_bound(store, scenario):
    looping = [AIMessage(content="", tool_calls=[tc("get_deployments", {"start": f"-{60 + i}m"}, str(i))])
               for i in range(20)]
    ctx = make_ctx(store, scenario, ScriptedLLM(tool_steps={"dependency": looping}))
    res = await run_tool_loop(ctx, "dependency", "s", "t", ["get_deployments"], round_=1, max_turns=3)
    assert res.turns == 3
    assert res.stopped_reason == "max_turns"


async def test_disallowed_tool_is_rejected(store, scenario):
    llm = ScriptedLLM(tool_steps={"dependency": [
        AIMessage(content="", tool_calls=[tc("query_logs", {"service": "payments"}, "x")]),
        AIMessage(content="ok"),
    ]})
    ctx = make_ctx(store, scenario, llm)
    res = await run_tool_loop(ctx, "dependency", "s", "t", ["get_deployments"], round_=1, max_turns=3)
    assert res.evidence == []
    msg = next(m for m in res.messages if isinstance(m, ToolMessage))
    assert "not available" in msg.content
    assert ctx.budget.tool_calls == 0


async def test_budget_exhaustion_stops_loop(store, scenario):
    calls = [AIMessage(content="", tool_calls=[tc("get_deployments", {"start": f"-{90 + i}m"}, str(i))])
             for i in range(5)]
    ctx = make_ctx(store, scenario, ScriptedLLM(tool_steps={"dependency": calls}), Budget(max_tool_calls=1))
    res = await run_tool_loop(ctx, "dependency", "s", "t", ["get_deployments"], round_=1, max_turns=5)
    assert len(res.evidence) == 1
    assert res.stopped_reason == "budget"
