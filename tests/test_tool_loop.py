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


async def test_per_task_quota_caps_calls_and_rejects_excess(store, scenario):
    # One turn asks for 5 calls with a quota of 3: 3 run, 2 are refused without spending global budget.
    calls = [tc("query_metrics", {"service": s, "metric": "latency_p99"}, str(i))
             for i, s in enumerate(["payments", "checkout", "api-gateway", "auth", "cache"])]
    llm = ScriptedLLM(tool_steps={"observability": [AIMessage(content="", tool_calls=calls),
                                                    AIMessage(content="", tool_calls=[calls[0]])]})
    ctx = make_ctx(store, scenario, llm)
    res = await run_tool_loop(ctx, "observability", "s", "t", ["query_metrics"], round_=1, max_turns=5,
                              max_calls=3)
    assert ctx.budget.tool_calls == 3
    refused = [m for m in res.messages if isinstance(m, ToolMessage) and "QUOTA_REACHED" in m.content]
    assert len(refused) == 2
    assert res.stopped_reason == "quota"
    assert res.turns == 1  # no further LLM turn once the quota is spent


async def test_no_quota_by_default(store, scenario):
    calls = [tc("query_metrics", {"service": s, "metric": "latency_p99"}, str(i))
             for i, s in enumerate(["payments", "checkout", "api-gateway", "auth"])]
    llm = ScriptedLLM(tool_steps={"observability": [AIMessage(content="", tool_calls=calls),
                                                    AIMessage(content="done")]})
    ctx = make_ctx(store, scenario, llm)
    res = await run_tool_loop(ctx, "observability", "s", "t", ["query_metrics"], round_=1, max_turns=5)
    assert ctx.budget.tool_calls == 4 and res.stopped_reason == "final"


async def test_specialist_applies_budget_quota_and_tells_the_model(store, scenario):
    from rootcause_agents.agents.specialists import run_specialist
    from rootcause_agents.schemas import SpecialistFindings, Task

    calls = [tc("query_metrics", {"service": s, "metric": "latency_p99"}, str(i))
             for i, s in enumerate(["payments", "checkout", "api-gateway", "auth", "cache"])]
    llm = ScriptedLLM(tool_steps={"observability": [AIMessage(content="", tool_calls=calls)]},
                      structured={"observability": [SpecialistFindings(key_observations=[], timeline=[], gaps=[])]})
    ctx = make_ctx(store, scenario, llm, Budget(max_tool_calls_per_task=2))
    task = Task(agent="observability", objective="check latency", services=["payments"], rationale="r")
    out = await run_specialist(ctx, "observability", task, 1)
    assert ctx.budget.tool_calls == 2 and len(out.evidence) == 2
    assert "2 tool calls" in llm.calls[0]["messages"][1].content
    assert not any("stopped early" in g for g in out.gaps)  # a spent quota is normal, not a gap
