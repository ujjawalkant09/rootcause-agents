import asyncio
from datetime import datetime, timedelta

import pytest

from rootcause_agents.budget import Budget, BudgetTracker
from rootcause_agents.sim.scenario import HistoricalIncident
from rootcause_agents.tools.executor import ToolExecutor
from rootcause_agents.tools.faults import FaultInjector


def make_executor(store, scenario, max_tool_calls=50):
    return ToolExecutor(
        store=store,
        injector=FaultInjector(scenario.faults),
        budget=BudgetTracker(Budget(max_tool_calls=max_tool_calls)),
        sleep_scale=0.0,
    )


async def test_query_logs_clusters_errors(store, scenario):
    ex = make_executor(store, scenario)
    res, ev = await ex.execute("query_logs", {"service": "payments", "level": "ERROR"}, agent="obs", round=1)
    assert res.status == "ok"
    top = res.data["clusters"][0]
    assert "ConnectionPoolTimeout" in top["template"]
    assert top["count"] > 100
    assert ev is not None and ev.id == "E-1" and ev.source_tool == "query_logs"
    assert "ConnectionPoolTimeout" in ev.summary


async def test_query_logs_default_levels_exclude_info(store, scenario):
    ex = make_executor(store, scenario)
    res, _ = await ex.execute("query_logs", {"service": "checkout"}, agent="obs", round=1)
    assert all(c["level"] in ("ERROR", "WARN") for c in res.data["clusters"])


async def test_metrics_timeout_is_retried_then_succeeds(store, scenario):
    ex = make_executor(store, scenario)
    res, ev = await ex.execute(
        "query_metrics", {"service": "payments-db", "metric": "db_connections"}, agent="obs", round=1
    )
    assert res.status == "ok"
    assert res.attempts == 3
    assert res.data["change"]["changed"] is True
    onset = datetime.fromisoformat(res.data["change"]["onset"])
    assert abs((onset - (scenario.incident.reported_at - timedelta(minutes=18))).total_seconds()) <= 180
    assert ev is not None


async def test_unknown_metric_returns_bad_input(store, scenario):
    ex = make_executor(store, scenario)
    res, ev = await ex.execute("query_metrics", {"service": "checkout", "metric": "db_connections"}, "obs", 1)
    assert res.status == "error" and res.error.code == "BAD_INPUT"
    assert "latency_p99" in res.error.message
    assert ev is None


async def test_invalid_args_return_bad_input(store, scenario):
    ex = make_executor(store, scenario)
    res, _ = await ex.execute("query_logs", {"svc": "x"}, "obs", 1)
    assert res.error.code == "BAD_INPUT"
    res, _ = await ex.execute("no_such_tool", {}, "obs", 1)
    assert res.error.code == "BAD_INPUT"


async def test_unavailable_fault_records_gap(store, scenario):
    ex = make_executor(store, scenario)
    res, ev = await ex.execute("search_incident_history", {"query": "checkout latency"}, "history", 1)
    assert res.status == "error" and res.error.code == "UNAVAILABLE"
    assert ev is None
    assert any("search_incident_history" in g for g in ex.data_gaps)


async def test_partial_and_stale_faults(store, scenario):
    from rootcause_agents.sim.scenario import FaultSpec

    ex = ToolExecutor(
        store=store,
        injector=FaultInjector([
            FaultSpec(tool="query_logs", fault="partial"),
            FaultSpec(tool="query_metrics", fault="stale", stale_minutes=20),
        ]),
        budget=BudgetTracker(Budget()),
        sleep_scale=0.0,
    )
    res, ev = await ex.execute("query_logs", {"service": "payments", "level": "ERROR"}, "obs", 1)
    assert res.status == "partial" and res.completeness < 1 and ev.completeness < 1
    res, _ = await ex.execute("query_metrics", {"service": "checkout", "metric": "latency_p99"}, "obs", 1)
    assert res.status == "stale" and res.warnings


async def test_deployments_and_dependencies(store, scenario):
    ex = make_executor(store, scenario)
    res, _ = await ex.execute("get_deployments", {}, "dep", 1)
    assert [d["id"] for d in res.data["deployments"]] == ["D-100"]
    res, _ = await ex.execute("get_service_dependencies", {"service": "checkout", "direction": "down"}, "dep", 1)
    names = {n for n in res.data["nodes"]}
    assert {"payments", "payments-db"} <= names
    res, _ = await ex.execute("get_config_changes", {}, "dep", 1)
    assert res.status == "empty"


async def test_history_search_ranks_by_similarity(store, scenario):
    store.history = [
        HistoricalIncident(id="H-1", title="DB pool exhausted", services=["payments-db"],
                           symptoms="checkout latency connection pool timeout", root_cause="pool", resolution="r"),
        HistoricalIncident(id="H-2", title="CDN outage", services=["cdn"], symptoms="static assets 404",
                           root_cause="cdn", resolution="r"),
    ]
    ex = ToolExecutor(store=store, injector=FaultInjector([]), budget=BudgetTracker(Budget()), sleep_scale=0)
    res, _ = await ex.execute("search_incident_history", {"query": "checkout connection pool timeout"}, "h", 1)
    assert res.data["incidents"][0]["incident_id"] == "H-1"


async def test_memoisation_returns_same_evidence(store, scenario):
    ex = make_executor(store, scenario)
    _, e1 = await ex.execute("get_deployments", {"service": "checkout"}, "dep", 1)
    _, e2 = await ex.execute("get_deployments", {"service": "checkout"}, "dep", 2)
    assert e1.id == e2.id
    assert ex.budget.tool_calls == 1
    assert ex.duplicate_calls == 1


async def test_budget_exhaustion(store, scenario):
    ex = make_executor(store, scenario, max_tool_calls=1)
    await ex.execute("get_deployments", {}, "dep", 1)
    res, ev = await ex.execute("get_config_changes", {}, "dep", 1)
    assert res.error.code == "BUDGET_EXHAUSTED" and ev is None


async def test_parallel_calls_get_unique_ids(store, scenario):
    ex = make_executor(store, scenario)
    calls = [ex.execute("query_logs", {"service": s}, "obs", 1) for s in ["checkout", "payments", "api-gateway"]]
    out = await asyncio.gather(*calls)
    ids = [ev.id for _, ev in out]
    assert len(set(ids)) == 3


def test_tool_schemas_are_openai_functions(store, scenario):
    ex = make_executor(store, scenario)
    schemas = ex.tool_schemas(["query_logs", "query_metrics"])
    assert [s["function"]["name"] for s in schemas] == ["query_logs", "query_metrics"]
    assert schemas[0]["type"] == "function"
    assert "service" in schemas[0]["function"]["parameters"]["properties"]
    with pytest.raises(KeyError):
        ex.tool_schemas(["nope"])
