"""The synthetic dataset must be valid, diverse, and solvable (key signals detectable by the tools)."""

from collections import Counter
from pathlib import Path

import pytest

from rootcause_agents.budget import Budget, BudgetTracker
from rootcause_agents.sim.generator import generate
from rootcause_agents.sim.scenario import load_history, load_scenario, load_topology
from rootcause_agents.tools.executor import ToolExecutor
from rootcause_agents.tools.faults import FaultInjector

DATA = Path(__file__).resolve().parents[1] / "data"
SCENARIOS = sorted((DATA / "scenarios").glob("*.yaml"))
BACKGROUND = ("request completed", "health check", "slow request")


def test_at_least_ten_scenarios_with_required_traits():
    assert len(SCENARIOS) >= 10
    traits = Counter(t for p in SCENARIOS for t in load_scenario(p).traits)
    for needed in ("misleading", "conflicting", "tool_failure", "inconclusive"):
        assert traits[needed] >= 1, needed


def test_history_loads():
    assert len(load_history(DATA / "history/incidents.yaml")) >= 10


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_scenario_is_consistent(path):
    sc = load_scenario(path)
    topo = load_topology(DATA / "topology.yaml")
    services = set(topo.services)
    assert sc.ground_truth.root_cause_component in services
    assert set(sc.incident.affected_services) <= services
    for s in sc.signals:
        assert s.service in services, s
    for d in sc.deployments:
        assert d.service in services
    for c in sc.config_changes:
        assert c.service in services
    assert sc.ground_truth.expected_outcomes


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
async def test_key_signals_are_detectable(path):
    """Without faults, every declared key signal must be discoverable through the tools."""
    sc = load_scenario(path)
    topo = load_topology(DATA / "topology.yaml")
    store = generate(sc, topo, load_history(DATA / "history/incidents.yaml"))
    ex = ToolExecutor(store, FaultInjector([]), BudgetTracker(Budget(max_tool_calls=100)), sleep_scale=0)
    for ks in sc.ground_truth.key_signals:
        if ks.kind == "metric":
            res, _ = await ex.execute("query_metrics", {"service": ks.service, "metric": ks.metric}, "t", 1)
            assert res.status == "ok", (ks, res.error)
            assert res.data["change"]["changed"], f"{sc.id}: {ks.service}/{ks.metric} change not detected"
        elif ks.kind == "log":
            res, _ = await ex.execute("query_logs", {"service": ks.service}, "t", 1)
            novel = [c for c in res.data["clusters"] if not any(b in c["template"] for b in BACKGROUND)]
            assert novel and novel[0]["count"] >= 100, f"{sc.id}: no strong log signal on {ks.service}"
        elif ks.kind == "deployment":
            res, _ = await ex.execute("get_deployments", {"service": ks.service}, "t", 1)
            assert res.data["deployments"], f"{sc.id}: deployment on {ks.service} missing"
        elif ks.kind == "config":
            res, _ = await ex.execute("get_config_changes", {"service": ks.service}, "t", 1)
            assert res.data["config_changes"], f"{sc.id}: config change on {ks.service} missing"
