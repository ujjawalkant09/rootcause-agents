from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_scenario_path() -> Path:
    return FIXTURES / "scenario_basic.yaml"


@pytest.fixture
def scenario(fixture_scenario_path):
    from rootcause_agents.sim.scenario import load_scenario

    return load_scenario(fixture_scenario_path)


@pytest.fixture
def topology():
    from rootcause_agents.sim.scenario import load_topology

    return load_topology(FIXTURES / "topology.yaml")


@pytest.fixture
def store(scenario, topology):
    from rootcause_agents.sim.generator import generate

    return generate(scenario, topology, history=[])
