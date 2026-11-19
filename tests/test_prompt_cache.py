"""Prompt-cache friendliness: every agent's prompt starts with a stable prefix (system message) that does not
depend on the incident, so OpenAI's automatic prefix caching can reuse it across calls and incidents."""

import yaml
from langchain_core.messages import SystemMessage

from rootcause_agents.budget import Budget
from rootcause_agents.config import Settings
from rootcause_agents.runner import investigate
from tests.conftest import FIXTURES
from tests.helpers import make_llm


async def _system_prompts(scenario_path, tmp_path):
    llm = make_llm()
    await investigate(scenario_path, "mas", settings=Settings(openai_api_key="sk", budget=Budget()), llm=llm,
                      topology_path=FIXTURES / "topology.yaml", history_path=None, out_dir=tmp_path, sleep_scale=0)
    prompts: dict[str, set[str]] = {}
    for c in llm.calls:
        first = c["messages"][0]
        assert isinstance(first, SystemMessage), c["role"]
        prompts.setdefault(c["role"], set()).add(first.content)
    return prompts


async def test_system_prefix_is_stable_across_incidents(tmp_path):
    other = yaml.safe_load((FIXTURES / "scenario_basic.yaml").read_text())
    other["id"] = "T02"
    other["incident"]["title"] = "Completely different incident title"
    other["incident"]["reported_at"] = "2026-04-01T09:00:00Z"
    other_path = tmp_path / "other.yaml"
    other_path.write_text(yaml.safe_dump(other))

    a = await _system_prompts(FIXTURES / "scenario_basic.yaml", tmp_path / "a")
    b = await _system_prompts(other_path, tmp_path / "b")
    for role in a:
        assert len(a[role]) == 1, f"{role}: system prompt varies within a run"
        assert a[role] == b[role], f"{role}: system prompt depends on the incident"
        text = next(iter(a[role]))
        assert "Checkout latency high" not in text and "2026-03-10" not in text
        assert "Known services:" in text, f"{role}: static topology context should live in the cached prefix"


def test_environment_lists_service_specific_metrics(store):
    """Agents (critic included) must know which saturation metrics exist so they can request them."""
    from types import SimpleNamespace

    from rootcause_agents.agents.common import fmt_services

    text = fmt_services(SimpleNamespace(store=store))
    spec = next((n, s) for n, s in store.topology.services.items() if s.metrics)
    assert spec[0] in text and all(m in text for m in spec[1].metrics)
    assert "latency_p99" in text  # the common metrics are stated once


def test_critic_prompt_forbids_demanding_unobtainable_evidence():
    from rootcause_agents.agents.common import load_prompt

    text = load_prompt("critic").lower()
    assert "cannot provide" in text and "need_more_evidence" in text
    assert "component and failure category" in text  # finer sub-mechanism gaps are issues, not grounds to reject


def test_output_heavy_prompts_cap_their_length():
    """Output tokens cost ~5x input tokens, so the agents that write the most must be told to be brief."""
    from rootcause_agents.agents.common import load_prompt

    for name in ("reasoning", "reporter", "findings"):
        assert "one sentence per" in load_prompt(name).lower(), name
