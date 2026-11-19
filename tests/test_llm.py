from langchain_core.messages import AIMessage, HumanMessage

from rootcause_agents.config import Settings, cost_usd
from rootcause_agents.llm import OpenAILLM, ScriptedLLM, Usage, usage_from_message
from rootcause_agents.schemas import CriticVerdict


async def test_scripted_structured_and_usage():
    seen = []
    llm = ScriptedLLM(
        structured={"critic": lambda msgs, schema: schema(decision="ACCEPT", issues=[])},
        on_usage=lambda role, model, u: seen.append((role, u.input_tokens, u.output_tokens)),
    )
    v = await llm.structured("critic", [HumanMessage("x")], CriticVerdict)
    assert v.decision == "ACCEPT"
    assert seen == [("critic", 100, 50)]
    assert llm.calls[0]["role"] == "critic"


async def test_scripted_tool_steps_sequence():
    llm = ScriptedLLM(tool_steps={"obs": [AIMessage(content="a"), AIMessage(content="b")]})
    assert (await llm.tool_step("obs", [], [])).content == "a"
    assert (await llm.tool_step("obs", [], [])).content == "b"
    # exhausted script -> final answer with no tool calls
    assert (await llm.tool_step("obs", [], [])).tool_calls == []


def test_cost_from_pricing_table():
    assert cost_usd("gpt-6.2-sol", 1_000_000, 1_000_000) == 12.0
    assert cost_usd("gpt-6-luna", 1_000_000, 0) == 0.10
    assert cost_usd("unknown-model", 10, 10) == 0.0


def test_usage_from_message():
    msg = AIMessage(content="x", usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18})
    assert usage_from_message(msg) == Usage(11, 7, 0)
    assert usage_from_message(AIMessage(content="x")) == Usage()


def test_models_per_system():
    s = Settings(openai_api_key="sk-test", main_model="M", worker_model="W")
    assert s.models_for("mas") == {r: "M" for r in s.models_for("mas")}
    cheap = s.models_for("mas-cheap")
    assert cheap["observability"] == "W" and cheap["dependency"] == "W" and cheap["reasoning"] == "M"
    assert s.models_for("single")["single"] == "M"


def test_openai_llm_builds_without_network():
    s = Settings(openai_api_key="sk-test", main_model="gpt-6.2-sol", worker_model="gpt-6-luna")
    llm = OpenAILLM(s, s.models_for("mas-cheap"))
    assert llm.model_for("observability") == "gpt-6-luna"
