"""Model/provider fallback chain, circuit breaker, prompt-cache key and cached-token accounting."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from rootcause_agents.config import Settings, cost_usd
from rootcause_agents.llm import CircuitBreaker, OpenAILLM, StructuredOutputError, Usage, usage_from_message
from rootcause_agents.schemas import CriticVerdict

USAGE = {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100,
         "input_token_details": {"cache_read": 800}}


class FakeChat:
    """Stands in for ChatOpenAI. `behavior` is a list of results/exceptions consumed per call."""

    def __init__(self, model, behavior):
        self.model, self.behavior, self.calls = model, behavior, 0

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, method=None, include_raw=False):
        return self

    async def ainvoke(self, messages, **kw):
        self.calls += 1
        r = self.behavior.pop(0) if self.behavior else RuntimeError("script exhausted")
        if isinstance(r, Exception):
            raise r
        return r


def structured_ok(decision="ACCEPT"):
    return {"raw": AIMessage(content="", usage_metadata=USAGE),
            "parsed": CriticVerdict(decision=decision, issues=[]), "parsing_error": None}


def make(settings, behaviors, clock=None):
    created = []

    def factory(model, base_url, api_key, model_kwargs):
        c = FakeChat(model, behaviors.get((model, base_url), []))
        created.append({"model": model, "base_url": base_url, "api_key": api_key, "model_kwargs": model_kwargs,
                        "client": c})
        return c

    events, usage = [], []
    llm = OpenAILLM(settings, settings.models_for("mas"), on_usage=lambda r, m, u: usage.append((r, m, u)),
                    on_event=lambda k, d: events.append((k, d)), client_factory=factory,
                    clock=clock or (lambda: 0.0))
    return llm, created, events, usage


def S(**kw):
    return Settings(openai_api_key="sk-main", main_model="main-m", worker_model="worker-m", **kw)


def test_usage_includes_cached_tokens():
    u = usage_from_message(AIMessage(content="", usage_metadata=USAGE))
    assert u == Usage(input_tokens=1000, output_tokens=100, cached_tokens=800)
    assert usage_from_message(AIMessage(content="")) == Usage()


def test_cost_uses_cached_price():
    # gpt-6.2-sol: input $2, cached $0.10, output $10 per 1M
    assert cost_usd("gpt-6.2-sol", 1_000_000, 0, cached_tokens=1_000_000) == pytest.approx(0.10)
    assert cost_usd("gpt-6.2-sol", 1_000_000, 1_000_000, cached_tokens=500_000) == pytest.approx(1.0 + 0.05 + 10.0)


async def test_primary_success_reports_usage_and_cache_key():
    llm, created, events, usage = make(S(), {("main-m", None): [structured_ok()]})
    v = await llm.structured("critic", [HumanMessage("x")], CriticVerdict)
    assert v.decision == "ACCEPT"
    assert usage == [("critic", "main-m", Usage(1000, 100, 800))]
    assert created[0]["model_kwargs"] == {"prompt_cache_key": "rootcause-agents:critic"}
    assert not [e for e in events if e[0] == "llm_fallback"]


async def test_falls_back_to_next_model_on_error():
    llm, _created, events, usage = make(S(fallback_models=["backup-m"]), {
        ("main-m", None): [RuntimeError("503 overloaded")],
        ("backup-m", None): [structured_ok()],
    })
    v = await llm.structured("critic", [HumanMessage("x")], CriticVerdict)
    assert v.decision == "ACCEPT"
    assert usage[-1][1] == "backup-m"
    assert ("llm_fallback", {"role": "critic", "from_model": "main-m", "to_model": "backup-m"}) in events


async def test_unparseable_output_triggers_fallback_but_tokens_are_counted():
    bad = {"raw": AIMessage(content="", usage_metadata=USAGE), "parsed": None, "parsing_error": "bad json"}
    llm, _, _, usage = make(S(fallback_models=["backup-m"]), {
        ("main-m", None): [bad], ("backup-m", None): [structured_ok()]})
    await llm.structured("critic", [HumanMessage("x")], CriticVerdict)
    assert [m for _, m, _ in usage] == ["main-m", "backup-m"]  # failed attempt still billed


async def test_provider_fallback_uses_second_endpoint():
    s = S(fallback_base_url="https://backup.example/v1", fallback_api_key="sk-backup",
          fallback_provider_models=["prov-m"])
    llm, created, _, usage = make(s, {
        ("main-m", None): [RuntimeError("down")],
        ("prov-m", "https://backup.example/v1"): [structured_ok()],
    })
    await llm.structured("critic", [HumanMessage("x")], CriticVerdict)
    prov = next(c for c in created if c["model"] == "prov-m")
    assert prov["api_key"] == "sk-backup"
    assert prov["model_kwargs"] == {}  # no OpenAI-specific cache key sent to a foreign provider
    assert usage[-1][1] == "prov-m"


async def test_all_endpoints_fail_raises_last_error():
    llm, _, _, _ = make(S(fallback_models=["backup-m"]), {
        ("main-m", None): [RuntimeError("a")], ("backup-m", None): [StructuredOutputError("b")]})
    with pytest.raises(StructuredOutputError):
        await llm.structured("critic", [HumanMessage("x")], CriticVerdict)


async def test_circuit_breaker_skips_failing_primary():
    now = {"t": 0.0}
    behaviors = {("main-m", None): [RuntimeError("x")] * 3,
                 ("backup-m", None): [AIMessage(content="ok", usage_metadata=USAGE) for _ in range(5)]}
    llm, created, events, _ = make(S(fallback_models=["backup-m"], breaker_threshold=3, breaker_cooldown_s=60),
                                   behaviors, clock=lambda: now["t"])
    for _ in range(3):
        await llm.tool_step("observability", [HumanMessage("x")], [])
    primary = next(c["client"] for c in created if c["model"] == "main-m")
    assert primary.calls == 3
    await llm.tool_step("observability", [HumanMessage("x")], [])  # circuit open → primary skipped
    assert primary.calls == 3
    assert any(k == "circuit_open" for k, _ in events)
    now["t"] = 61.0  # cooldown elapsed → half-open, primary tried again
    primary.behavior.append(AIMessage(content="ok", usage_metadata=USAGE))
    await llm.tool_step("observability", [HumanMessage("x")], [])
    assert primary.calls == 4


def test_circuit_breaker_unit():
    t = {"now": 0.0}
    b = CircuitBreaker(threshold=2, cooldown_s=10, clock=lambda: t["now"])
    assert b.allow("m")
    b.failure("m")
    b.failure("m")
    assert not b.allow("m")
    t["now"] = 11
    assert b.allow("m")
    b.success("m")
    assert b.allow("m")


def test_settings_parse_fallbacks_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("FALLBACK_MODELS", "gpt-6.1-terra, gpt-6-luna")
    monkeypatch.setenv("FALLBACK_BASE_URL", "https://backup.example/v1")
    monkeypatch.setenv("FALLBACK_API_KEY", "sk-b")
    monkeypatch.setenv("FALLBACK_PROVIDER_MODELS", "gpt-6.2-sol")
    s = Settings.from_env()
    assert s.fallback_models == ["gpt-6.1-terra", "gpt-6-luna"]
    assert s.fallback_base_url == "https://backup.example/v1"
    assert s.fallback_provider_models == ["gpt-6.2-sol"]
