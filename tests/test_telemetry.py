"""Langfuse observability: optional, one trace per investigation, LLM calls via LangChain callbacks,
tool calls / critic verdicts / fallbacks as events, outcome + cost as scores. Tested with a fake client."""

from contextlib import contextmanager
from contextvars import ContextVar

from langchain_core.messages import AIMessage, HumanMessage

from rootcause_agents.budget import Budget
from rootcause_agents.config import Settings
from rootcause_agents.llm import OpenAILLM
from rootcause_agents.runner import investigate
from rootcause_agents.telemetry import LangfuseTelemetry, NoopTelemetry, make_telemetry
from tests.conftest import FIXTURES
from tests.helpers import make_llm


class FakeSpan:
    def __init__(self, log):
        self.log = log

    def update(self, **kw):
        self.log.append(("span_update", kw))


class FakeLangfuse:
    """Tracks the current observation in a ContextVar, like OpenTelemetry does (per asyncio task)."""

    def __init__(self):
        self.log = []
        self.current: ContextVar = ContextVar("current", default=None)

    @contextmanager
    def start_as_current_observation(self, **kw):
        self.log.append(("start", {**kw, "parent": self.current.get()}))
        token = self.current.set(kw["name"])
        try:
            yield FakeSpan(self.log)
        finally:
            self.current.reset(token)
            self.log.append(("end", {}))

    def create_event(self, **kw):
        self.log.append(("event", kw))

    def create_score(self, **kw):
        self.log.append(("score", kw))

    def get_current_trace_id(self):
        return "trace-123"

    def get_trace_url(self, trace_id=None):
        return f"https://cloud.langfuse.com/trace/{trace_id}"

    def flush(self):
        self.log.append(("flush", {}))


@contextmanager
def fake_propagate(**kw):
    FAKE_PROPAGATED.append(kw)
    yield


FAKE_PROPAGATED: list = []


def lf():
    client = FakeLangfuse()
    return client, LangfuseTelemetry(client=client, propagate=fake_propagate, handler_factory=lambda: "HANDLER")


def test_disabled_without_keys():
    t = make_telemetry(Settings(openai_api_key="sk"))
    assert isinstance(t, NoopTelemetry) and not t.enabled
    assert t.callbacks() == []
    with t.investigation(run_id="r", system="mas", scenario_id="S01", title="t"):
        t.on_event({"event": "tool_call"})
    assert t.trace_url is None


def test_enabled_with_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://eu.example")  # real env vars win over .env
    s = Settings.from_env()
    assert s.langfuse_public_key == "pk-lf-x" and s.langfuse_base_url == "https://eu.example"


def test_one_trace_per_investigation_with_attributes():
    FAKE_PROPAGATED.clear()
    client, t = lf()
    with t.investigation(run_id="run-1", system="mas", scenario_id="S01", title="Checkout 5xx",
                         incident={"title": "Checkout 5xx", "symptoms": ["5xx"]}, session_id="eval-1"):
        assert t.callbacks() == ["HANDLER"]
    start = next(kw for k, kw in client.log if k == "start")
    assert start["name"] == "investigate-incident" and start["as_type"] == "agent"  # stable, low-cardinality
    assert start["input"] == {"title": "Checkout 5xx", "symptoms": ["5xx"]}
    p = FAKE_PROPAGATED[0]
    assert p["session_id"] == "eval-1" and p["trace_name"] == "investigate-incident"
    assert set(p["tags"]) >= {"mas", "S01"} and p["metadata"]["run_id"] == "run-1"
    assert t.trace_url == "https://cloud.langfuse.com/trace/trace-123"
    assert client.log[-1][0] == "flush"


def test_only_warning_events_are_forwarded():
    """Tool calls / verdicts are observations (not events); events are kept for resilience signals only."""
    client, t = lf()
    with t.investigation(run_id="r", system="mas", scenario_id="S01", title="t"):
        t.on_event({"event": "tool_call", "tool": "query_logs"})
        t.on_event({"event": "verdict", "decision": "ACCEPT", "issues": []})
        t.on_event({"event": "llm_fallback", "role": "critic", "from_model": "a", "to_model": "b"})
        t.on_event({"event": "llm_call", "role": "critic"})
    names = [kw["name"] for k, kw in client.log if k == "event"]
    assert names == ["llm_fallback"]
    lvl = next(kw for k, kw in client.log if k == "event" and kw["name"] == "llm_fallback")["level"]
    assert lvl == "WARNING"


def test_telemetry_errors_never_break_the_run():
    class Broken(FakeLangfuse):
        def create_event(self, **kw):
            raise RuntimeError("langfuse down")

    t = LangfuseTelemetry(client=Broken(), propagate=fake_propagate, handler_factory=lambda: "H")
    with t.investigation(run_id="r", system="mas", scenario_id="S01", title="t"):
        t.on_event({"event": "tool_call", "tool": "x"})  # must not raise


async def test_openai_llm_attaches_callbacks_to_the_model_not_the_chain():
    """Callbacks + name live on the chat model, so a structured-output call is ONE generation named after the agent
    (callbacks on the outer chain would also trace every parser/RunnableLambda step as noise)."""
    from rootcause_agents.schemas import CriticVerdict

    seen = []

    class Chat:
        name = None
        callbacks = None
        metadata = None

        def bind_tools(self, tools):
            return self

        def with_structured_output(self, schema, method=None, include_raw=False):
            return self

        async def ainvoke(self, messages, config=None, **kw):
            seen.append(config)
            if kw or config:
                raise AssertionError("no per-call callbacks expected")
            return {"raw": AIMessage(content=""), "parsed": CriticVerdict(decision="ACCEPT", issues=[]),
                    "parsing_error": None}

    chat = Chat()
    llm = OpenAILLM(Settings(openai_api_key="sk", main_model="m"), {"critic": "m"},
                    client_factory=lambda *a: chat, callbacks=lambda: ["CB"])
    await llm.structured("critic", [HumanMessage("x")], CriticVerdict)
    assert chat.callbacks == ["CB"] and chat.name == "critic"
    assert chat.metadata == {"agent": "critic", "model": "m", "primary_provider": True}


async def test_runner_reports_scores_and_trace_url(tmp_path):
    client, t = lf()
    report = await investigate(FIXTURES / "scenario_basic.yaml", "mas",
                               settings=Settings(openai_api_key="sk", budget=Budget()), llm=make_llm(),
                               topology_path=FIXTURES / "topology.yaml", history_path=None, out_dir=tmp_path,
                               sleep_scale=0, telemetry=t)
    scores = {kw["name"]: kw["value"] for k, kw in client.log if k == "score"}
    assert scores["outcome"] == report.outcome and "cost_usd" in scores and "top_confidence" in scores
    end = [kw for k, kw in client.log if k == "span_update"][-1]
    assert end["output"]["outcome"] == report.outcome
    starts = [kw for k, kw in client.log if k == "start"]
    by_name = {kw["name"]: kw for kw in starts}
    # every agent node is its own typed observation under the root ...
    for name, typ in [("plan-investigation", "agent"), ("observability-agent", "agent"),
                      ("dependency-agent", "agent"), ("search_incident_history", "retriever"),
                      ("generate-hypotheses", "agent"), ("critique-hypotheses", "evaluator"),
                      ("write-report", "agent")]:
        assert by_name[name]["as_type"] == typ and by_name[name]["parent"] == "investigate-incident", name
        assert "round" in (by_name[name].get("metadata") or {}), name
    # ... and tool calls are `tool` observations nested under the agent that requested them
    assert "search-incident-history" not in by_name  # no duplicate dispatch node around the retriever
    tools = [kw for kw in starts if kw["as_type"] == "tool"]
    assert tools and all(kw["parent"] in ("observability-agent", "dependency-agent") for kw in tools)
    assert all(kw["name"] in ("query_logs", "query_metrics", "get_deployments", "get_config_changes",
                              "get_service_dependencies") for kw in tools)
    assert report.stats.trace_url == "https://cloud.langfuse.com/trace/trace-123"


async def test_eval_batch_is_one_langfuse_session(tmp_path, monkeypatch):
    import rootcause_agents.evaluation as ev

    seen = []
    real = ev.investigate

    async def spy(*a, **kw):
        seen.append(kw.get("session_id"))
        return await real(*a, **kw)

    monkeypatch.setattr(ev, "investigate", spy)
    await ev.run_matrix(["mas"], [FIXTURES / "scenario_basic.yaml"] * 2, Settings(openai_api_key="sk", budget=Budget()),
                        llm_factory=lambda s: make_llm(), out_dir=tmp_path, topology_path=FIXTURES / "topology.yaml",
                        history_path=None, sleep_scale=0)
    assert len(set(seen)) == 1 and seen[0].startswith("eval-")
