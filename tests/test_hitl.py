"""Persistent sessions (LangGraph checkpointer) + human-in-the-loop (interrupt / resume).

Offline tests use the in-memory checkpointer; `test_postgres_roundtrip` runs against the local Docker Postgres
(skipped when it is not reachable)."""

import os

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from rootcause_agents.budget import Budget, BudgetTracker
from rootcause_agents.config import Settings
from rootcause_agents.llm import Usage
from rootcause_agents.persistence import InMemorySessionRegistry, checkpoint_serde
from rootcause_agents.runner import resume_session, start_session
from rootcause_agents.schemas import CriticVerdict
from tests.conftest import FIXTURES
from tests.helpers import ACCEPT, MORE, make_llm

STUCK = CriticVerdict(decision="INCONCLUSIVE", issues=["cannot tell DB from deploy"])


def settings(**kw):
    return Settings(openai_api_key="sk", budget=Budget(max_rounds=1), **kw)


async def start(llm, tmp_path, cp, reg, hitl="stuck", ticket=None):
    return await start_session(FIXTURES / "scenario_basic.yaml", "mas", settings=settings(), llm=llm,
                               topology_path=FIXTURES / "topology.yaml", history_path=None, out_dir=tmp_path,
                               sleep_scale=0, hitl=hitl, checkpointer=cp, registry=reg, ticket=ticket)


async def resume(llm, thread_id, answer, cp, reg):
    return await resume_session(thread_id, answer, settings=settings(), llm=llm,
                                topology_path=FIXTURES / "topology.yaml", history_path=None,
                                checkpointer=cp, registry=reg, sleep_scale=0)


def test_budget_snapshot_restore():
    b = BudgetTracker(Budget())
    b.tool_calls = 7
    b.add_usage("critic", Usage(100, 20, 10), 0.5)
    snap = b.snapshot()
    b2 = BudgetTracker(Budget())
    b2.restore(snap)
    assert (b2.tool_calls, b2.input_tokens, b2.output_tokens, b2.cached_tokens, b2.llm_calls) == (7, 100, 20, 10, 1)
    assert b2.cost_usd == pytest.approx(0.5) and b2.by_role["critic"].calls == 1


def test_checkpoint_serde_allowlists_our_models():
    from rootcause_agents.schemas import Evidence, Hypothesis, InvestigationReport

    allowed = checkpoint_serde()._allowed_msgpack_modules
    for cls in (Evidence, Hypothesis, InvestigationReport, CriticVerdict):
        assert ("rootcause_agents.schemas", cls.__name__) in allowed


async def test_stuck_investigation_pauses_and_asks_the_human(tmp_path):
    cp, reg = InMemorySaver(serde=checkpoint_serde()), InMemorySessionRegistry()
    res = await start(make_llm(critic=[STUCK]), tmp_path, cp, reg)
    assert res.status == "waiting" and res.report is None
    q = res.question
    assert q["reason"] == "stuck" and q["critic_decision"] == "INCONCLUSIVE"
    assert q["hypotheses"][0]["component"] == "payments-db" and "cannot tell DB from deploy" in q["critic_issues"]
    sess = reg.get(res.thread_id)
    assert sess["status"] == "waiting" and sess["question"]["reason"] == "stuck"


async def test_resume_with_hint_runs_another_round_and_records_human_input(tmp_path):
    cp, reg = InMemorySaver(serde=checkpoint_serde()), InMemorySessionRegistry()
    llm1 = make_llm(critic=[STUCK])
    llm1.model = "gpt-6.1-terra"  # priced, so cost accumulates
    first = await start(llm1, tmp_path, cp, reg)
    cost_before = first.cost_usd
    assert cost_before > 0
    llm2 = make_llm(critic=[ACCEPT])  # a brand-new LLM/context: proves state comes from the checkpoint
    llm2.model = "gpt-6.1-terra"
    done = await resume(llm2, first.thread_id, {"action": "continue", "hint": "payments-db config changed at 14:00"},
                        cp, reg)
    assert done.status == "done"
    r = done.report
    assert r.human_inputs == ["payments-db config changed at 14:00"]
    assert r.stats.rounds == 2
    assert r.stats.cost_usd > cost_before  # usage is cumulative across the pause
    orch_prompts = [c["messages"][1].content for c in llm2.calls if c["role"] == "orchestrator"]
    assert orch_prompts and "payments-db config changed at 14:00" in orch_prompts[0]
    ids = [e.id for e in r.evidence]
    assert len(ids) == len(set(ids))  # evidence IDs keep counting after the resume (no clashes)
    assert reg.get(first.thread_id)["status"] == "done" and reg.get(first.thread_id)["outcome"] == r.outcome


async def test_resume_with_report_now_finishes_without_another_round(tmp_path):
    cp, reg = InMemorySaver(serde=checkpoint_serde()), InMemorySessionRegistry()
    first = await start(make_llm(critic=[STUCK]), tmp_path, cp, reg)
    done = await resume(make_llm(), first.thread_id, {"action": "report", "hint": ""}, cp, reg)
    assert done.status == "done" and done.report.stats.rounds == 1 and done.report.outcome == "INCONCLUSIVE"
    assert done.report.human_inputs == []


async def test_review_mode_asks_even_when_the_critic_accepts(tmp_path):
    cp, reg = InMemorySaver(serde=checkpoint_serde()), InMemorySessionRegistry()
    res = await start(make_llm(critic=[ACCEPT]), tmp_path, cp, reg, hitl="review")
    assert res.status == "waiting" and res.question["reason"] == "review"
    done = await resume(make_llm(), res.thread_id, {"action": "report", "hint": ""}, cp, reg)
    assert done.status == "done" and done.report.outcome != "INCONCLUSIVE"


async def test_hitl_off_never_pauses(tmp_path):
    cp, reg = InMemorySaver(serde=checkpoint_serde()), InMemorySessionRegistry()
    res = await start(make_llm(critic=[STUCK]), tmp_path, cp, reg, hitl="off")
    assert res.status == "done" and res.report.outcome == "INCONCLUSIVE"


async def test_human_rounds_are_bounded(tmp_path):
    cp, reg = InMemorySaver(serde=checkpoint_serde()), InMemorySessionRegistry()
    llm = make_llm(critic=[STUCK] * 10)
    res = await start(llm, tmp_path, cp, reg)
    asked = 1
    while res.status == "waiting":
        res = await resume(make_llm(critic=[STUCK] * 10), res.thread_id, {"action": "continue", "hint": "try again"},
                           cp, reg)
        asked += res.status == "waiting"
    assert asked <= 2 and res.status == "done"  # MAX_HUMAN_ROUNDS = 2


async def test_unknown_thread_is_a_clear_error(tmp_path):
    with pytest.raises(KeyError):
        await resume(make_llm(), "nope", {"action": "report"}, InMemorySaver(serde=checkpoint_serde()), InMemorySessionRegistry())


PG = os.environ.get("TEST_DATABASE_URL", "postgresql://rootcause:rootcause@localhost:5433/rootcause_test")


def _pg_up() -> bool:
    try:
        import psycopg

        with psycopg.connect(PG, connect_timeout=2):
            return True
    except Exception:
        return False


@pytest.mark.skipif(not _pg_up(), reason="local Postgres (docker compose up -d) not reachable")
async def test_postgres_roundtrip_survives_a_new_connection(tmp_path):
    """Pause in one connection, resume in a fresh one: the session really lives in Postgres."""
    from rootcause_agents.persistence import open_checkpointer, open_registry

    s = settings(database_url=PG)
    async with open_checkpointer(s) as cp, open_registry(s) as reg:
        first = await start(make_llm(critic=[STUCK]), tmp_path, cp, reg)
        assert first.status == "waiting"
    async with open_checkpointer(s) as cp2, open_registry(s) as reg2:
        assert (await reg2.aget(first.thread_id))["status"] == "waiting"
        done = await resume(make_llm(critic=[ACCEPT]), first.thread_id, {"action": "continue", "hint": "check db"},
                            cp2, reg2)
        assert done.status == "done" and done.report.human_inputs == ["check db"]
        assert (await reg2.aget(first.thread_id))["status"] == "done"


class FakeRerun(BaseException):
    """Mimics Streamlit's script-control exception that kills a run when the user clicks during it."""


async def test_interrupted_run_is_marked_and_can_continue_from_its_checkpoint(tmp_path):
    cp, reg = InMemorySaver(serde=checkpoint_serde()), InMemorySessionRegistry()

    def killer(ev):
        if ev["event"] == "route":
            raise FakeRerun()

    with pytest.raises(FakeRerun):
        await start_session(FIXTURES / "scenario_basic.yaml", "mas", settings=settings(), llm=make_llm(critic=[STUCK]),
                            topology_path=FIXTURES / "topology.yaml", history_path=None, out_dir=tmp_path,
                            sleep_scale=0, hitl="stuck", checkpointer=cp, registry=reg, echo=killer)
    rows = await reg.alist()
    assert rows[0]["status"] == "interrupted" and "FakeRerun" in rows[0]["error"]
    tid = rows[0]["thread_id"]

    llm2 = make_llm(critic=[STUCK])
    res = await resume_session(tid, None, settings=settings(), llm=llm2, topology_path=FIXTURES / "topology.yaml",
                               history_path=None, checkpointer=cp, registry=reg, sleep_scale=0)
    # continues from the last checkpoint: only the step that was interrupted (critic) re-runs, then it pauses
    assert res.status == "waiting" and res.question["reason"] == "stuck"
    assert [c["role"] for c in llm2.calls] == ["critic"]
    assert reg.get(tid)["status"] == "waiting" and reg.get(tid)["error"] is None


async def test_failed_run_records_the_error(tmp_path, monkeypatch):
    from rootcause_agents import graph as graph_mod

    cp, reg = InMemorySaver(serde=checkpoint_serde()), InMemorySessionRegistry()

    def boom(*a, **kw):
        raise RuntimeError("db went away")

    monkeypatch.setattr(graph_mod, "should_continue", boom)  # an error inside the graph itself
    with pytest.raises(RuntimeError):
        await start(make_llm(critic=[MORE]), tmp_path, cp, reg)
    row = (await reg.alist())[0]
    assert row["status"] == "failed" and "db went away" in row["error"]


async def test_answer_is_required_only_for_paused_sessions(tmp_path):
    cp, reg = InMemorySaver(serde=checkpoint_serde()), InMemorySessionRegistry()
    first = await start(make_llm(critic=[STUCK]), tmp_path, cp, reg)
    with pytest.raises(ValueError, match="waiting for an answer"):
        await resume(make_llm(), first.thread_id, None, cp, reg)


async def test_crash_in_routing_after_the_critic_is_recovered_on_continue(tmp_path):
    """Real-world case: the critic's writes were saved but the routing after it crashed, so a plain resume would
    end the graph with no report. Continue must re-run the routing and reach the human question."""
    from rootcause_agents.graph import build_graph  # noqa: F401  (documents which graph is exercised)
    from rootcause_agents.tracing import Tracer

    cp, reg = InMemorySaver(serde=checkpoint_serde()), InMemorySessionRegistry()
    real_event = Tracer.event

    def crashing_event(self, kind, data=None, **kw):
        real_event(self, kind, data, **kw)
        if kind == "route":
            raise RuntimeError("crash inside routing")

    Tracer.event = crashing_event
    try:
        with pytest.raises(RuntimeError):
            await start(make_llm(critic=[STUCK]), tmp_path, cp, reg)
    finally:
        Tracer.event = real_event
    tid = (await reg.alist())[0]["thread_id"]
    llm2 = make_llm(critic=[STUCK])
    res = await resume(llm2, tid, None, cp, reg)
    assert res.status == "waiting" and res.question["reason"] == "stuck"
    assert [c["role"] for c in llm2.calls if c["role"] in ("orchestrator", "reasoning")] == []


def test_echo_errors_never_break_a_run(tmp_path):
    from rootcause_agents.tracing import Tracer

    def bad_echo(ev):
        raise ValueError("ui went away")

    t = Tracer(tmp_path / "t.jsonl", echo=bad_echo)
    t.event("route", loop=False)  # must not raise
    assert t.count("route") == 1 and "route" in (tmp_path / "t.jsonl").read_text()
