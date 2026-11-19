"""Redesigned UI: friendly options, progress timeline from trace events, simple multipage app."""

from pathlib import Path

from rootcause_agents.ui.view import TEAM_OPTIONS, stages_from_events

ROOT = Path(__file__).resolve().parents[1]


def test_team_options_use_plain_language_and_map_to_systems():
    labels = list(TEAM_OPTIONS)
    assert labels == ["Team of agents", "Team (budget)", "Single agent"]
    assert [TEAM_OPTIONS[x]["system"] for x in labels] == ["mas", "mas-cheap", "single"]
    assert all(TEAM_OPTIONS[x]["cost"] > 0 and TEAM_OPTIONS[x]["minutes"] > 0 for x in labels)


def ev(kind, **kw):
    return {"event": kind, "t": 1.0, **kw}


def test_stages_show_progress_in_plain_words():
    events = [ev("run_start"), ev("plan", round=1, tasks=[{"agent": "observability", "objective": "timeline"}]),
              ev("tool_call", agent="observability", tool="query_logs", status="ok", params={}),
              ev("tool_error", tool="query_metrics", code="TIMEOUT", attempt=1),
              ev("tool_call", agent="dependency", tool="get_deployments", status="ok", params={})]
    st = {s["key"]: s for s in stages_from_events(events)}
    assert [s["key"] for s in stages_from_events(events)] == ["plan", "gather", "reason", "critic", "report"]
    assert st["plan"]["state"] == "complete" and "1 task" in st["plan"]["detail"]
    assert st["gather"]["state"] == "running" and "2 tool calls" in st["gather"]["detail"]
    assert "1 retry" in st["gather"]["detail"]
    assert st["reason"]["state"] == "pending" and st["report"]["state"] == "pending"


def test_stages_follow_rounds_questions_and_completion():
    events = [ev("plan", round=1, tasks=[]), ev("hypotheses", hypotheses=[["payments-db", "resource_exhaustion", 0.8]]),
              ev("verdict", decision="NEED_MORE_EVIDENCE", issues=[]), ev("route", loop=True),
              ev("plan", round=2, tasks=[{"agent": "observability", "objective": "x"}])]
    st = {s["key"]: s for s in stages_from_events(events)}
    assert st["plan"]["label"].endswith("(round 2)") and st["plan"]["state"] == "complete"
    assert st["reason"]["state"] == "pending"  # a new round restarts the later stages
    done = [*events, ev("hypotheses", hypotheses=[["payments-db", "resource_exhaustion", 0.85]]),
            ev("verdict", decision="ACCEPT", issues=[]), ev("report", outcome="ROOT_CAUSE_IDENTIFIED")]
    st = {s["key"]: s for s in stages_from_events(done)}
    assert all(s["state"] == "complete" for s in st.values())
    assert "payments-db" in st["reason"]["detail"] and "accepted" in st["critic"]["detail"].lower()
    waiting = [*events[:3], ev("human_question", reason="stuck")]
    st = {s["key"]: s for s in stages_from_events(waiting)}
    assert st["critic"]["state"] == "complete" and "your input" in st["critic"]["detail"].lower()


def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(ROOT / "src/rootcause_agents/ui/app.py"), default_timeout=30)


def test_home_is_a_simple_two_step_flow():
    at = _app().run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown) + " ".join(h.value for h in at.subheader)
    assert "What happened?" in text and "How should we investigate?" in text
    assert any(b.label.startswith("Start investigation") for b in at.button)
    assert len(at.sidebar) == 0 or not any(w for w in at.sidebar.selectbox)  # no settings crammed in the sidebar


def test_agents_question_is_a_chat_message_with_quick_replies():
    at = _app()
    at.session_state["pending"] = {
        "thread_id": "t-1", "scenario": None, "custom": False,
        "question": {"reason": "stuck", "question": "The investigation is stuck. Any hint?", "round": 1,
                     "critic_decision": "INCONCLUSIVE", "critic_issues": ["cannot separate DB from deploy"],
                     "hypotheses": [{"component": "payments-db", "category": "resource_exhaustion",
                                     "confidence": 0.55, "title": "pool exhaustion"}], "human_rounds_left": 2}}
    at.run()
    assert not at.exception
    assert at.chat_message and at.chat_input
    labels = [b.label for b in at.button]
    assert "Write the report now" in labels and "Investigate further" in labels


def test_eval_overview_reads_boolean_text_correctly(tmp_path):
    from rootcause_agents.ui.view import eval_overview

    p = tmp_path / "summary.csv"
    p.write_text("system,top1_correct,outcome_correct,cost_usd,latency_s,tool_calls\n"
                 "mas,True,True,0.1,100,30\nmas,False,True,0.2,200,40\nsingle,True,False,0.04,50,39\n")
    rows = {r["system"]: r for r in eval_overview(p)}
    assert rows["mas"]["accuracy"] == 0.5 and rows["mas"]["outcome"] == 1.0
    assert rows["mas"]["cost"] == 0.15 and rows["mas"]["runs"] == 2
    assert rows["single"]["accuracy"] == 1.0 and rows["single"]["outcome"] == 0.0


def test_stuck_question_avoids_jargon():
    import inspect

    from rootcause_agents import graph

    src = inspect.getsource(graph)
    assert "(critic: {v.decision})" not in src
