"""Shared UI building blocks: settings, background jobs, progress timeline, agents' questions, report view."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import streamlit as st

from rootcause_agents.config import Settings
from rootcause_agents.persistence import open_checkpointer, open_registry
from rootcause_agents.ui.jobs import Job
from rootcause_agents.ui.view import outcome_badge, scenario_options, stages_from_events

PAGES: dict[str, Any] = {}  # st.Page objects, set by app.py (used for st.switch_page)


@st.cache_resource
def settings() -> Settings:
    return Settings.from_env()


@st.cache_resource
def scenarios() -> list[dict[str, Any]]:
    return scenario_options(settings().data_dir)


@st.cache_resource
def _pool() -> ThreadPoolExecutor:
    """Worker threads for investigations: a UI rerun (any click) only redraws the page, it cannot kill a run."""
    return ThreadPoolExecutor(max_workers=4, thread_name_prefix="investigation")


@st.cache_data(ttl=5, show_spinner=False)
def sessions(limit: int = 100) -> list[dict[str, Any]]:
    async def go():
        async with open_registry(settings()) as reg:
            return await reg.alist(limit)

    try:
        return asyncio.run(go())
    except Exception:
        return []


def waiting_count() -> int:
    return sum(1 for r in sessions() if r.get("status") == "waiting")


# ---------------------------------------------------------------- jobs
def submit(label: str, work, **context) -> None:
    st.session_state["job"] = Job.submit(_pool(), label, work, context=context)
    for k in ("job_log", "job_error"):
        st.session_state.pop(k, None)
    sessions.clear()


def start_investigation(path, system: str, run_settings: Settings, hitl: str, ticket, scenario, custom: bool):
    from rootcause_agents.runner import start_session

    async def go(echo):
        async with open_checkpointer(run_settings) as cp, open_registry(run_settings) as reg:
            return await start_session(path, system, settings=run_settings, echo=echo, ticket=ticket, hitl=hitl,
                                       checkpointer=cp, registry=reg)

    st.session_state.pop("last", None)
    st.session_state.pop("pending", None)
    submit("Investigating", go, kind="session", scenario=scenario, custom=custom)


def resume_investigation(thread_id: str, answer: dict | None, scenario=None, custom: bool = False):
    from rootcause_agents.runner import resume_session

    s = settings()

    async def go(echo):
        async with open_checkpointer(s) as cp, open_registry(s) as reg:
            return await resume_session(thread_id, answer, settings=s, echo=echo, checkpointer=cp, registry=reg)

    st.session_state.pop("pending", None)
    st.session_state.pop("last", None)
    label = "Continuing with your input" if answer is not None else "Continuing from the last checkpoint"
    submit(label, go, kind="session", scenario=scenario, custom=custom)


def store_result(res, scenario=None, custom: bool = False) -> None:
    if res is None:
        return
    if res.status == "waiting":
        st.session_state["pending"] = {"thread_id": res.thread_id, "question": res.question,
                                       "scenario": scenario, "custom": custom}
        st.session_state.pop("last", None)
    else:
        st.session_state.pop("pending", None)
        st.session_state["last"] = (res.report, scenario, custom)


def timeline(events: list[Any], *, live: bool) -> None:
    """Plain-language stages with the details that matter; upcoming stages as one quiet line."""
    stages = stages_from_events(events)
    for stg in (x for x in stages if x["state"] != "pending"):
        running = live and stg["state"] == "running"
        label = f":shimmer[{stg['label']}]" if running else stg["label"]
        with st.status(label, type="step", state="running" if running else "complete", expanded=running):
            st.caption(stg["detail"] or "Working…")
    upcoming = [x["label"] for x in stages if x["state"] == "pending"]
    if upcoming:
        st.caption("Next: " + " → ".join(upcoming))


@st.fragment(run_every=1.0)
def job_progress() -> None:
    """Live progress of the running job; when it finishes, store the result and redraw the page."""
    job: Job | None = st.session_state.get("job")
    if job is None:
        return
    if not job.done:
        with st.chat_message("assistant", avatar=":material/smart_toy:"):
            st.markdown(f"**{job.label}…** You can keep using the app; the agents keep working.")
            trace = next((e.get("trace_url") for e in job.events if isinstance(e, dict) and e.get("trace_url")), None)
            if trace:
                st.caption(f"[Watch this run live in Langfuse]({trace})")
            timeline(job.events, live=True)
            with st.expander("Technical log", icon=":material/terminal:"):
                for line in job.lines()[-60:]:
                    st.markdown(line)
        return
    st.session_state.pop("job")
    st.session_state["job_log"] = job.events
    try:
        res = job.result()
    except Exception as e:
        st.session_state["job_error"] = f"{type(e).__name__}: {e}"
    else:
        if job.context.get("kind") == "session":
            store_result(res, job.context.get("scenario"), job.context.get("custom", False))
        elif job.context.get("kind") == "eval":
            st.session_state["eval_view"] = job.context["summary"]
    sessions.clear()
    st.rerun(scope="app")


def finished_banner() -> None:
    if err := st.session_state.get("job_error"):
        st.error(f"{err}\n\nThe session is saved; you can continue it from **Sessions**.",
                 title="The investigation stopped", icon=":material/error:")


# ---------------------------------------------------------------- agents' question (human-in-the-loop)
def question_chat(thread_id: str, q: dict, *, key: str, scenario=None, custom: bool = False) -> None:
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        st.markdown("**The agents need your input**")
        st.write(q.get("question", ""))
        hyps = q.get("hypotheses") or []
        if hyps:
            st.markdown("Current best guesses:")
            for h in hyps:
                st.markdown(f"- **{h['component']}** · {h['category'].replace('_', ' ')} · "
                            f"{h['confidence']:.0%} — {h['title']}")
        if q.get("critic_issues"):
            with st.expander("Why the critic is not convinced", icon=":material/help:"):
                for i in q["critic_issues"]:
                    st.markdown(f"- {i}")
        with st.container(horizontal=True):
            done = st.button("Write the report now", type="primary", icon=":material/description:",
                             key=f"report_{key}")
            more = st.button("Investigate further", icon=":material/manage_search:", key=f"more_{key}")
        st.caption(f"Or type a hint below. Questions left in this run: {q.get('human_rounds_left', 0)}. "
                   "The session is saved, so you can also answer later from Sessions.")
    hint = st.chat_input("Give the agents a hint, e.g. “the payments DB pool was resized this morning”",
                         key=f"hint_{key}")
    if hint or more or done:
        answer = {"action": "report" if done else "continue", "hint": hint or ""}
        resume_investigation(thread_id, answer, scenario, custom)
        st.rerun()


# ---------------------------------------------------------------- report
def _cites(ids: list[str]) -> str:
    return " ".join(f"`{i}`" for i in ids)


def report_view(report, scenario=None, custom: bool = False) -> None:
    label, color = outcome_badge(report.outcome)
    top = report.ranked_hypotheses[0] if report.ranked_hypotheses else None
    s = report.stats
    with st.container(border=True):
        st.badge(label, color=color, icon=":material/verified:" if color == "green" else None)
        if top:
            st.subheader(f"{top.root_cause_component} · {top.root_cause_category.replace('_', ' ')}")
            st.progress(min(max(top.confidence, 0.0), 1.0), text=f"Confidence {top.confidence:.0%}")
        else:
            st.subheader("No root cause identified")
        st.write(report.summary)
        if report.human_inputs:
            st.info("\n".join(f"- {h}" for h in report.human_inputs), title="Human input (a lead, not evidence)",
                    icon=":material/support_agent:")
        st.caption(f"${s.cost_usd:.3f} · {s.latency_s:.0f}s · {s.rounds} round{'s' if s.rounds != 1 else ''} · "
                   f"{s.tool_calls} tool calls · {s.llm_calls} LLM calls")
        with st.container(horizontal=True):
            if s.trace_url:
                st.link_button("Open in Langfuse", s.trace_url, icon=":material/monitoring:")
            from rootcause_agents.report import render_markdown

            st.download_button("Download report", render_markdown(report), file_name=f"{report.incident_id}.md",
                               icon=":material/download:")

    t_find, t_next, t_hyp, t_ev, t_cost = st.tabs([
        ":material/fact_check: Findings", ":material/checklist: Next steps", ":material/account_tree: Hypotheses",
        ":material/database: Evidence", ":material/payments: Cost"])
    with t_find:
        st.markdown("**Facts** (each cites the tool result it comes from)")
        for c in report.facts:
            st.markdown(f"- {c.text} {_cites(c.evidence_ids)}")
        st.markdown("**Inferences**")
        for c in report.inferences:
            st.markdown(f"- {c.text} {_cites(c.evidence_ids)}")
        if report.assumptions:
            st.markdown("**Assumptions** (not backed by evidence)")
            for c in report.assumptions:
                st.markdown(f"- {c.text}")
    with t_next:
        colors = {"P0": "red", "P1": "orange", "P2": "blue", "P3": "gray"}
        for a in report.recommended_actions:
            st.markdown(f":{colors.get(a.priority, 'gray')}-badge[{a.priority}] "
                        f":gray-badge[{a.type.replace('_', ' ')}] {a.action}")
    with t_hyp:
        st.dataframe(pd.DataFrame([{"rank": i + 1, "component": h.root_cause_component,
                                    "category": h.root_cause_category.replace("_", " "),
                                    "confidence": h.confidence, "hypothesis": h.title}
                                   for i, h in enumerate(report.ranked_hypotheses)]),
                     hide_index=True, column_config={"confidence": st.column_config.ProgressColumn(
                         "confidence", min_value=0, max_value=1, format="percent")})
        for h in report.ranked_hypotheses:
            with st.expander(f"How it would happen: {h.title}"):
                for c in h.mechanism:
                    st.markdown(f"- :gray-badge[{c.kind}] {c.text} {_cites(c.evidence_ids)}")
        if report.critic_issues:
            with st.expander("Critic's remaining concerns", icon=":material/rate_review:"):
                for i in report.critic_issues:
                    st.markdown(f"- {i}")
    with t_ev:
        st.dataframe(pd.DataFrame([{"id": e.id, "tool": e.source_tool, "status": e.status, "agent": e.collected_by,
                                    "round": e.round, "summary": e.summary} for e in report.evidence]),
                     hide_index=True)
        if report.data_gaps:
            st.markdown("**Data gaps** (tool failures and missing data)")
            for g in report.data_gaps:
                st.markdown(f"- {g}")
    with t_cost:
        if s.by_role:
            st.dataframe(pd.DataFrame([{"agent": r.role, "calls": r.calls, "input tokens": r.input_tokens,
                                        "cached": r.cached_tokens, "output tokens": r.output_tokens,
                                        "cost ($)": round(r.cost_usd, 4)} for r in s.by_role]), hide_index=True)
    if scenario is not None and not custom:
        from rootcause_agents.evaluation import grade

        g = grade(report, scenario)
        with st.expander("Check against the answer key (sample incidents only)", icon=":material/quiz:"):
            st.markdown(f"Real cause: **{g['truth_component']} · {g['truth_category'].replace('_', ' ')}**  \n"
                        f"Root cause correct: {'yes' if g['top1_correct'] else 'no'} · "
                        f"Outcome acceptable: {'yes' if g['outcome_correct'] else 'no'}")
