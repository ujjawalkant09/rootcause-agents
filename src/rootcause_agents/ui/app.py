"""RootCause Agents – Streamlit UI.

Start with:  uv run rootcause-agents ui   (or double-click start-ui.command on macOS)

Tabs: Investigate (pick a test incident or write your own ticket, watch the agents live, answer their questions,
read the report), Sessions (persistent LangGraph sessions in Postgres; resume any paused one), Evaluation, Past runs.
Nothing calls the API until a Run button is pressed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import streamlit as st

from rootcause_agents.config import SYSTEMS, Settings
from rootcause_agents.evaluation import grade
from rootcause_agents.persistence import open_checkpointer, open_registry
from rootcause_agents.ui.view import (
    SYSTEM_HELP,
    format_event,
    list_eval_results,
    list_runs,
    outcome_badge,
    scenario_options,
)

# measured on the S01 pilot with the cheaper models (USD per incident); used only for the cost estimate
EST_COST = {"mas": 0.11, "mas-cheap": 0.06, "single": 0.035}

st.set_page_config(page_title="RootCause Agents", page_icon="🔍", layout="wide")


@st.cache_resource
def _settings() -> Settings:
    return Settings.from_env()


settings = _settings()
scenarios = scenario_options(settings.data_dir)
by_label = {s["label"]: s for s in scenarios}

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🔍 RootCause Agents")
    st.caption("Multi-agent incident investigation · LangGraph + OpenAI")
    system = st.radio("System", SYSTEMS, format_func=lambda s: f"{s}", captions=[SYSTEM_HELP[s] for s in SYSTEMS])
    max_rounds = st.slider("Max investigation rounds", 1, 3, settings.budget.max_rounds,
                           help="More rounds let the critic ask for more evidence (more cost).")
    hitl_label = st.segmented_control(
        "Ask me for input", ["Off", "When stuck", "Review"], default="When stuck",
        help="When stuck: the agents pause and ask you when the critic cannot conclude. Review: they also ask you "
             "to approve the conclusion before the report. Multi-agent systems only; at most 2 questions per run.")
    hitl = {"Off": "off", "When stuck": "stuck", "Review": "review"}.get(hitl_label or "Off", "off")
    st.divider()
    st.markdown(f"**Main model:** `{settings.main_model}`  \n**Cheap worker model:** `{settings.worker_model}`  \n"
                f"**Fallbacks:** `{', '.join(settings.fallback_models) or 'none'}`")
    if settings.openai_api_key:
        st.success("OpenAI key loaded", icon="🔑")
    else:
        st.error("OPENAI_API_KEY missing in .env", icon="🔑")
    if settings.database_url:
        st.success("Sessions saved in Postgres", icon="🗄️")
    else:
        st.warning("Sessions in memory only (set DATABASE_URL)", icon="🗄️")
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        st.success("Langfuse tracing on", icon="📡")
        st.link_button("Open Langfuse", settings.langfuse_base_url)
    else:
        st.info("Langfuse off (add LANGFUSE_* keys to .env)", icon="📡")

run_settings = settings.model_copy(update={"budget": settings.budget.model_copy(update={"max_rounds": max_rounds})})
can_run = bool(settings.openai_api_key)


def _render_report(report, scenario=None, custom: bool = False) -> None:
    label, color = outcome_badge(report.outcome)
    top = report.ranked_hypotheses[0] if report.ranked_hypotheses else None
    st.subheader("Result")
    st.badge(label, color=color)
    if top:
        st.markdown(f"### {top.root_cause_component} · `{top.root_cause_category}` · confidence {top.confidence:.2f}")
    st.write(report.summary)

    s = report.stats
    cols = st.columns(6)
    cols[0].metric("Cost", f"${s.cost_usd:.3f}")
    cols[1].metric("Time", f"{s.latency_s:.0f}s")
    cols[2].metric("Rounds", s.rounds)
    cols[3].metric("Tool calls", s.tool_calls)
    cols[4].metric("LLM calls", s.llm_calls)
    cols[5].metric("Tokens", f"{s.input_tokens + s.output_tokens:,}")
    if s.trace_url:
        st.link_button("📡 Open this run in Langfuse", s.trace_url)

    if report.ranked_hypotheses:
        st.markdown("#### Ranked hypotheses")
        st.dataframe(pd.DataFrame([{"#": i + 1, "component": h.root_cause_component, "category": h.root_cause_category,
                                    "confidence": round(h.confidence, 2), "hypothesis": h.title}
                                   for i, h in enumerate(report.ranked_hypotheses)]), hide_index=True,
                     width="stretch")
        for h in report.ranked_hypotheses:
            with st.expander(f"Mechanism · {h.title}"):
                for c in h.mechanism:
                    st.markdown(f"- **{c.kind}** {c.text} {' '.join(f'`{e}`' for e in c.evidence_ids)}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ✅ Facts (cited)")
        for c in report.facts:
            st.markdown(f"- {c.text} {' '.join(f'`{e}`' for e in c.evidence_ids)}")
        st.markdown("#### 💭 Assumptions")
        for c in report.assumptions or []:
            st.markdown(f"- {c.text}")
    with c2:
        st.markdown("#### 🔗 Inferences")
        for c in report.inferences:
            st.markdown(f"- {c.text} {' '.join(f'`{e}`' for e in c.evidence_ids)}")
        st.markdown("#### 🛠️ Recommended actions")
        for a in report.recommended_actions:
            st.markdown(f"- **{a.priority}** `{a.type}` {a.action}")

    if report.data_gaps:
        with st.expander(f"⚠️ Data gaps ({len(report.data_gaps)})"):
            for g in report.data_gaps:
                st.markdown(f"- {g}")
    if report.critic_issues:
        with st.expander("🧐 Critic's remaining concerns"):
            for i in report.critic_issues:
                st.markdown(f"- {i}")
    with st.expander(f"📚 Evidence ledger ({len(report.evidence)} items)"):
        st.dataframe(pd.DataFrame([{"id": e.id, "tool": e.source_tool, "status": e.status, "by": e.collected_by,
                                    "round": e.round, "summary": e.summary} for e in report.evidence]),
                     hide_index=True, width="stretch")
    if s.by_role:
        with st.expander("💰 Tokens and cost by agent"):
            st.dataframe(pd.DataFrame([r.model_dump() for r in s.by_role]), hide_index=True, width="stretch")
    if scenario is not None and not custom:
        with st.expander("🎯 Compare with the ground truth (evaluation only, never shown to the agents)"):
            g = grade(report, scenario)
            st.markdown(f"- Truth: **{g['truth_component']} / {g['truth_category']}**\n"
                        f"- Top-1 correct: {'✅' if g['top1_correct'] else '❌'} · "
                        f"Outcome acceptable: {'✅' if g['outcome_correct'] else '❌'} · "
                        f"Key-evidence recall: {g['key_evidence_recall']:.2f}")
    from rootcause_agents.report import render_markdown

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Report (Markdown)", render_markdown(report), file_name=f"{report.incident_id}.md")
    d2.download_button("⬇️ Report (JSON)", report.model_dump_json(indent=2), file_name=f"{report.incident_id}.json")


def _live_log():
    def echo(ev):
        if ev.get("event") in ("run_start", "run_resume") and ev.get("trace_url"):
            st.markdown(f"📡 [Live trace in Langfuse]({ev['trace_url']})")
        line = format_event(ev)
        if line:
            st.markdown(line)

    return echo


def _execute(label: str, coro_factory):
    """Run one session segment with a live activity log; returns a SessionResult or None on error."""
    with st.status(label, expanded=True) as status:
        try:
            res = asyncio.run(coro_factory(_live_log()))
        except Exception as e:
            status.update(label="Investigation failed", state="error")
            st.error(f"{type(e).__name__}: {e}")
            return None
        if res.status == "waiting":
            status.update(label=f"Paused: the agents need your input · ${res.cost_usd:.3f} so far",
                          state="complete", expanded=False)
        else:
            status.update(label=f"Done · {outcome_badge(res.report.outcome)[0]} · ${res.report.stats.cost_usd:.3f}",
                          state="complete", expanded=False)
    return res


def _start(path, ticket):
    from rootcause_agents.runner import start_session

    async def go(echo):
        async with open_checkpointer(run_settings) as cp, open_registry(run_settings) as reg:
            return await start_session(path, system, settings=run_settings, echo=echo, ticket=ticket, hitl=hitl,
                                       checkpointer=cp, registry=reg)

    return _execute(f"Investigating with **{system}** …", go)


def _resume(thread_id: str, answer: dict):
    from rootcause_agents.runner import resume_session

    async def go(echo):
        async with open_checkpointer(run_settings) as cp, open_registry(run_settings) as reg:
            return await resume_session(thread_id, answer, settings=run_settings, echo=echo, checkpointer=cp,
                                        registry=reg)

    return _execute("Resuming with your input …", go)


def _store_result(res, scenario=None, custom: bool = False) -> None:
    if res is None:
        return
    if res.status == "waiting":
        st.session_state["pending"] = {"thread_id": res.thread_id, "question": res.question,
                                       "scenario": scenario, "custom": custom}
        st.session_state.pop("last", None)
    else:
        st.session_state.pop("pending", None)
        st.session_state["last"] = (res.report, scenario, custom)


def _question_panel(thread_id: str, question: dict, key: str, scenario=None, custom: bool = False) -> None:
    """Show the agents' question and collect the human answer (LangGraph interrupt → resume)."""
    with st.container(border=True):
        st.markdown("#### :material/support_agent: The agents need your input")
        st.info(question.get("question", ""), title="Stuck" if question.get("reason") == "stuck" else "Review")
        if question.get("hypotheses"):
            st.dataframe(pd.DataFrame(question["hypotheses"]), hide_index=True)
        if question.get("critic_issues"):
            with st.expander("Critic's concerns"):
                for i in question["critic_issues"]:
                    st.markdown(f"- {i}")
        st.caption(f"Session `{thread_id}` · saved, so you can also answer later from the Sessions tab · "
                   f"{question.get('human_rounds_left', 0)} question(s) left in this run")
        with st.form(f"answer_{key}"):
            hint = st.text_area("Your hint (optional)", key=f"hint_{key}",
                                placeholder="e.g. The payments DB connection pool was resized this morning")
            c1, c2 = st.columns(2)
            more = c1.form_submit_button("Investigate further with my hint", type="primary", width="stretch")
            done = c2.form_submit_button("Write the report now", width="stretch")
        if more or done:
            res = _resume(thread_id, {"action": "continue" if more else "report", "hint": hint})
            _store_result(res, scenario, custom)
            st.rerun()


tab_inv, tab_sessions, tab_eval, tab_runs = st.tabs(["🔍 Investigate", "💬 Sessions", "📊 Evaluation", "🗂️ Past runs"])

# ---------------------------------------------------------------- investigate
with tab_inv:
    mode = st.segmented_control("Ticket", ["Test incident", "Write your own ticket"], default="Test incident") \
        or "Test incident"
    choice = by_label[st.selectbox(
        "Simulated environment" if mode != "Test incident" else "Test incident", list(by_label),
        help="The agents investigate this environment's simulated logs, metrics, deploys and dependencies.")]
    inc = choice["scenario"].incident
    ticket = None
    if mode == "Test incident":
        with st.container(border=True):
            st.markdown(f"**{inc.title}**  \n{inc.description}")
            st.markdown(f"Reported at `{inc.reported_at:%Y-%m-%d %H:%M} UTC` · affected: "
                        + ", ".join(f"`{s}`" for s in inc.affected_services))
            st.markdown("Symptoms: " + "; ".join(inc.symptoms))
            st.caption("Scenario traits: " + (", ".join(choice["traits"]) or "none"))
    else:
        st.caption("Your ticket is investigated against the simulated telemetry of the environment chosen above "
                   f"(timeline around {inc.reported_at:%Y-%m-%d %H:%M} UTC). No real production systems are queried.")
        title = st.text_input("Title", placeholder="e.g. Checkout is slow and some payments fail")
        description = st.text_area("Description", placeholder="What users or monitors reported…")
        symptoms = st.text_area("Symptoms (one per line)", placeholder="checkout p99 latency > 5s\n5xx rate ~8%")
        affected = st.text_input("Affected services (comma separated)", value=", ".join(inc.affected_services))
        ticket = {"title": title.strip(), "description": description.strip(),
                  "symptoms": [x.strip() for x in symptoms.splitlines() if x.strip()],
                  "affected_services": [x.strip() for x in affected.split(",") if x.strip()]}

    est = EST_COST[system]
    ready = can_run and (mode == "Test incident" or bool(ticket and ticket["title"]))
    if st.button(f"Run investigation (≈ ${est:.2f})", icon=":material/play_arrow:", type="primary",
                 disabled=not ready):
        _store_result(_start(choice["path"], ticket), choice["scenario"], mode != "Test incident")
    pending = st.session_state.get("pending")
    if pending:
        _question_panel(pending["thread_id"], pending["question"], "inv", pending["scenario"], pending["custom"])
    elif "last" in st.session_state:
        _render_report(*st.session_state["last"])

# ---------------------------------------------------------------- sessions (persistent, LangGraph checkpointer)
with tab_sessions:
    st.markdown("Every multi-agent investigation is a LangGraph **thread** checkpointed after each step "
                + ("in **Postgres**" if settings.database_url else "**in memory** (set `DATABASE_URL` for Postgres)")
                + ". Paused sessions can be answered and resumed at any time, even after restarting the app.")

    async def _list_sessions():
        async with open_registry(settings) as reg:
            return await reg.alist(50)

    try:
        sessions = asyncio.run(_list_sessions())
    except Exception as e:
        sessions = []
        st.error(f"Could not read sessions: {e}", title="Database unavailable (is `docker compose up -d` running?)")
    if not sessions:
        st.info("No sessions yet.")
    else:
        st.dataframe(pd.DataFrame([{k: r.get(k) for k in ("thread_id", "status", "system", "title", "outcome",
                                                           "cost_usd", "updated_at")} for r in sessions]),
                     hide_index=True)
        waiting = [r for r in sessions if r.get("status") == "waiting"]
        if waiting:
            pick = st.selectbox("Answer a paused session", [r["thread_id"] for r in waiting])
            row = next(r for r in waiting if r["thread_id"] == pick)
            q = row["question"] if isinstance(row["question"], dict) else {}
            _question_panel(pick, q, f"s_{pick}")
        done_rows = [r for r in sessions if r.get("status") == "done" and r.get("out_dir")]
        if done_rows:
            pick_done = st.selectbox("Open a finished session", [r["thread_id"] for r in done_rows])
            rd = next(r for r in done_rows if r["thread_id"] == pick_done)
            report_md = Path(rd["out_dir"]) / "report.md"
            if rd.get("trace_url"):
                st.link_button("Open in Langfuse", rd["trace_url"], icon=":material/monitoring:")
            if report_md.exists():
                with st.expander("Report", expanded=False):
                    st.markdown(report_md.read_text())

# ---------------------------------------------------------------- evaluation
with tab_eval:
    st.markdown("Run every selected incident on every selected system and compare accuracy, grounding, cost, "
                "latency and tokens.")
    e_sys = st.multiselect("Systems", SYSTEMS, default=list(SYSTEMS))
    e_sc = st.multiselect("Incidents", list(by_label), default=list(by_label))
    est_total = sum(EST_COST[s] for s in e_sys) * len(e_sc)
    c1, c2 = st.columns(2)
    cap = c1.number_input("Hard spend cap (USD)", min_value=0.1, value=max(2.0, round(est_total * 1.3, 1)), step=0.5)
    c2.metric("Estimated cost", f"${est_total:.2f}")
    if st.button("▶️ Run evaluation", type="primary", disabled=not (can_run and e_sys and e_sc)):
        from datetime import datetime

        from rootcause_agents.evaluation import run_matrix

        out = settings.data_dir.parent / "eval" / "results" / f"ui-{datetime.now():%Y%m%d-%H%M%S}"
        with st.status("Running evaluation …", expanded=True) as status:
            asyncio.run(run_matrix(e_sys, [by_label[x]["path"] for x in e_sc], run_settings, out_dir=out,
                                   progress=lambda m: st.markdown(f"- {m}"), max_cost_usd=cap))
            status.update(label="Evaluation finished", state="complete", expanded=False)
        st.session_state["eval_view"] = str(out / "summary.md")
    results = list_eval_results(settings.data_dir.parent / "eval" / "results")
    if results:
        names = [str(p) for p in results]
        default = names.index(st.session_state["eval_view"]) if st.session_state.get("eval_view") in names else 0
        pick = st.selectbox("Results", names, index=default,
                            format_func=lambda p: p.split("eval/results/")[-1].replace("/summary.md", "") or "latest")
        st.markdown(Path(pick).read_text())
    else:
        st.info("No evaluation results yet.")

# ---------------------------------------------------------------- past runs
with tab_runs:
    runs = list_runs(settings.runs_dir)
    if not runs:
        st.info("No runs yet. Start one in the Investigate tab.")
    else:
        st.dataframe(pd.DataFrame([{k: r[k] for k in ("name", "system", "outcome", "cost_usd", "latency_s")}
                                   for r in runs]), hide_index=True, width="stretch")
        pick = st.selectbox("Open run", [r["name"] for r in runs])
        r = next(x for x in runs if x["name"] == pick)
        if r.get("trace_url"):
            st.link_button("📡 Open in Langfuse", r["trace_url"])
        st.markdown((r["path"] / "report.md").read_text())
