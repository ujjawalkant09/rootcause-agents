"""Investigate: describe what happened, choose how to investigate, start. Progress, questions and the report
appear on the same page."""

import streamlit as st

from rootcause_agents.ui import common
from rootcause_agents.ui.view import TEAM_OPTIONS

s = common.settings()
scenarios = common.scenarios()
by_title = {sc["title"]: sc for sc in scenarios}
job = st.session_state.get("job")
pending = st.session_state.get("pending")
last = st.session_state.get("last")

if not s.openai_api_key:
    st.warning("Add your OpenAI key to `.env` (OPENAI_API_KEY) and restart the app.", title="OpenAI key missing",
               icon=":material/key_off:")

# ---------------------------------------------------------------- results first, when there is something to show
if job is not None and job.context.get("kind") == "session":
    common.job_progress()
    st.stop()

common.finished_banner()
if pending:
    common.question_chat(pending["thread_id"], pending["question"], key="inv", scenario=pending["scenario"],
                         custom=pending["custom"])
    st.stop()

if last:
    common.report_view(*last)
    if st.button("Start a new investigation", icon=":material/add:"):
        for k in ("last", "job_log", "job_error"):
            st.session_state.pop(k, None)
        st.rerun()
    st.stop()

# ---------------------------------------------------------------- step 1: what happened?
st.caption("Describe an incident and a team of AI agents will investigate logs, metrics, deployments and "
           "dependencies, then report the most likely root cause with evidence.")

st.subheader("What happened?", anchor=False)
mode = st.segmented_control("Incident", ["Pick a sample incident", "Describe your own"],
                            default="Pick a sample incident", label_visibility="collapsed",
                            key="mode") or "Pick a sample incident"
ticket = None
if mode == "Pick a sample incident":
    title = st.pills("Sample incidents", list(by_title), default=scenarios[0]["title"],
                     label_visibility="collapsed", key="sample") or scenarios[0]["title"]
    choice = by_title[title]
    inc = choice["scenario"].incident
    with st.container(border=True):
        st.markdown(f"**{inc.title}**  \n{inc.description}")
        st.caption("Symptoms: " + "; ".join(inc.symptoms) + " · Affected: " + ", ".join(inc.affected_services))
else:
    from rootcause_agents.sim.scenario import load_topology

    services = sorted(load_topology(s.data_dir / "topology.yaml").services)
    title_in = st.text_input("Title", placeholder="e.g. Card payments failing since this morning", key="t_title")
    desc = st.text_area("What are users or monitors seeing?", key="t_desc",
                        placeholder="e.g. Most card payments are declined; retrying does not help.")
    symptoms = st.text_area("Symptoms (one per line, optional)", key="t_sym", height=80,
                            placeholder="card payment errors ~30%")
    affected = st.multiselect("Affected services", services, key="t_aff", placeholder="Choose services")
    env_title = st.selectbox("Simulated data to investigate", list(by_title), key="t_env",
                             help="This demo has no real production data. Your ticket is investigated against "
                                  "the simulated logs, metrics and deploys of this environment.")
    choice = by_title[env_title]
    ticket = {"title": title_in.strip(), "description": desc.strip(),
              "symptoms": [x.strip() for x in symptoms.splitlines() if x.strip()],
              "affected_services": affected or list(choice["scenario"].incident.affected_services)}

# ---------------------------------------------------------------- step 2: how should we investigate?
st.subheader("How should we investigate?", anchor=False)
with st.container(horizontal=True, vertical_alignment="bottom"):
    team = st.segmented_control("Investigation team", list(TEAM_OPTIONS), default="Team (budget)",
                                key="team") or "Team (budget)"
    with st.popover("Advanced", icon=":material/tune:"):
        rounds = st.slider("Max investigation rounds", 1, 3, s.budget.max_rounds, key="rounds",
                           help="Each extra round lets the critic ask for more evidence (more time and cost).")
        review = st.toggle("Always ask me before the final report", key="review",
                           help="Pause for your approval even when the critic is satisfied.")
opt = TEAM_OPTIONS[team]
ask = opt["system"] != "single" and st.toggle(
    "Let the agents ask me when they are stuck", value=True, key="ask",
    help="The investigation pauses and asks you for a hint instead of guessing. Answer now or later.")
st.caption(opt["help"])

hitl = "review" if (ask and review) else ("stuck" if ask else "off")
ready = bool(s.openai_api_key) and (mode == "Pick a sample incident" or bool(ticket and ticket["title"]))
if st.button(f"Start investigation · about ${opt['cost']:.2f} · ~{opt['minutes']:g} min", type="primary",
             icon=":material/play_arrow:", width="stretch", disabled=not ready):
    run_settings = s.model_copy(update={"budget": s.budget.model_copy(update={"max_rounds": rounds})})
    common.start_investigation(choice["path"], opt["system"], run_settings, hitl, ticket, choice["scenario"],
                               custom=mode != "Pick a sample incident")
    st.rerun()
if mode != "Pick a sample incident" and not (ticket and ticket["title"]):
    st.caption("Add a title to start.")
