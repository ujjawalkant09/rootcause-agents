"""Sessions: every multi-agent investigation is a saved LangGraph thread. Answer the agents, continue stopped runs,
re-open finished reports."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from rootcause_agents.ui import common
from rootcause_agents.ui.view import list_runs, outcome_badge

s = common.settings()
rows = common.sessions()
busy = "job" in st.session_state

st.caption("Investigations are saved after every step"
           + (" in Postgres" if s.database_url else " in memory (set DATABASE_URL to keep them across restarts)")
           + ", so you can answer the agents or continue a stopped run at any time.")


@st.dialog("Report", width="large", position="right")
def show_report(out_dir: str, trace_url: str | None) -> None:
    md = Path(out_dir) / "report.md"
    if trace_url:
        st.link_button("Open in Langfuse", trace_url, icon=":material/monitoring:")
    st.markdown(md.read_text() if md.exists() else "_Report file not found._")


waiting = [r for r in rows if r.get("status") == "waiting"]
stale = (datetime.now(UTC) - timedelta(minutes=15)).isoformat()
stopped = [r for r in rows if r.get("status") in ("interrupted", "failed")
           or (r.get("status") == "running" and str(r.get("updated_at", "")) < stale)]
running = [r for r in rows if r.get("status") == "running" and r not in stopped]
done = [r for r in rows if r.get("status") == "done"]

st.subheader(f"Waiting for you ({len(waiting)})", anchor=False)
if not waiting:
    st.caption("Nothing needs your answer right now.")
for r in waiting:
    q = r.get("question") or {}
    q = json.loads(q) if isinstance(q, str) else q
    with st.container(border=True):
        st.markdown(f"**{r.get('title') or r['thread_id']}**")
        st.caption(q.get("question", ""))
        if st.button("Answer", key=f"ans_{r['thread_id']}", icon=":material/chat:", type="primary", disabled=busy):
            st.session_state["pending"] = {"thread_id": r["thread_id"], "question": q, "scenario": None,
                                           "custom": True}
            st.session_state.pop("last", None)
            st.switch_page(common.PAGES["investigate"])

if running:
    st.subheader(f"Running ({len(running)})", anchor=False)
    for r in running:
        st.markdown(f"- {r.get('title') or r['thread_id']} :gray[· started {str(r.get('created_at', ''))[:16]}]")

if stopped:
    st.subheader(f"Stopped before finishing ({len(stopped)})", anchor=False)
    for r in stopped:
        with st.container(border=True):
            st.markdown(f"**{r.get('title') or r['thread_id']}**")
            st.caption(r.get("error") or "The run stopped before finishing.")
            if st.button("Continue from the last checkpoint", key=f"cont_{r['thread_id']}",
                         icon=":material/play_arrow:", disabled=busy,
                         help="Finished steps are not repeated, so this costs only the remaining work."):
                common.resume_investigation(r["thread_id"], None)
                st.switch_page(common.PAGES["investigate"])

st.subheader(f"Finished ({len(done)})", anchor=False)
if done:
    table = pd.DataFrame([{"incident": r.get("title"), "result": outcome_badge(r.get("outcome") or "")[0],
                           "cost ($)": round(r.get("cost_usd") or 0, 3), "updated": str(r.get("updated_at"))[:16].replace("T", " "),
                           "_id": r["thread_id"]} for r in done])
    event = st.dataframe(table.drop(columns="_id"), hide_index=True, on_select="rerun", selection_mode="single-row",
                         key="done_table")
    picked = event.selection.rows if event and event.selection else []
    if picked:
        r = done[picked[0]]
        show_report(r.get("out_dir") or "", r.get("trace_url"))
    st.caption("Select a row to open its report.")
else:
    st.caption("No finished investigations yet.")

with st.expander("Older reports (command line and evaluation runs)", icon=":material/folder_open:"):
    old = list_runs(s.runs_dir)
    if old:
        st.dataframe(pd.DataFrame([{k: x[k] for k in ("name", "system", "outcome", "cost_usd")} for x in old]),
                     hide_index=True)
    else:
        st.caption("None.")
