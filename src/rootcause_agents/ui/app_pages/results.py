"""Results: how the multi-agent system compares with a single agent on the 10 sample incidents."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from rootcause_agents.ui import common
from rootcause_agents.ui.view import TEAM_OPTIONS, eval_overview, list_eval_results

s = common.settings()
root = s.data_dir.parent / "eval" / "results"
NAMES = {v["system"]: k for k, v in TEAM_OPTIONS.items()}

st.caption("Every sample incident run by each investigation team, with the same tools, data and budget.")
results = list_eval_results(root)
if not results:
    st.info("No evaluation results yet. Run one below.", icon=":material/info:")
else:
    labels = {str(p): p.parent.name if p.parent != root else "latest" for p in results}
    default = st.session_state.get("eval_view")
    pick = st.selectbox("Evaluation run", list(labels), format_func=labels.get,
                        index=list(labels).index(default) if default in labels else 0)
    csv_path = Path(pick).parent / "summary.csv"
    if csv_path.exists():
        overview = eval_overview(csv_path)
        order = {v["system"]: i for i, v in enumerate(TEAM_OPTIONS.values())}
        overview.sort(key=lambda r: order.get(r["system"], 99))
        with st.container(horizontal=True):
            for r in overview:
                with st.container(border=True):
                    st.markdown(f"**{NAMES.get(r['system'], r['system'])}**")
                    st.metric("Root cause correct", f"{r['accuracy']:.0%}")
                    st.caption(f"Right outcome {r['outcome']:.0%}  \n${r['cost']:.3f} per incident · "
                               f"{r['latency']:.0f}s · {r['tools']:.0f} tool calls · {r['runs']} runs")
        chart = pd.DataFrame({"team": [NAMES.get(r["system"], r["system"]) for r in overview],
                              "cost per incident ($)": [r["cost"] for r in overview]}).set_index("team")
        st.caption("Cost per incident")
        st.bar_chart(chart, height=220)
    with st.expander("Full tables", icon=":material/table_chart:"):
        st.markdown(Path(pick).read_text())

with st.expander("Run a new evaluation", icon=":material/science:"):
    st.caption("Runs every selected incident with every selected team. Real OpenAI cost; a hard cap stops it.")
    teams = st.pills("Teams", list(TEAM_OPTIONS), default=list(TEAM_OPTIONS), selection_mode="multi")
    titles = [sc["title"] for sc in common.scenarios()]
    picked = st.multiselect("Incidents", titles, default=titles)
    est = sum(TEAM_OPTIONS[t]["cost"] for t in teams) * len(picked)
    cap = st.number_input("Stop when spending reaches (USD)", min_value=0.1, value=max(1.0, round(est * 1.3, 1)),
                          step=0.5)
    busy = "job" in st.session_state
    if st.button(f"Run evaluation · about ${est:.2f}", icon=":material/play_arrow:",
                 disabled=busy or not (teams and picked and s.openai_api_key)):
        from rootcause_agents.evaluation import run_matrix

        out = root / f"ui-{datetime.now():%Y%m%d-%H%M%S}"
        paths = [sc["path"] for sc in common.scenarios() if sc["title"] in picked]
        systems = [TEAM_OPTIONS[t]["system"] for t in teams]

        async def go(echo):
            return await run_matrix(systems, paths, s, out_dir=out, progress=echo, max_cost_usd=cap)

        common.submit("Running evaluation", go, kind="eval", summary=str(out / "summary.md"))
        st.rerun()
    job = st.session_state.get("job")
    if job is not None and job.context.get("kind") == "eval":
        common.job_progress()
