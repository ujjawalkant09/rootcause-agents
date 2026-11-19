"""RootCause Agents – web UI (Streamlit).

Start with:  uv run rootcause-agents ui   (or double-click start-ui.command on macOS)

Pages (top navigation):
- Investigate: describe what happened, choose how to investigate, watch progress, answer the agents, read the report
- Sessions: saved investigations; answer the agents, continue stopped runs, re-open reports
- Results: multi-agent vs single-agent comparison; run a new evaluation
- Settings: connections, models and limits
Nothing calls the API until a Start/Run button is pressed. Investigations run in a background thread, so using
the app while they run never stops them.
"""

from pathlib import Path

import streamlit as st

from rootcause_agents.ui import common

st.set_page_config(page_title="RootCause Agents", page_icon=":material/troubleshoot:", layout="centered")

PAGES_DIR = Path(__file__).with_name("app_pages")
waiting = common.waiting_count()
PAGES = {
    "investigate": st.Page(PAGES_DIR / "investigate.py", title="Investigate", icon=":material/troubleshoot:",
                           default=True, help="Describe an incident and let the agents find the root cause."),
    "sessions": st.Page(PAGES_DIR / "sessions.py", title=f"Sessions ({waiting})" if waiting else "Sessions",
                        icon=":material/forum:", url_path="sessions",
                        help="Saved investigations: answer the agents' questions, continue stopped runs, "
                             "re-open reports."),
    "results": st.Page(PAGES_DIR / "results.py", title="Results", icon=":material/leaderboard:",
                       url_path="results", help="How the team of agents compares with a single agent."),
    "settings": st.Page(PAGES_DIR / "settings_page.py", title="Settings", icon=":material/settings:",
                        url_path="settings", help="Connections, models and limits."),
}
common.PAGES = PAGES

page = st.navigation(list(PAGES.values()), position="top")
st.title("RootCause Agents" if page.title == "Investigate" else page.title.split(" (")[0], anchor=False)
page.run()
