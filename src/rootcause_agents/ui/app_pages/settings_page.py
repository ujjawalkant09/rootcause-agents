"""Settings: connections, models and limits (read from .env)."""

import streamlit as st

from rootcause_agents.ui import common

s = common.settings()


def status(ok: bool, good: str, bad: str) -> None:
    st.badge(good if ok else bad, color="green" if ok else "orange",
             icon=":material/check_circle:" if ok else ":material/warning:")


st.caption("Configured in `.env`. Restart the app after changing it.")
with st.container(border=True):
    st.markdown("**Connections**")
    with st.container(horizontal=True):
        status(bool(s.openai_api_key), "OpenAI key loaded", "OpenAI key missing")
        status(bool(s.database_url), "Sessions saved in Postgres", "Sessions in memory only")
        status(bool(s.langfuse_public_key and s.langfuse_secret_key), "Langfuse tracing on", "Langfuse off")
    if s.langfuse_public_key:
        st.link_button("Open Langfuse", s.langfuse_base_url, icon=":material/monitoring:")
    if s.database_url:
        st.caption("Postgres runs in Docker: `docker compose up -d`.")
with st.container(border=True):
    st.markdown("**Models**")
    st.markdown(f"- Main model: `{s.main_model}`\n- Budget model (evidence-gathering agents): `{s.worker_model}`\n"
                f"- Fallbacks when a model fails: `{', '.join(s.fallback_models) or 'none'}`")
with st.container(border=True):
    st.markdown("**Limits per investigation**")
    b = s.budget
    st.markdown(f"- Rounds: {b.max_rounds} · tool calls: {b.max_tool_calls} ({b.max_tool_calls_per_task} per task)\n"
                f"- Tokens: {b.max_total_tokens:,} · time: {b.max_wall_clock_s:.0f}s\n"
                "- The agents ask you at most 2 questions per investigation.")
