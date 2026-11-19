#!/bin/bash
# Double-click this file (macOS) to open the RootCause Agents UI in your browser.
cd "$(dirname "$0")" || exit 1
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed: https://docs.astral.sh/uv/getting-started/installation/"; read -r; exit 1
fi
uv sync --quiet && uv run streamlit run src/rootcause_agents/ui/app.py
