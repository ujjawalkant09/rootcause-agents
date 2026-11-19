.PHONY: setup app test lint live

setup:  ## install dependencies
	uv sync

app:  ## open the web UI (same as double-clicking start-ui.command)
	uv run streamlit run src/rootcause_agents/ui/app.py

test:  ## offline test suite (no API key needed)
	uv run pytest -q

lint:
	uv run ruff check src tests

live:  ## real-API smoke tests (needs OPENAI_API_KEY)
	uv run pytest -m live -q
