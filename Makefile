.PHONY: setup test live lint demo eval

setup:
	uv sync

test:
	uv run pytest -q

live:
	uv run pytest -m live -q

lint:
	uv run ruff check src tests

demo:
	uv run rootcause-agents investigate S03 --system mas

eval:
	uv run rootcause-agents eval --systems mas,single,mas-cheap
