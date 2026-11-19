"""Thin wrapper: `uv run python eval/run_eval.py [--systems ...]` == `uv run rootcause-agents eval ...`."""

from rootcause_agents.cli import app

if __name__ == "__main__":
    app(["eval", *__import__("sys").argv[1:]])
