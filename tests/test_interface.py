"""The web UI is the only user interface: no command-line tool, and the launcher starts Streamlit directly."""

import importlib.util
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_there_is_no_cli_module_or_console_script():
    assert importlib.util.find_spec("rootcause_agents.cli") is None
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert "scripts" not in project
    assert not any(d.startswith(("typer", "rich")) for d in project["dependencies"])


def test_launchers_start_the_web_ui_directly():
    app = "src/rootcause_agents/ui/app.py"
    assert f"streamlit run {app}" in (ROOT / "start-ui.command").read_text()
    assert f"streamlit run {app}" in (ROOT / "Makefile").read_text()


def test_docs_do_not_describe_cli_commands():
    for doc in ("README.md", "CLAUDE.md"):
        text = (ROOT / doc).read_text()
        assert "rootcause-agents investigate" not in text and "rootcause-agents eval" not in text, doc
