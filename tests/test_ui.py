"""Streamlit UI: pure view helpers (unit tests) + an offline smoke test of the page (no API calls on load)."""

import json
from pathlib import Path

from rootcause_agents.ui.view import (
    format_event,
    list_eval_results,
    list_runs,
    outcome_badge,
    scenario_options,
)

ROOT = Path(__file__).resolve().parents[1]


def test_scenario_options_lists_all_scenarios_with_titles():
    opts = scenario_options(ROOT / "data")
    assert len(opts) >= 10
    first = opts[0]
    assert first["id"] == "S01" and first["title"] and isinstance(first["traits"], list)
    assert first["label"].startswith("S01 · ")


def test_format_event_covers_live_steps_and_hides_noise():
    tool = format_event({"event": "tool_call", "t": 1.0, "agent": "observability", "tool": "query_logs",
                         "params": {"service": "payments"}, "status": "ok", "evidence_id": "E-3"})
    assert tool and "observability" in tool and "query_logs" in tool and "E-3" in tool
    verdict = format_event({"event": "verdict", "t": 2.0, "decision": "ACCEPT", "issues": ["x"]})
    assert verdict and "ACCEPT" in verdict
    fb = format_event({"event": "llm_fallback", "t": 3.0, "role": "critic", "from_model": "a", "to_model": "b"})
    assert fb and "a" in fb and "b" in fb
    assert format_event({"event": "llm_call", "t": 1.0, "role": "x"}) is None  # shown in the cost table instead


def test_outcome_badge():
    assert outcome_badge("ROOT_CAUSE_IDENTIFIED")[1] == "green"
    assert outcome_badge("PROBABLE")[1] == "orange"
    assert outcome_badge("INCONCLUSIVE")[1] == "gray"


def test_list_runs_and_eval_results(tmp_path):
    run = tmp_path / "runs" / "20260101-000000-S01-mas"
    run.mkdir(parents=True)
    (run / "report.json").write_text(json.dumps({"outcome": "PROBABLE", "system": "mas",
                                                 "stats": {"cost_usd": 0.1, "latency_s": 9.0}}))
    (run / "report.md").write_text("# r")
    runs = list_runs(tmp_path / "runs")
    assert runs[0]["name"] == run.name and runs[0]["outcome"] == "PROBABLE" and runs[0]["cost_usd"] == 0.1

    ev = tmp_path / "eval" / "pilot"
    ev.mkdir(parents=True)
    (ev / "summary.md").write_text("# Evaluation results")
    assert list_eval_results(tmp_path / "eval") == [ev / "summary.md"]
    assert list_runs(tmp_path / "missing") == [] and list_eval_results(tmp_path / "missing") == []


def test_app_renders_offline_without_calling_the_api():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "src/rootcause_agents/ui/app.py"), default_timeout=30).run()
    assert not at.exception
    assert any("Run investigation" in b.label for b in at.button)
    assert at.selectbox and any("S01" in str(o) for o in at.selectbox[0].options)


def test_cli_has_ui_command():
    from typer.testing import CliRunner

    from rootcause_agents.cli import app

    res = CliRunner().invoke(app, ["ui", "--help"])
    assert res.exit_code == 0 and "browser" in res.output.lower()


def test_question_panel_renders_for_a_paused_session():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "src/rootcause_agents/ui/app.py"), default_timeout=30)
    at.session_state["pending"] = {
        "thread_id": "t-1", "scenario": None, "custom": False,
        "question": {"reason": "stuck", "question": "The investigation is stuck. Any hint?", "round": 1,
                     "critic_decision": "INCONCLUSIVE", "critic_issues": ["cannot separate DB from deploy"],
                     "hypotheses": [{"component": "payments-db", "category": "resource_exhaustion",
                                     "confidence": 0.55, "title": "pool exhaustion"}], "human_rounds_left": 2}}
    at.run()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert "Investigate further with my hint" in labels and "Write the report now" in labels
    assert any("The investigation is stuck" in i.value for i in at.info)
