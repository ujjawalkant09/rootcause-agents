"""Guard: agent code and prompts must not contain scenario IDs or ground-truth strings."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GUARDED = [ROOT / "src/rootcause_agents/agents", ROOT / "src/rootcause_agents/prompts", ROOT / "src/rootcause_agents/graph.py"]


def _guarded_text() -> str:
    parts = []
    for p in GUARDED:
        files = [p] if p.is_file() else sorted(p.rglob("*"))
        parts += [f.read_text() for f in files if f.is_file() and f.suffix in (".py", ".md")]
    return "\n".join(parts)


def test_no_scenario_ids_or_answers_in_agents():
    text = _guarded_text()
    scenario_files = sorted((ROOT / "data/scenarios").glob("*.yaml"))
    assert scenario_files, "no scenarios found"
    for f in scenario_files:
        sc = yaml.safe_load(f.read_text())
        assert not re.search(rf"\b{re.escape(sc['id'])}\b", text), f"scenario id {sc['id']} referenced"
        for d in sc.get("deployments", []):
            assert d["id"] not in text, f"deployment id {d['id']} referenced"
        for c in sc.get("config_changes", []):
            assert c["id"] not in text and c["key"] not in text, f"config change {c['id']} referenced"
        for s in sc.get("signals", []):
            if s["type"] == "log_burst":
                head = s["template"].split("{")[0].strip()[:25]
                assert head not in text, f"log template {head!r} referenced"
