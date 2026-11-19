"""Pure helpers for the Streamlit UI (kept separate so they are unit-testable without Streamlit)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rootcause_agents.sim.scenario import load_scenario

OUTCOME_BADGES = {
    "ROOT_CAUSE_IDENTIFIED": ("Root cause identified", "green"),
    "PROBABLE": ("Probable root cause", "orange"),
    "INCONCLUSIVE": ("Inconclusive", "gray"),
}

SYSTEM_HELP = {
    "mas": "Multi-agent (6 agents, critic feedback loop) on the main model",
    "mas-cheap": "Multi-agent with the cheap model for the data-gathering agents",
    "single": "Single-agent baseline (one tool-calling agent)",
}


def scenario_options(data_dir: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted((data_dir / "scenarios").glob("*.yaml")):
        sc = load_scenario(p)
        out.append({"id": sc.id, "title": sc.incident.title, "traits": list(sc.traits), "path": p,
                    "label": f"{sc.id} · {sc.incident.title}", "scenario": sc})
    return out


def outcome_badge(outcome: str) -> tuple[str, str]:
    return OUTCOME_BADGES.get(outcome, (outcome, "gray"))


def format_event(ev: dict[str, Any]) -> str | None:
    """One Markdown line for the live activity log, or None for events not worth showing."""
    kind = ev.get("event")
    t = f"`{ev.get('t', 0):6.1f}s`"
    if kind == "tool_call":
        icon = "✅" if ev.get("status") == "ok" else "⚠️"
        params = ", ".join(f"{k}={v}" for k, v in (ev.get("params") or {}).items())
        return (f"{t} 🔧 **{ev.get('agent')}** → `{ev.get('tool')}({params[:90]})` {icon} {ev.get('status')}"
                f" {ev.get('evidence_id') or ''}")
    if kind == "tool_error":
        return f"{t} ⚠️ tool `{ev.get('tool')}` failed: {ev.get('code')} (attempt {ev.get('attempt')}), retrying"
    if kind == "plan":
        tasks = "; ".join(f"**{x['agent']}**: {x['objective'][:80]}" for x in ev.get("tasks", []))
        return f"{t} 🧭 **orchestrator** round {ev.get('round')}: {tasks}"
    if kind == "findings":
        return f"{t} 📋 **{ev.get('agent')}** finished: {len(ev.get('evidence', []))} evidence items"
    if kind == "hypotheses":
        hs = ", ".join(f"{c} / {cat} ({conf:.2f})" for c, cat, conf in ev.get("hypotheses", []))
        return f"{t} 🧠 **reasoning**: {hs}"
    if kind == "verdict":
        issues = "; ".join(ev.get("issues", []))[:220]
        return f"{t} 🧐 **critic** → **{ev.get('decision')}**: {issues}"
    if kind == "route":
        return f"{t} ↪️ {'loop back for more evidence' if ev.get('loop') else 'write the report'}"
    if kind == "llm_fallback":
        return f"{t} 🔁 **{ev.get('role')}**: model fallback {ev.get('from_model')} → {ev.get('to_model')}"
    if kind == "llm_error":
        return f"{t} ⚠️ **{ev.get('role')}**: {ev.get('model')} error: {str(ev.get('error'))[:120]}"
    if kind == "node_error":
        return f"{t} ❌ node `{ev.get('node')}` failed: {str(ev.get('error'))[:120]}"
    if kind == "report":
        return f"{t} 📝 **report** written: {ev.get('outcome')}"
    return None


def list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        return []
    out = []
    for d in sorted((p for p in runs_dir.iterdir() if (p / "report.json").exists()), reverse=True):
        try:
            r = json.loads((d / "report.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        s = r.get("stats", {})
        out.append({"name": d.name, "path": d, "outcome": r.get("outcome"), "system": r.get("system"),
                    "cost_usd": s.get("cost_usd"), "latency_s": s.get("latency_s"), "trace_url": s.get("trace_url")})
    return out


def list_eval_results(eval_dir: Path) -> list[Path]:
    if not eval_dir.exists():
        return []
    return sorted(eval_dir.rglob("summary.md"), key=lambda p: p.stat().st_mtime, reverse=True)
