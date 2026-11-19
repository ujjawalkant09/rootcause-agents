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


# ---------------------------------------------------------------- redesigned UI helpers
# Plain-language choices; cost/time measured in the evaluation (gpt-6.1-terra main, gpt-6-luna workers).
TEAM_OPTIONS: dict[str, dict[str, Any]] = {
    "Team of agents": {"system": "mas", "cost": 0.11, "minutes": 2.5,
                       "help": "Six agents plan, gather evidence in parallel, and a critic double-checks the answer."},
    "Team (budget)": {"system": "mas-cheap", "cost": 0.06, "minutes": 2.5,
                      "help": "Same team, with a cheaper model for the evidence-gathering agents."},
    "Single agent": {"system": "single", "cost": 0.04, "minutes": 1.0,
                     "help": "One agent with the same tools. Fastest and cheapest; no critic and no questions."},
}

STAGES = [("plan", "Plan the investigation"), ("gather", "Gather evidence"), ("reason", "Compare possible causes"),
          ("critic", "Critic review"), ("report", "Write the report")]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def stages_from_events(events: list[Any]) -> list[dict[str, Any]]:
    """Turn trace events into a five-stage timeline (label, state, detail) for non-technical readers.
    A new round restarts the stages after planning."""
    order = [k for k, _ in STAGES]
    state = dict.fromkeys(order, "pending")
    detail = dict.fromkeys(order, "")
    round_no, tools, retries, failed = 1, 0, 0, 0

    def reach(key: str) -> None:
        idx = order.index(key)
        for k in order[:idx]:
            if state[k] != "complete":
                state[k] = "complete"
        state[key] = "running"

    for ev in events:
        if not isinstance(ev, dict):
            continue
        kind = ev.get("event")
        if kind == "plan":
            round_no = ev.get("round", round_no)
            for k in order[1:]:
                state[k], detail[k] = "pending", ""
            tools = retries = failed = 0
            tasks = ev.get("tasks", [])
            state["plan"] = "complete"
            detail["plan"] = f"{_plural(len(tasks), 'task')}: " + "; ".join(
                f"{t.get('agent')} – {str(t.get('objective', ''))[:70]}" for t in tasks[:4]) if tasks else "plan ready"
        elif kind in ("tool_call", "tool_error", "findings"):
            reach("gather")
            if kind == "tool_call":
                tools += 1
                failed += ev.get("status") == "error"
            elif kind == "tool_error":
                retries += 1
            bits = [_plural(tools, "tool call")]
            if retries:
                bits.append(f"{_plural(retries, 'retry')} after tool failures".replace("retrys", "retries"))
            if failed:
                bits.append(f"{failed} failed")
            detail["gather"] = ", ".join(bits)
        elif kind == "hypotheses":
            reach("reason")
            hs = ev.get("hypotheses") or []
            detail["reason"] = "; ".join(f"{c} ({cat.replace('_', ' ')}, {conf:.0%})" for c, cat, conf in hs[:3])
            state["reason"] = "complete"
        elif kind == "verdict":
            reach("critic")
            d = ev.get("decision", "")
            detail["critic"] = {"ACCEPT": "Accepted the leading cause",
                                "NEED_MORE_EVIDENCE": "Asked for more evidence",
                                "INCONCLUSIVE": "Not enough evidence to conclude"}.get(d, d)
            state["critic"] = "complete"
        elif kind == "human_question":
            state["critic"] = "complete"
            detail["critic"] = (detail["critic"] + " · " if detail["critic"] else "") + "waiting for your input"
        elif kind in ("report", "run_end"):
            for k in order:
                state[k] = "complete"
            if kind == "report":
                detail["report"] = str(ev.get("outcome", "")).replace("_", " ").capitalize()

    labels = dict(STAGES)
    out = []
    for k in order:
        label = labels[k] + (f" (round {round_no})" if k == "plan" and round_no > 1 else "")
        out.append({"key": k, "label": label, "state": state[k], "detail": detail[k]})
    return out


def eval_overview(csv_path: Path) -> list[dict[str, Any]]:
    """Per-system averages from an evaluation summary.csv (booleans are stored as 'True'/'False' text)."""
    import csv

    def num(v: Any) -> float:
        if str(v).strip().lower() in ("true", "false"):
            return 1.0 if str(v).strip().lower() == "true" else 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    groups: dict[str, list[dict[str, Any]]] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            groups.setdefault(row["system"], []).append(row)
    out = []
    for system, rows in groups.items():
        def mean(field: str, rows=rows) -> float:
            vals = [num(r.get(field)) for r in rows]
            vals = [v for v in vals if v == v]
            return round(sum(vals) / len(vals), 4) if vals else float("nan")

        out.append({"system": system, "accuracy": mean("top1_correct"), "outcome": mean("outcome_correct"),
                    "cost": mean("cost_usd"), "latency": mean("latency_s"), "tools": mean("tool_calls"),
                    "runs": len(rows)})
    return out
