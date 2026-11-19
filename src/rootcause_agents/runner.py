"""Entry point: run one investigation (multi-agent or single-agent) and persist report + trace."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from rootcause_agents.agents.single_agent import run_single
from rootcause_agents.budget import BudgetTracker
from rootcause_agents.config import SYSTEMS, Settings, cost_usd
from rootcause_agents.context import RunContext
from rootcause_agents.graph import build_graph
from rootcause_agents.llm import LLM, OpenAILLM, Usage
from rootcause_agents.report import render_markdown
from rootcause_agents.schemas import InvestigationReport
from rootcause_agents.sim.generator import generate
from rootcause_agents.sim.scenario import Scenario, load_history, load_scenario, load_topology
from rootcause_agents.tools.executor import ToolExecutor
from rootcause_agents.tools.faults import FaultInjector
from rootcause_agents.tracing import Tracer


def resolve_scenario(ref: str | Path, settings: Settings) -> Path:
    p = Path(ref)
    if p.exists():
        return p
    matches = sorted((settings.data_dir / "scenarios").glob(f"{ref}*.yaml"))
    if not matches:
        raise FileNotFoundError(f"Scenario {ref!r} not found in {settings.data_dir / 'scenarios'}")
    return matches[0]


async def investigate(
    scenario_ref: str | Path,
    system: str = "mas",
    *,
    settings: Settings | None = None,
    llm: LLM | None = None,
    topology_path: Path | None = None,
    history_path: Path | str | None = "default",
    out_dir: Path | None = None,
    sleep_scale: float = 1.0,
    echo: Callable[[dict[str, Any]], None] | None = None,
) -> InvestigationReport:
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {SYSTEMS}")
    settings = settings or Settings.from_env()
    scenario: Scenario = load_scenario(resolve_scenario(scenario_ref, settings))
    topology = load_topology(topology_path or settings.data_dir / "topology.yaml")
    if history_path == "default":
        history_path = settings.data_dir / "history" / "incidents.yaml"
    history = load_history(history_path) if history_path else []
    store = generate(scenario, topology, history)

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{scenario.id}-{system}"
    out_dir = out_dir or settings.runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tracer = Tracer(out_dir / "trace.jsonl", echo=echo)
    budget = BudgetTracker(settings.budget.model_copy())

    def on_usage(role: str, model: str, u: Usage) -> None:
        c = cost_usd(model, u.input_tokens, u.output_tokens, u.cached_tokens)
        budget.add_usage(role, u, c)
        tracer.event("llm_call", role=role, model=model, input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                     cached_tokens=u.cached_tokens, cost_usd=c)

    if llm is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (add it to .env)")
        llm = OpenAILLM(settings, settings.models_for(system), on_usage=on_usage, on_event=tracer.event)
    elif hasattr(llm, "on_usage"):
        llm.on_usage = on_usage  # type: ignore[attr-defined]

    executor = ToolExecutor(store, FaultInjector(scenario.faults), budget, sleep_scale=sleep_scale,
                            on_event=tracer.event)
    ctx = RunContext(incident=scenario.incident_report(), store=store, executor=executor, llm=llm, budget=budget,
                     tracer=tracer, system=system)
    tracer.event("run_start", run_id=run_id, system=system, incident=ctx.incident.title)

    if system == "single":
        report = await run_single(ctx)
    else:
        graph = build_graph(ctx)
        final = await graph.ainvoke({"round": 0}, {"recursion_limit": 60,
                                                   "max_concurrency": settings.max_concurrency})
        report = final["report"]

    (out_dir / "report.json").write_text(report.model_dump_json(indent=2))
    (out_dir / "report.md").write_text(render_markdown(report))
    tracer.event("run_end", outcome=report.outcome, cost_usd=report.stats.cost_usd, latency_s=report.stats.latency_s)
    return report
