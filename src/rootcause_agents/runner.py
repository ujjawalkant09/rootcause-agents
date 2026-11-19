"""Entry point: run one investigation (multi-agent or single-agent) and persist report + trace."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph.types import Command

from rootcause_agents.agents.single_agent import run_single
from rootcause_agents.budget import BudgetTracker
from rootcause_agents.config import SYSTEMS, Settings, cost_usd
from rootcause_agents.context import RunContext
from rootcause_agents.graph import HITL_MODES, build_graph
from rootcause_agents.llm import LLM, OpenAILLM, Usage
from rootcause_agents.report import render_markdown
from rootcause_agents.schemas import InvestigationReport
from rootcause_agents.sim.generator import generate
from rootcause_agents.sim.scenario import Scenario, load_history, load_scenario, load_topology
from rootcause_agents.telemetry import NoopTelemetry, make_telemetry
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


@dataclass
class SessionResult:
    """Outcome of one session segment: finished with a report, or paused waiting for human input."""
    status: str  # "done" | "waiting"
    thread_id: str
    report: InvestigationReport | None = None
    question: dict[str, Any] | None = None
    cost_usd: float = 0.0
    out_dir: Path | None = None


@dataclass
class _Run:
    ctx: RunContext
    scenario: Scenario
    telemetry: NoopTelemetry
    tracer: Tracer
    out_dir: Path
    run_id: str


def _build_run(scenario_path: Path, system: str, settings: Settings, *, llm: LLM | None, topology_path: Path | None,
               history_path: Path | str | None, out_dir: Path | None, sleep_scale: float,
               echo: Callable[[dict[str, Any]], None] | None, telemetry: NoopTelemetry | None,
               ticket: dict[str, Any] | None, run_id: str | None = None, resume: bool = False) -> _Run:
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {SYSTEMS}")
    scenario: Scenario = load_scenario(scenario_path)
    if ticket:
        allowed = {"title", "description", "symptoms", "affected_services"}
        scenario = scenario.model_copy(update={"incident": scenario.incident.model_copy(
            update={k: v for k, v in ticket.items() if k in allowed and v})})
    topology = load_topology(topology_path or settings.data_dir / "topology.yaml")
    if history_path == "default":
        history_path = settings.data_dir / "history" / "incidents.yaml"
    history = load_history(history_path) if history_path else []
    store = generate(scenario, topology, history)  # deterministic (seeded): identical after a resume

    run_id = run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{scenario.id}-{system}"
    out_dir = out_dir or settings.runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    telemetry = telemetry or make_telemetry(settings)

    def sink(ev: dict[str, Any]) -> None:
        telemetry.on_event(ev)
        if echo:
            echo(ev)

    tracer = Tracer(out_dir / "trace.jsonl", echo=sink, append=resume)
    budget = BudgetTracker(settings.budget.model_copy())

    def on_usage(role: str, model: str, u: Usage) -> None:
        c = cost_usd(model, u.input_tokens, u.output_tokens, u.cached_tokens)
        budget.add_usage(role, u, c)
        tracer.event("llm_call", role=role, model=model, input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                     cached_tokens=u.cached_tokens, cost_usd=c)

    if llm is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (add it to .env)")
        llm = OpenAILLM(settings, settings.models_for(system), on_usage=on_usage, on_event=tracer.event,
                        callbacks=telemetry.callbacks)
    elif hasattr(llm, "on_usage"):
        llm.on_usage = on_usage  # type: ignore[attr-defined]

    executor = ToolExecutor(store, FaultInjector(scenario.faults), budget, sleep_scale=sleep_scale,
                            on_event=tracer.event, observe=telemetry.observe if telemetry.enabled else None)
    ctx = RunContext(incident=scenario.incident_report(), store=store, executor=executor, llm=llm, budget=budget,
                     tracer=tracer, system=system, telemetry=telemetry)
    return _Run(ctx, scenario, telemetry, tracer, out_dir, run_id)


async def _drive(run: _Run, graph_input: Any, *, settings: Settings, hitl: str, checkpointer: Any, registry: Any,
                 session_id: str | None) -> SessionResult:
    """Run one segment and keep the session registry truthful even when the run dies midway."""
    try:
        return await _drive_inner(run, graph_input, settings=settings, hitl=hitl, checkpointer=checkpointer,
                                  registry=registry, session_id=session_id)
    except Exception as e:  # a real error: the last checkpoint is kept, so the session can continue later
        if registry is not None:
            await registry.aupsert(run.run_id, status="failed", error=f"{type(e).__name__}: {e}"[:500])
        raise
    except BaseException as e:  # stopped from outside (UI rerun, Ctrl-C, cancellation)
        if registry is not None:
            with contextlib.suppress(BaseException):  # best effort while shutting down
                await registry.aupsert(run.run_id, status="interrupted",
                                       error=f"{type(e).__name__}: run stopped before finishing"[:500])
        raise


async def _recover_routing(graph: Any, cfg: dict[str, Any], final: dict[str, Any], tracer: Tracer) -> dict[str, Any]:
    """A crash inside the routing after the critic leaves the critic's writes saved but no next step, so a resumed
    graph ends without a report. Re-apply the critic's routing (``update_state(as_node="critic")``) and continue."""
    snap = await graph.aget_state(cfg)
    if snap.next or not final.get("verdicts"):
        return final
    tracer.event("routing_recovered", after="critic")
    await graph.aupdate_state(cfg, {}, as_node="critic")
    return await graph.ainvoke(None, cfg)


async def _drive_inner(run: _Run, graph_input: Any, *, settings: Settings, hitl: str, checkpointer: Any,
                       registry: Any, session_id: str | None) -> SessionResult:
    ctx, telemetry, tracer = run.ctx, run.telemetry, run.tracer
    incident = ctx.incident.model_dump(mode="json", include={"id", "title", "description", "reported_at",
                                                             "affected_services", "symptoms"})
    with telemetry.investigation(run_id=run.run_id, system=ctx.system, scenario_id=run.scenario.id,
                                 title=ctx.incident.title, incident=incident, session_id=session_id):
        tracer.event("run_start" if isinstance(graph_input, dict) else "run_resume", run_id=run.run_id,
                     system=ctx.system, incident=ctx.incident.title, trace_url=telemetry.trace_url)
        if ctx.system == "single":
            report = await run_single(ctx)
        else:
            graph = build_graph(ctx, hitl=hitl, checkpointer=checkpointer)
            cfg = {"recursion_limit": 80, "max_concurrency": settings.max_concurrency,
                   "configurable": {"thread_id": run.run_id}}
            final = await graph.ainvoke(graph_input, cfg)
            if not final.get("__interrupt__") and "report" not in final and checkpointer is not None:
                final = await _recover_routing(graph, cfg, final, tracer)
            if final.get("__interrupt__"):
                question = final["__interrupt__"][0].value
                if registry is not None:
                    await registry.aupsert(run.run_id, status="waiting", question=question, error=None,
                                           cost_usd=round(ctx.budget.cost_usd, 6), trace_url=telemetry.trace_url)
                tracer.event("run_paused", reason=question.get("reason"), cost_usd=round(ctx.budget.cost_usd, 6))
                telemetry.pause(question)
                return SessionResult("waiting", run.run_id, question=question, cost_usd=ctx.budget.cost_usd,
                                     out_dir=run.out_dir)
            if "report" not in final:
                raise RuntimeError("investigation graph ended without a report or a question")
            report = final["report"]
        if telemetry.trace_url:
            report = report.model_copy(update={"stats": report.stats.model_copy(
                update={"trace_url": telemetry.trace_url})})
        telemetry.finish(report)

    (run.out_dir / "report.json").write_text(report.model_dump_json(indent=2))
    (run.out_dir / "report.md").write_text(render_markdown(report))
    tracer.event("run_end", outcome=report.outcome, cost_usd=report.stats.cost_usd, latency_s=report.stats.latency_s)
    if registry is not None:
        await registry.aupsert(run.run_id, status="done", outcome=report.outcome, question=None, error=None,
                               cost_usd=report.stats.cost_usd, trace_url=report.stats.trace_url)
    return SessionResult("done", run.run_id, report=report, cost_usd=report.stats.cost_usd, out_dir=run.out_dir)


async def start_session(
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
    telemetry: NoopTelemetry | None = None,
    session_id: str | None = None,
    ticket: dict[str, Any] | None = None,
    hitl: str = "off",
    checkpointer: Any = None,
    registry: Any = None,
) -> SessionResult:
    """Start an investigation. With a checkpointer every step is persisted (thread_id = run id); with
    ``hitl`` = "stuck"/"review" the graph may pause and return status "waiting" with a question."""
    if hitl not in HITL_MODES:
        raise ValueError(f"hitl must be one of {HITL_MODES}")
    if hitl != "off" and checkpointer is None:
        raise ValueError("human-in-the-loop needs a checkpointer (see persistence.open_checkpointer)")
    settings = settings or Settings.from_env()
    path = resolve_scenario(scenario_ref, settings)
    run = _build_run(path, system, settings, llm=llm, topology_path=topology_path, history_path=history_path,
                     out_dir=out_dir, sleep_scale=sleep_scale, echo=echo, telemetry=telemetry, ticket=ticket)
    if registry is not None:
        await registry.aupsert(run.run_id, status="running", system=system, scenario_id=run.scenario.id,
                               title=run.ctx.incident.title, out_dir=str(run.out_dir))
    meta = {"scenario_path": str(path), "system": system, "ticket": ticket, "run_id": run.run_id,
            "out_dir": str(run.out_dir), "hitl": hitl}
    return await _drive(run, {"round": 0, "meta": meta}, settings=settings, hitl=hitl, checkpointer=checkpointer,
                        registry=registry, session_id=session_id or (run.run_id if hitl != "off" else None))


async def resume_session(
    thread_id: str,
    answer: dict[str, Any] | None,
    *,
    settings: Settings | None = None,
    llm: LLM | None = None,
    topology_path: Path | None = None,
    history_path: Path | str | None = "default",
    checkpointer: Any,
    registry: Any = None,
    sleep_scale: float = 1.0,
    echo: Callable[[dict[str, Any]], None] | None = None,
    telemetry: NoopTelemetry | None = None,
) -> SessionResult:
    """Resume a session from its last checkpoint; the run context is rebuilt from it (works from a new process).

    - paused for human input: pass ``answer`` = {"action": "continue"|"report", "hint": str};
    - interrupted or failed midway: pass ``answer=None`` to continue where it stopped (finished steps are not
      re-run; the pending writes of the last step are kept by LangGraph).
    """
    settings = settings or Settings.from_env()
    saved = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
    if saved is None:
        raise KeyError(f"No saved session {thread_id!r}")
    state = saved.checkpoint["channel_values"]
    writes = list(saved.pending_writes or [])
    waiting = any(ch == "__interrupt__" for _, ch, _ in writes)
    if waiting and answer is None:
        raise ValueError(f"Session {thread_id!r} is waiting for an answer from the on-call engineer")
    # the last step may have finished without a new checkpoint: its pending writes hold the latest counters
    meta = state["meta"]
    snap = next((v for _, ch, v in reversed(writes) if ch == "snapshot"), state.get("snapshot", {}))
    pending_evidence = [e for _, ch, v in writes if ch == "evidence" for e in (v or [])]
    run = _build_run(Path(meta["scenario_path"]), meta["system"], settings, llm=llm, topology_path=topology_path,
                     history_path=history_path, out_dir=Path(meta["out_dir"]), sleep_scale=sleep_scale, echo=echo,
                     telemetry=telemetry, ticket=meta.get("ticket"), run_id=thread_id, resume=True)
    ctx = run.ctx
    ctx.budget.restore(snap.get("budget", {}))
    ctx.executor._next_id = max([snap.get("next_evidence_id", 0)]
                                + [int(e.id.split("-")[1]) for e in [*state.get("evidence", []), *pending_evidence]])
    ctx.executor.data_gaps.extend(snap.get("data_gaps", []))
    ctx.executor.errors = snap.get("tool_errors", 0)
    ctx.degraded_nodes.extend(snap.get("degraded_nodes", []))
    ctx.human_inputs.extend(state.get("human_inputs", []))
    if registry is not None:
        await registry.aupsert(thread_id, status="running", error=None)
    graph_input = Command(resume=answer) if answer is not None else None
    return await _drive(run, graph_input, settings=settings, hitl=meta.get("hitl", "off"),
                        checkpointer=checkpointer, registry=registry, session_id=thread_id)


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
    telemetry: NoopTelemetry | None = None,
    session_id: str | None = None,
    ticket: dict[str, Any] | None = None,
) -> InvestigationReport:
    """Run one unattended investigation (no human-in-the-loop) and return its report. ``ticket`` overrides the
    incident text (title, description, symptoms, affected_services) for a custom ticket."""
    res = await start_session(scenario_ref, system, settings=settings, llm=llm, topology_path=topology_path,
                              history_path=history_path, out_dir=out_dir, sleep_scale=sleep_scale, echo=echo,
                              telemetry=telemetry, session_id=session_id, ticket=ticket)
    assert res.report is not None
    return res.report
