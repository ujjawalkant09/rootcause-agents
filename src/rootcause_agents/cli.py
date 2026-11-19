"""Command-line interface: list scenarios, run one investigation, run the evaluation matrix."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from rootcause_agents.config import SYSTEMS, Settings
from rootcause_agents.sim.scenario import load_scenario

app = typer.Typer(add_completion=False, help="Multi-agent production incident investigation (LangGraph + OpenAI).")
console = Console()

_LIVE_EVENTS = {"plan", "round_start", "tool_call", "tool_error", "findings", "hypotheses", "verdict", "route",
                "node_error", "report"}


def _echo(ev: dict[str, Any]) -> None:
    kind = ev["event"]
    if kind not in _LIVE_EVENTS:
        return
    t = f"[dim]{ev['t']:6.1f}s[/dim]"
    if kind == "tool_call":
        console.print(f"{t} [cyan]{ev['agent']}[/cyan] → {ev['tool']}({ev['params']}) "
                      f"[{'green' if ev['status'] == 'ok' else 'yellow'}]{ev['status']}[/] {ev.get('evidence_id') or ''}")
    elif kind == "tool_error":
        console.print(f"{t} [red]tool error[/red] {ev['tool']} {ev['code']} (attempt {ev['attempt']})")
    elif kind == "plan":
        console.print(f"{t} [bold magenta]orchestrator[/bold magenta] round {ev['round']}: "
                      + "; ".join(f"{x['agent']}: {x['objective'][:70]}" for x in ev["tasks"]))
    elif kind == "hypotheses":
        console.print(f"{t} [bold]reasoning[/bold] attempt {ev['attempt']}: {ev.get('hypotheses', '')} "
                      f"{'[red]' + str(ev['errors']) + '[/red]' if ev.get('errors') else ''}")
    elif kind == "verdict":
        console.print(f"{t} [bold yellow]critic[/bold yellow] {ev['decision']}: {'; '.join(ev['issues'])[:200]}")
        for q in ev.get("requests", []):
            console.print(f"         ↳ request: {q}")
    elif kind == "node_error":
        console.print(f"{t} [red]node error[/red] {ev['node']}: {ev['error'][:150]}")
    elif kind == "route":
        console.print(f"{t} route: {'loop back for more evidence' if ev['loop'] else 'report'} "
                      f"(new evidence this round: {ev['new_evidence']})")
    elif kind == "report":
        console.print(f"{t} [bold green]report[/bold green] outcome={ev['outcome']} top={ev['top']}")


@app.command("list")
def list_scenarios() -> None:
    """List the synthetic test incidents."""
    settings = Settings.from_env()
    table = Table("ID", "Title", "Traits")
    for p in sorted((settings.data_dir / "scenarios").glob("*.yaml")):
        sc = load_scenario(p)
        table.add_row(sc.id, sc.incident.title, ", ".join(sc.traits))
    console.print(table)


@app.command()
def investigate(
    scenario: str = typer.Argument(..., help="Scenario ID (e.g. S03) or path to a scenario YAML"),
    system: str = typer.Option("mas", help=f"One of {SYSTEMS}"),
    quiet: bool = typer.Option(False, help="Do not stream the live trace"),
) -> None:
    """Run one investigation and print the report."""
    from rootcause_agents.report import render_markdown
    from rootcause_agents.runner import investigate as run

    try:
        report = asyncio.run(run(scenario, system, echo=None if quiet else _echo))
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None
    console.print(Markdown(render_markdown(report)))


@app.command("eval")
def evaluate(
    systems: str = typer.Option("mas,single,mas-cheap", help="Comma-separated systems"),
    scenarios: str = typer.Option("all", help="Comma-separated scenario IDs or 'all'"),
    repeats: int = typer.Option(1, min=1),
    concurrency: int = typer.Option(2, min=1),
    out: Path = typer.Option(Path("eval/results"), help="Output directory"),
) -> None:
    """Run the comparison matrix and write eval/results/summary.md."""
    from rootcause_agents.evaluation import run_matrix

    settings = Settings.from_env()
    if not settings.openai_api_key:
        console.print("[red]Error:[/red] OPENAI_API_KEY is not set (add it to .env)")
        raise typer.Exit(code=1)
    all_paths = sorted((settings.data_dir / "scenarios").glob("*.yaml"))
    paths = all_paths if scenarios == "all" else [
        p for p in all_paths if load_scenario(p).id in {s.strip() for s in scenarios.split(",")}]
    sys_list = [s.strip() for s in systems.split(",")]
    console.print(f"Running {len(paths)} scenarios × {sys_list} × {repeats}")
    asyncio.run(run_matrix(sys_list, paths, settings, repeats=repeats, out_dir=out, concurrency=concurrency,
                           progress=lambda m: console.print(m)))
    console.print(Markdown((out / "summary.md").read_text()))


if __name__ == "__main__":  # pragma: no cover
    app()
