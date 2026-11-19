"""Prompt loading and compact formatting of shared state for agent prompts."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from rootcause_agents.context import RunContext
from rootcause_agents.schemas import CriticVerdict, Evidence, Hypothesis, IncidentReport, SpecialistFindings
from rootcause_agents.sim.generator import DEFAULT_METRICS

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


@cache
def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text()


def system_prompt(ctx: RunContext, name: str) -> str:
    """Stable, cacheable prefix: role instructions + static environment context (never incident-specific).

    OpenAI caches identical prompt prefixes, so everything that does not change between calls/incidents goes
    here, and all dynamic content (incident, evidence, critiques) goes in the user message after it.
    """
    return f"{load_prompt(name)}\n\n## Environment\n{fmt_services(ctx)}"


def fmt_incident(inc: IncidentReport) -> str:
    return (f"INCIDENT {inc.id}: {inc.title}\nReported at: {inc.reported_at.isoformat()}\n"
            f"Description: {inc.description}\nAffected services: {', '.join(inc.affected_services)}\n"
            f"Symptoms: " + "; ".join(inc.symptoms))


def fmt_human(ctx: RunContext) -> str:
    """Human-in-the-loop input. A lead to verify with tools, never evidence (it cannot be cited)."""
    if not ctx.human_inputs:
        return ""
    return ("HUMAN INPUT from the on-call engineer (a lead to verify with tools; NOT evidence, cannot be cited):\n"
            + "\n".join(f"- {h}" for h in ctx.human_inputs))


def fmt_services(ctx: RunContext) -> str:
    """Static environment description (cache-friendly): services, tiers and which metrics can be queried."""
    topo = ctx.store.topology

    def one(name: str, spec) -> str:
        extra = f"; extra metrics: {', '.join(spec.metrics)}" if spec.metrics else ""
        return f"{name} ({spec.tier}{extra})"

    return ("Known services: " + ", ".join(one(n, s) for n, s in sorted(topo.services.items()))
            + "\nMetrics on every service: " + ", ".join(DEFAULT_METRICS))


def fmt_evidence(evidence: list[Evidence], limit: int = 80) -> str:
    if not evidence:
        return "EVIDENCE LEDGER: (empty)"
    lines = ["EVIDENCE LEDGER (id | tool | status | summary):"]
    for e in sorted(evidence, key=lambda e: int(e.id.split("-")[1]))[-limit:]:
        flag = "" if e.status == "ok" else f" [{e.status}, completeness {e.completeness:.0%}]"
        warn = f" WARN: {'; '.join(e.warnings)}" if e.warnings else ""
        lines.append(f"- {e.id} | {e.source_tool} | {e.status}{flag} | {e.summary}{warn}")
    return "\n".join(lines)


def fmt_findings(findings: list[SpecialistFindings]) -> str:
    if not findings:
        return "SPECIALIST FINDINGS: (none)"
    lines = ["SPECIALIST FINDINGS:"]
    for f in findings:
        for c in f.key_observations:
            lines.append(f"- obs: {c.text} {c.evidence_ids}")
        for c in f.timeline:
            lines.append(f"- timeline: {c.text} {c.evidence_ids}")
        for g in f.gaps:
            lines.append(f"- gap: {g}")
        for c in f.conflicts:
            lines.append(f"- conflict: {c}")
    return "\n".join(lines)


def fmt_gaps(gaps: list[str]) -> str:
    uniq = list(dict.fromkeys(gaps))
    return "DATA GAPS:\n" + ("\n".join(f"- {g}" for g in uniq) if uniq else "- none")


def fmt_hypotheses(hyps: list[Hypothesis]) -> str:
    return "HYPOTHESES:\n" + "\n".join(h.model_dump_json() for h in hyps) if hyps else "HYPOTHESES: (none)"


def fmt_verdicts(verdicts: list[CriticVerdict]) -> str:
    if not verdicts:
        return "PREVIOUS CRITIQUES: (none)"
    lines = ["PREVIOUS CRITIQUES:"]
    for i, v in enumerate(verdicts, 1):
        lines.append(f"- round {i}: {v.decision}; issues: {'; '.join(v.issues)}")
        for r in v.evidence_requests:
            lines.append(f"  - request ({r.target_agent}, {r.services}): {r.question}")
    return "\n".join(lines)
