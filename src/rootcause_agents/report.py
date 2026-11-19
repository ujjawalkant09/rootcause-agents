"""Render an InvestigationReport as Markdown."""

from __future__ import annotations

from rootcause_agents.schemas import Claim, InvestigationReport


def _claims(claims: list[Claim]) -> str:
    if not claims:
        return "_none_\n"
    return "".join(f"- {c.text}" + (f" `[{', '.join(c.evidence_ids)}]`" if c.evidence_ids else "") + "\n"
                   for c in claims)


def render_markdown(r: InvestigationReport) -> str:
    s = r.stats
    out = [
        f"# Incident investigation – {r.incident_id}\n",
        f"**System:** `{r.system}` · **Outcome:** **{r.outcome}** · rounds {s.rounds} · tool calls {s.tool_calls} "
        f"({s.tool_errors} errors) · LLM calls {s.llm_calls} · tokens {s.input_tokens}+{s.output_tokens} "
        f"({s.cached_input_tokens} cached) · "
        f"${s.cost_usd:.4f} · {s.latency_s:.1f}s\n",
        f"## Summary\n{r.summary}\n",
        "## Ranked hypotheses\n",
        "| # | Root cause | Component | Category | Confidence | Supporting | Contradicting |\n|---|---|---|---|---|---|---|\n",
    ]
    for h in r.ranked_hypotheses:
        out.append(f"| {h.id} | {h.title} | `{h.root_cause_component}` | {h.root_cause_category} | {h.confidence:.2f} | "
                   f"{', '.join(h.supporting_evidence_ids) or '–'} | {', '.join(h.contradicting_evidence_ids) or '–'} |\n")
    for h in r.ranked_hypotheses:
        out.append(f"\n**{h.id} mechanism:**\n" + _claims(h.mechanism))
        if h.unexplained_symptoms:
            out.append("Unexplained: " + "; ".join(h.unexplained_symptoms) + "\n")
    out += ["\n## Facts (evidence-backed)\n", _claims(r.facts),
            "\n## Inferences\n", _claims(r.inferences),
            "\n## Assumptions (not evidence-backed)\n", _claims(r.assumptions),
            "\n## Data gaps\n", "".join(f"- {g}\n" for g in r.data_gaps) or "_none_\n",
            "\n## Recommended actions\n"]
    for a in r.recommended_actions:
        ev = f" `[{', '.join(a.evidence_ids)}]`" if a.evidence_ids else ""
        out.append(f"- **{a.priority}** ({a.type}) {a.action}{ev}\n")
    if r.human_inputs:
        out += ["\n## Human input (on-call engineer, not evidence)\n", "".join(f"- {h}\n" for h in r.human_inputs)]
    if r.critic_issues:
        out += ["\n## Critic issues raised\n", "".join(f"- {i}\n" for i in r.critic_issues)]
    out.append("\n## Evidence ledger\n| ID | Tool | Status | Summary |\n|---|---|---|---|\n")
    for e in r.evidence:
        out.append(f"| {e.id} | {e.source_tool} | {e.status} | {e.summary.replace('|', '/')} |\n")
    if s.by_role:
        out.append("\n## Token usage by agent\n| Agent | Calls | Input | Cached | Output | Cost ($) |\n|---|---|---|---|---|---|\n")
        for u in s.by_role:
            out.append(f"| {u.role} | {u.calls} | {u.input_tokens} | {u.cached_tokens} | {u.output_tokens} | "
                       f"{u.cost_usd:.4f} |\n")
        if s.llm_fallbacks:
            out.append(f"\nModel fallbacks used: {s.llm_fallbacks}\n")
    if s.degraded_nodes:
        out.append("\n**Degraded nodes:** " + ", ".join(s.degraded_nodes) + "\n")
    return "".join(out)
