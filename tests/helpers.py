"""Scripted 'brains' for offline end-to-end tests. They react to prompt content generically."""

from __future__ import annotations

import asyncio
import re

from langchain_core.messages import AIMessage, ToolMessage

from rootcause_agents.llm import ScriptedLLM
from rootcause_agents.schemas import (
    BaselineConclusion,
    Claim,
    CriticVerdict,
    EvidenceRequest,
    Hypothesis,
    HypothesisSet,
    InvestigationPlan,
    RecommendedAction,
    ReportDraft,
    SpecialistFindings,
    Task,
)


def text_of(messages) -> str:
    return "\n".join(str(m.content) for m in messages)


def ids_in(messages) -> list[str]:
    """Evidence IDs that really exist: ledger lines ('- E-3 | ...') or tool results ('"evidence_id": "E-3"')."""
    seen, out = set(), []
    text = text_of(messages)
    found = re.findall(r"^- (E-\d+) \|", text, flags=re.MULTILINE) + re.findall(r'"evidence_id": "(E-\d+)"', text)
    for i in found:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def round_in(messages) -> int:
    m = re.findall(r"Round:\s*(\d+)", text_of(messages))
    return int(m[-1]) if m else 1


def tc(name, args, id):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


def specialist_steps(tool_calls_fn):
    """First turn: tool calls (vary by round so they are not memoised); after tool results: final answer."""

    def step(messages, tools):
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="Findings complete.")
        return AIMessage(content="", tool_calls=tool_calls_fn(round_in(messages)))

    return step


def obs_calls(r):
    return [tc("query_logs", {"service": "payments", "level": "ERROR", "start": f"-{60 + r}m"}, f"o1-{r}"),
            tc("query_metrics", {"service": "payments-db", "metric": "db_connections", "start": f"-{120 + r}m"},
               f"o2-{r}")]


def dep_calls(r):
    return [tc("get_deployments", {"start": f"-{180 - r}m"}, f"d1-{r}")]


def hypothesis_set(messages, schema):
    ids = ids_in(messages)
    return HypothesisSet(hypotheses=[
        Hypothesis(id="H1", title="DB pool exhaustion", root_cause_component="payments-db",
                   root_cause_category="resource_exhaustion",
                   mechanism=[Claim(text="pool saturated", kind="inference", evidence_ids=ids[:1])],
                   supporting_evidence_ids=ids[:2], contradicting_evidence_ids=[], unexplained_symptoms=[],
                   confidence=0.8),
        Hypothesis(id="H2", title="Bad checkout deploy", root_cause_component="checkout",
                   root_cause_category="bad_deploy", mechanism=[], supporting_evidence_ids=ids[-1:],
                   contradicting_evidence_ids=ids[:1], unexplained_symptoms=["pool errors"], confidence=0.1),
    ])


def findings(messages, schema):
    return SpecialistFindings(key_observations=[Claim(text="obs", kind="fact", evidence_ids=ids_in(messages)[:2])])


def report_draft(messages, schema):
    ids = ids_in(messages)
    return ReportDraft(summary="Pool exhaustion in payments-db.",
                       facts=[Claim(text="pool errors", kind="fact", evidence_ids=ids[:1])],
                       inferences=[Claim(text="db caused latency", kind="inference", evidence_ids=ids[:2])],
                       assumptions=[Claim(text="no traffic surge", kind="assumption")],
                       recommended_actions=[RecommendedAction(action="Raise pool size", type="mitigate_now",
                                                              priority="P0", evidence_ids=ids[:1])])


def plan(messages, schema):
    return InvestigationPlan(rationale="split by source", tasks=[
        Task(agent="observability", objective="Timeline of errors/latency", services=["checkout", "payments"], rationale="timeline"),
        Task(agent="dependency", objective="Recent changes", services=["checkout"], rationale="changes"),
    ])


ACCEPT = CriticVerdict(decision="ACCEPT", issues=[])
MORE = CriticVerdict(decision="NEED_MORE_EVIDENCE", issues=["temporal order unclear"], evidence_requests=[
    EvidenceRequest(target_agent="observability", question="Did DB saturation precede the deploy?",
                    services=["payments-db"], discriminates="H1 vs H2")])


def make_llm(critic=None, reasoning=None, obs=None, dep=None, **overrides) -> ScriptedLLM:
    structured = {
        "orchestrator": plan,
        "observability": findings,
        "dependency": findings,
        "reasoning": reasoning or hypothesis_set,
        "critic": critic if critic is not None else [ACCEPT],
        "reporter": report_draft,
    }
    structured.update(overrides)
    return ScriptedLLM(
        structured=structured,
        tool_steps={"observability": obs or specialist_steps(obs_calls),
                    "dependency": dep or specialist_steps(dep_calls)},
    )


class SlowLLM(ScriptedLLM):
    """Adds latency to tool steps so we can prove specialists run concurrently."""

    delay = 0.25

    async def tool_step(self, role, messages, tools):
        await asyncio.sleep(self.delay)
        return await super().tool_step(role, messages, tools)


def baseline_llm():
    def step(messages, tools):
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="I have enough evidence.")
        return AIMessage(content="", tool_calls=obs_calls(1) + dep_calls(1))

    def conclusion(messages, schema):
        hs = hypothesis_set(messages, HypothesisSet)
        return BaselineConclusion(hypotheses=hs.hypotheses, conclusive=True, report=report_draft(messages, ReportDraft))

    return ScriptedLLM(structured={"single": conclusion}, tool_steps={"single": step})
