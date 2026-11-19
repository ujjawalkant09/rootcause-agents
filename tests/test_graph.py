"""End-to-end tests of the LangGraph multi-agent pipeline and the single-agent baseline (offline)."""

import time

from rootcause_agents.budget import Budget
from rootcause_agents.config import Settings
from rootcause_agents.llm import StructuredOutputError
from rootcause_agents.runner import investigate
from rootcause_agents.schemas import CriticVerdict
from tests.conftest import FIXTURES
from tests.helpers import (
    ACCEPT,
    MORE,
    SlowLLM,
    baseline_llm,
    hypothesis_set,
    make_llm,
    obs_calls,
    specialist_steps,
)


def settings(**budget):
    return Settings(openai_api_key="sk-test", budget=Budget(**budget))


async def run(llm, system="mas", tmp_path=None, **budget):
    return await investigate(
        FIXTURES / "scenario_basic.yaml", system, settings=settings(**budget), llm=llm,
        topology_path=FIXTURES / "topology.yaml", history_path=None, out_dir=tmp_path, sleep_scale=0.0,
    )


async def test_happy_path_single_round(tmp_path):
    llm = make_llm()
    report = await run(llm, tmp_path=tmp_path)
    assert report.outcome == "ROOT_CAUSE_IDENTIFIED"
    assert report.stats.degraded_nodes == []  # no silent fallbacks on the happy path
    assert report.stats.rounds == 1
    assert report.ranked_hypotheses[0].root_cause_component == "payments-db"
    collectors = {e.collected_by for e in report.evidence}
    assert {"observability", "dependency"} <= collectors
    assert report.stats.tool_calls >= 3 and report.stats.llm_calls > 0
    assert report.stats.input_tokens > 0
    # history tool is unavailable in this scenario -> recorded as a data gap, run continues
    assert any("search_incident_history" in g for g in report.data_gaps)
    assert (tmp_path / "report.json").exists() and (tmp_path / "report.md").exists()
    assert (tmp_path / "trace.jsonl").exists()


async def test_critic_feedback_loop_triggers_second_round(tmp_path):
    llm = make_llm(critic=[MORE, ACCEPT])
    report = await run(llm, tmp_path=tmp_path)
    assert report.stats.rounds == 2
    orch_calls = [c for c in llm.calls if c["role"] == "orchestrator"]
    assert len(orch_calls) == 2
    assert "Did DB saturation precede the deploy?" in str(orch_calls[1]["messages"][-1].content)
    assert report.outcome == "ROOT_CAUSE_IDENTIFIED"


async def test_max_rounds_bound(tmp_path):
    llm = make_llm(critic=lambda m, s: MORE)
    report = await run(llm, tmp_path=tmp_path, max_rounds=3)
    assert report.stats.rounds == 3
    assert report.outcome in ("PROBABLE", "INCONCLUSIVE")


async def test_no_progress_stops_loop(tmp_path):
    # specialists repeat identical (memoised) calls every round -> no new evidence -> stop after round 2
    llm = make_llm(critic=lambda m, s: MORE, obs=specialist_steps(lambda r: obs_calls(1)),
                   dep=specialist_steps(lambda r: []))
    report = await run(llm, tmp_path=tmp_path, max_rounds=3)
    assert report.stats.rounds == 2


async def test_inconclusive_verdict(tmp_path):
    llm = make_llm(critic=[CriticVerdict(decision="INCONCLUSIVE", issues=["critical metrics missing"])])
    report = await run(llm, tmp_path=tmp_path)
    assert report.outcome == "INCONCLUSIVE"
    assert "critical metrics missing" in report.critic_issues


async def test_reasoning_retries_on_invalid_citations(tmp_path):
    attempts = {"n": 0}

    def flaky_reasoning(messages, schema):
        attempts["n"] += 1
        hs = hypothesis_set(messages, schema)
        if attempts["n"] == 1:
            hs.hypotheses[0].supporting_evidence_ids = ["E-999"]
        return hs

    report = await run(make_llm(reasoning=flaky_reasoning), tmp_path=tmp_path)
    assert attempts["n"] == 2
    assert "E-999" not in report.ranked_hypotheses[0].supporting_evidence_ids


async def test_specialist_failure_degrades_gracefully(tmp_path):
    def boom(messages, tools):
        raise RuntimeError("LLM provider down")

    report = await run(make_llm(dep=boom), tmp_path=tmp_path)
    assert any("dependency" in n for n in report.stats.degraded_nodes)
    assert report.outcome in ("ROOT_CAUSE_IDENTIFIED", "PROBABLE", "INCONCLUSIVE")
    assert any("dependency" in g for g in report.data_gaps)


async def test_reporter_failure_falls_back_to_deterministic_report(tmp_path):
    def broken(messages, schema):
        raise StructuredOutputError("bad json")

    report = await run(make_llm(reporter=broken), tmp_path=tmp_path)
    assert report.summary  # deterministic fallback summary
    assert report.ranked_hypotheses


async def test_specialists_run_in_parallel(tmp_path):
    base = make_llm()
    slow = SlowLLM(structured=base._structured, tool_steps=base._tool_steps)
    t = time.perf_counter()
    await run(slow, tmp_path=tmp_path)
    elapsed = time.perf_counter() - t
    # 2 specialists x 2 tool steps x 0.25s = 1.0s sequential; parallel ≈ 0.5s
    assert elapsed < 0.85, elapsed


async def test_single_agent_baseline(tmp_path):
    report = await run(baseline_llm(), system="single", tmp_path=tmp_path)
    assert report.system == "single"
    # fairness: the baseline's own 'conclusive' flag maps to ACCEPT and the same thresholds apply
    assert report.outcome == "ROOT_CAUSE_IDENTIFIED"
    assert report.stats.tool_calls >= 3
    assert report.ranked_hypotheses[0].root_cause_component == "payments-db"
