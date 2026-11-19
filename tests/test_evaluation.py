import json

from typer.testing import CliRunner

from rootcause_agents.budget import Budget
from rootcause_agents.config import Settings
from rootcause_agents.evaluation import grade, run_matrix, summarize
from rootcause_agents.schemas import Claim, Evidence, Hypothesis, InvestigationReport, RunStats
from tests.conftest import FIXTURES
from tests.helpers import baseline_llm, make_llm


def ev(id, tool, params, data=None):
    return Evidence(id=id, source_tool=tool, params=params, status="ok", completeness=1.0, summary="s",
                    data=data or {}, collected_by="x", round=1)


def hyp(comp, cat, conf, sup):
    return Hypothesis(id="H", title="t", root_cause_component=comp, root_cause_category=cat,
                      mechanism=[Claim(text="m", kind="inference", evidence_ids=sup)], supporting_evidence_ids=sup,
                      contradicting_evidence_ids=[], unexplained_symptoms=[], confidence=conf)


def report(hyps, outcome="ROOT_CAUSE_IDENTIFIED", evidence=None, facts=None):
    return InvestigationReport(
        incident_id="I", system="mas", outcome=outcome, summary="s", ranked_hypotheses=hyps,
        facts=facts or [], inferences=[], assumptions=[Claim(text="a", kind="assumption")], data_gaps=[],
        recommended_actions=[], evidence=evidence or [],
        stats=RunStats(system="mas", tool_calls=5, input_tokens=1000, output_tokens=200, cost_usd=0.01, latency_s=3),
    )


EVIDENCE = [
    ev("E-1", "query_metrics", {"service": "payments-db", "metric": "db_connections"}),
    ev("E-2", "query_logs", {"service": "payments"}),
    ev("E-3", "get_deployments", {}, {"deployments": [{"service": "checkout"}]}),
]


def test_grade_correct_diagnosis(scenario):
    r = report([hyp("Payments-DB", "resource_exhaustion", 0.8, ["E-1", "E-2"]),
                hyp("checkout", "bad_deploy", 0.1, ["E-3"])], evidence=EVIDENCE)
    g = grade(r, scenario)
    assert g["top1_correct"] and g["top3_correct"] and g["outcome_correct"]
    assert g["key_evidence_recall"] == 1.0
    assert g["citation_validity"] == 1.0
    assert abs(g["brier"] - 0.04) < 1e-9
    assert g["confident_wrong"] is False


def test_grade_wrong_and_confident(scenario):
    r = report([hyp("checkout", "bad_deploy", 0.9, ["E-3"]), hyp("payments-db", "resource_exhaustion", 0.05,
                                                                   ["E-1"])], evidence=EVIDENCE)
    g = grade(r, scenario)
    assert not g["top1_correct"] and g["top3_correct"]
    assert g["confident_wrong"] is True
    assert g["key_evidence_recall"] == 0.5  # E-1 (metric) cited by H2; payments logs (E-2) not cited


def test_alias_matches_and_coverage(scenario):
    r = report([hyp("payments database", "resource_exhaustion", 0.7, ["E-1"])], outcome="PROBABLE",
               evidence=EVIDENCE, facts=[Claim(text="f", kind="fact", evidence_ids=["E-404"])])
    g = grade(r, scenario)
    assert g["top1_correct"]
    assert g["citation_validity"] == 0.5  # E-404 invalid
    assert g["key_evidence_recall"] == 0.5  # logs on payments not cited


async def test_run_matrix_and_summary(tmp_path):
    factories = {"mas": make_llm, "single": baseline_llm}
    settings = Settings(openai_api_key="sk-test", budget=Budget())
    rows = await run_matrix(
        systems=["mas", "single"], scenario_paths=[FIXTURES / "scenario_basic.yaml"], settings=settings,
        llm_factory=lambda system: factories[system](), out_dir=tmp_path, topology_path=FIXTURES / "topology.yaml",
        history_path=None, sleep_scale=0.0,
    )
    assert {r["system"] for r in rows} == {"mas", "single"}
    assert all(r["top1_correct"] for r in rows)
    lines = (tmp_path / "runs.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["scenario_id"] == "T01"
    md = (tmp_path / "summary.md").read_text()
    assert "Top-1 accuracy" in md and "mas" in md and "single" in md
    assert "Tokens by agent" in md and "Cache hit rate" in md
    assert all("cache_hit_rate" in r and "tokens_by_role" in r for r in rows)
    assert (tmp_path / "summary.csv").exists()
    agg = summarize(rows)
    assert agg["mas"]["n"] == 1 and agg["mas"]["top1_accuracy"] == 1.0


def test_cli_list_scenarios():
    from rootcause_agents.cli import app

    res = CliRunner().invoke(app, ["list"])
    assert res.exit_code == 0
    assert "S01" in res.output and "S10" in res.output


async def test_run_matrix_stops_at_cost_cap(tmp_path):
    """A hard spend cap: once the cap is reached, the remaining runs are skipped (recorded, not run)."""
    from rootcause_agents.llm import Usage

    def pricey():  # each LLM call reports 1M input tokens on a priced model -> clearly over a tiny cap
        llm = make_llm()
        llm.usage_per_call = Usage(1_000_000, 0, 0)
        llm.model = "gpt-6.1-terra"
        return llm

    settings = Settings(openai_api_key="sk-test", budget=Budget())
    paths = [FIXTURES / "scenario_basic.yaml"] * 3
    rows = await run_matrix(systems=["mas"], scenario_paths=paths, settings=settings, llm_factory=lambda s: pricey(),
                            out_dir=tmp_path, topology_path=FIXTURES / "topology.yaml", history_path=None,
                            sleep_scale=0.0, concurrency=1, max_cost_usd=0.5)
    ran = [r for r in rows if not r.get("skipped")]
    assert len(ran) == 1 and len(rows) == 3
    assert all(r["skipped"] == "cost cap $0.50 reached" for r in rows if r.get("skipped"))
    assert summarize(rows)["mas"]["n"] == 1  # skipped runs do not pollute the metrics
