from rootcause_agents.budget import Budget, BudgetTracker, decide_outcome, should_continue
from rootcause_agents.llm import Usage
from rootcause_agents.schemas import CriticVerdict, Hypothesis


def H(conf, cat="resource_exhaustion"):
    return Hypothesis(id="H", title="t", root_cause_component="c", root_cause_category=cat, mechanism=[],
                      supporting_evidence_ids=["E-1"], contradicting_evidence_ids=[], unexplained_symptoms=[],
                      confidence=conf)


def V(d):
    return CriticVerdict(decision=d, issues=[])


def test_outcome_identified_requires_confidence_and_margin():
    assert decide_outcome(V("ACCEPT"), [H(0.8), H(0.1)], limits_hit=False) == "ROOT_CAUSE_IDENTIFIED"
    assert decide_outcome(V("ACCEPT"), [H(0.8), H(0.7)], limits_hit=False) == "PROBABLE"
    assert decide_outcome(V("ACCEPT"), [H(0.6)], limits_hit=False) == "PROBABLE"


def test_outcome_inconclusive_cases():
    assert decide_outcome(V("INCONCLUSIVE"), [H(0.9), H(0.05)], limits_hit=False) == "INCONCLUSIVE"
    assert decide_outcome(V("NEED_MORE_EVIDENCE"), [H(0.4), H(0.3)], limits_hit=True) == "INCONCLUSIVE"
    assert decide_outcome(V("NEED_MORE_EVIDENCE"), [H(0.6), H(0.2)], limits_hit=True) == "PROBABLE"
    assert decide_outcome(V("ACCEPT"), [H(0.9, cat="unknown")], limits_hit=False) == "INCONCLUSIVE"
    assert decide_outcome(V("ACCEPT"), [], limits_hit=False) == "INCONCLUSIVE"


def test_should_continue():
    b = BudgetTracker(Budget(max_rounds=3))
    assert should_continue(V("NEED_MORE_EVIDENCE"), round_=1, budget=b, new_evidence=3) is True
    assert should_continue(V("NEED_MORE_EVIDENCE"), round_=3, budget=b, new_evidence=3) is False
    assert should_continue(V("NEED_MORE_EVIDENCE"), round_=1, budget=b, new_evidence=0) is False
    assert should_continue(V("ACCEPT"), round_=1, budget=b, new_evidence=3) is False


def test_budget_tokens_and_time():
    b = BudgetTracker(Budget(max_total_tokens=100))
    b.add_usage("critic", Usage(60, 50, 0), cost=0.01)
    assert b.exhausted() and "tokens" in b.exhausted_reason()
    b2 = BudgetTracker(Budget(max_wall_clock_s=0))
    assert b2.exhausted()


def test_usage_tracked_per_role():
    b = BudgetTracker(Budget())
    b.add_usage("critic", Usage(1000, 100, 800), cost=0.002)
    b.add_usage("critic", Usage(500, 50, 0), cost=0.001)
    b.add_usage("observability", Usage(200, 20, 100), cost=0.0005)
    assert b.cached_tokens == 900 and b.input_tokens == 1700 and b.llm_calls == 3
    critic = b.by_role["critic"]
    assert (critic.calls, critic.input_tokens, critic.output_tokens, critic.cached_tokens) == (2, 1500, 150, 800)
    assert abs(critic.cost_usd - 0.003) < 1e-12
