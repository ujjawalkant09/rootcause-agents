from rootcause_agents.schemas import Claim, Hypothesis, RecommendedAction, ReportDraft
from rootcause_agents.validation import sanitize_hypotheses, sanitize_report, validate_hypotheses


def H(sup, contra=(), mech_ids=(), conf=0.7, cat="resource_exhaustion", id="H1"):
    return Hypothesis(
        id=id, title="t", root_cause_component="payments-db", root_cause_category=cat,
        mechanism=[Claim(text="m", kind="inference", evidence_ids=list(mech_ids))],
        supporting_evidence_ids=list(sup), contradicting_evidence_ids=list(contra),
        unexplained_symptoms=[], confidence=conf,
    )


KNOWN = {"E-1", "E-2", "E-3"}


def test_valid_hypotheses_have_no_errors():
    assert validate_hypotheses([H(["E-1"], ["E-2"], ["E-3"]), H(["E-2"], id="H2")], KNOWN, min_count=2) == []


def test_unknown_ids_are_reported():
    errs = validate_hypotheses([H(["E-1", "E-99"]), H(["E-2"], ["E-77"], id="H2")], KNOWN, min_count=2)
    assert any("E-99" in e for e in errs) and any("E-77" in e for e in errs)


def test_hypothesis_without_support_is_reported():
    errs = validate_hypotheses([H([]), H(["E-1"], id="H2")], KNOWN, min_count=2)
    assert any("H1" in e and "supporting" in e for e in errs)


def test_min_count_enforced():
    errs = validate_hypotheses([H(["E-1"])], KNOWN, min_count=2)
    assert any("at least 2" in e for e in errs)


def test_unknown_category_may_lack_support():
    assert validate_hypotheses([H([], cat="unknown"), H(["E-1"], id="H2")], KNOWN, min_count=2) == []


def test_sanitize_strips_ids_and_penalises():
    out = sanitize_hypotheses([H(["E-99"], conf=0.8), H(["E-1", "E-50"], conf=0.6, id="H2")], KNOWN)
    by_id = {h.id: h for h in out}
    assert by_id["H1"].supporting_evidence_ids == [] and by_id["H1"].confidence <= 0.4
    assert by_id["H2"].supporting_evidence_ids == ["E-1"]
    assert out[0].confidence >= out[1].confidence  # re-ranked


def test_sanitize_report_reclassifies_uncited_facts():
    draft = ReportDraft(
        summary="s",
        facts=[Claim(text="cited", kind="fact", evidence_ids=["E-1"]),
               Claim(text="uncited", kind="fact", evidence_ids=["E-404"])],
        inferences=[], assumptions=[],
        recommended_actions=[RecommendedAction(action="a", type="fix", priority="P1", evidence_ids=["E-9"])],
    )
    clean = sanitize_report(draft, KNOWN)
    assert [c.text for c in clean.facts] == ["cited"]
    assert any(c.text == "uncited" and c.kind == "assumption" for c in clean.assumptions)
    assert clean.recommended_actions[0].evidence_ids == []
