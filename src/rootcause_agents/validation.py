"""Code-level grounding checks: claims must cite evidence that actually exists."""

from __future__ import annotations

from rootcause_agents.schemas import Claim, Hypothesis, ReportDraft

NO_SUPPORT_CONFIDENCE_CAP = 0.4


def validate_hypotheses(hypotheses: list[Hypothesis], known_ids: set[str], min_count: int = 2) -> list[str]:
    errors: list[str] = []
    if len(hypotheses) < min_count:
        errors.append(f"Provide at least {min_count} competing hypotheses (got {len(hypotheses)}).")
    for h in hypotheses:
        cited = list(h.supporting_evidence_ids) + list(h.contradicting_evidence_ids)
        cited += [eid for c in h.mechanism for eid in c.evidence_ids]
        unknown = sorted({e for e in cited if e not in known_ids})
        if unknown:
            errors.append(f"{h.id}: cites unknown evidence IDs {unknown}; only cite IDs from the evidence list.")
        if h.root_cause_category != "unknown" and not [e for e in h.supporting_evidence_ids if e in known_ids]:
            errors.append(f"{h.id}: has no valid supporting evidence IDs.")
    return errors


def _clean_claims(claims: list[Claim], known: set[str]) -> list[Claim]:
    return [c.model_copy(update={"evidence_ids": [e for e in c.evidence_ids if e in known]}) for c in claims]


def sanitize_hypotheses(hypotheses: list[Hypothesis], known_ids: set[str]) -> list[Hypothesis]:
    """Recovery path after retries: drop unknown IDs, cap confidence of unsupported hypotheses, re-rank."""
    out = []
    for h in hypotheses:
        sup = [e for e in h.supporting_evidence_ids if e in known_ids]
        con = [e for e in h.contradicting_evidence_ids if e in known_ids]
        conf = h.confidence if sup or h.root_cause_category == "unknown" else min(h.confidence * 0.5,
                                                                                  NO_SUPPORT_CONFIDENCE_CAP)
        out.append(h.model_copy(update={"supporting_evidence_ids": sup, "contradicting_evidence_ids": con,
                                        "mechanism": _clean_claims(h.mechanism, known_ids), "confidence": conf}))
    return sorted(out, key=lambda h: h.confidence, reverse=True)


def sanitize_report(draft: ReportDraft, known_ids: set[str]) -> ReportDraft:
    """Facts must cite existing evidence; uncited 'facts' are demoted to assumptions."""
    facts, demoted = [], []
    for c in _clean_claims(draft.facts, known_ids):
        (facts if c.evidence_ids else demoted).append(c if c.evidence_ids else c.model_copy(update={"kind": "assumption"}))
    inferences = [c.model_copy(update={"kind": "inference"}) for c in _clean_claims(draft.inferences, known_ids)]
    assumptions = [c.model_copy(update={"kind": "assumption"}) for c in _clean_claims(draft.assumptions, known_ids)]
    actions = [a.model_copy(update={"evidence_ids": [e for e in a.evidence_ids if e in known_ids]})
               for a in draft.recommended_actions]
    return draft.model_copy(update={"facts": facts, "inferences": inferences, "assumptions": assumptions + demoted,
                                    "recommended_actions": actions})
