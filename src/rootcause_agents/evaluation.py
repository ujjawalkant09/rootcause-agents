"""Grading and the MAS vs MAS-cheap vs single-agent evaluation matrix.

Ground truth is read here (and in the simulator) only – never by agents.
"""

from __future__ import annotations

import asyncio
import csv
import json
import traceback
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from rootcause_agents.config import Settings
from rootcause_agents.llm import LLM
from rootcause_agents.runner import investigate
from rootcause_agents.schemas import Evidence, InvestigationReport
from rootcause_agents.sim.scenario import KeySignal, Scenario, load_scenario


def _norm(s: str) -> str:
    return s.strip().lower().replace("_", "-").replace(" ", "-")


def component_matches(pred: str, sc: Scenario) -> bool:
    gt = sc.ground_truth
    accepted = {_norm(gt.root_cause_component), *(_norm(a) for a in gt.aliases)}
    return _norm(pred) in accepted


def category_matches(pred: str, sc: Scenario) -> bool:
    gt = sc.ground_truth
    return pred == gt.root_cause_category or pred in gt.alt_categories


def _cited_ids(r: InvestigationReport) -> list[str]:
    ids: list[str] = []
    for h in r.ranked_hypotheses:
        ids += h.supporting_evidence_ids + h.contradicting_evidence_ids
        ids += [e for c in h.mechanism for e in c.evidence_ids]
    for c in r.facts + r.inferences + r.assumptions:
        ids += c.evidence_ids
    for a in r.recommended_actions:
        ids += a.evidence_ids
    return ids


def _signal_covered(ks: KeySignal, evidence: list[Evidence]) -> bool:
    for e in evidence:
        svc = e.params.get("service")
        if ks.kind == "metric" and e.source_tool == "query_metrics" and svc == ks.service \
                and e.params.get("metric") == ks.metric:
            return True
        if ks.kind == "log" and e.source_tool == "query_logs" and svc == ks.service:
            return True
        if ks.kind == "deployment" and e.source_tool == "get_deployments" and any(
                d.get("service") == ks.service for d in (e.data or {}).get("deployments", [])):
            return True
        if ks.kind == "config" and e.source_tool == "get_config_changes" and any(
                c.get("service") == ks.service for c in (e.data or {}).get("config_changes", [])):
            return True
        if ks.kind == "dependency" and e.source_tool == "get_service_dependencies" and svc == ks.service:
            return True
    return False


def grade(r: InvestigationReport, sc: Scenario) -> dict[str, Any]:
    hyps = r.ranked_hypotheses
    top = hyps[0] if hyps else None
    top1 = bool(top and component_matches(top.root_cause_component, sc)
                and category_matches(top.root_cause_category, sc))
    top3 = any(component_matches(h.root_cause_component, sc) for h in hyps[:3])
    ev_by_id = {e.id: e for e in r.evidence}
    cited = list(dict.fromkeys(_cited_ids(r)))  # unique IDs
    valid = [i for i in cited if i in ev_by_id]
    citation_validity = len(valid) / len(cited) if cited else 0.0
    claims = [c for h in hyps for c in h.mechanism] + r.facts + r.inferences
    claim_coverage = sum(1 for c in claims if c.evidence_ids) / len(claims) if claims else 0.0
    cited_evidence = [ev_by_id[i] for i in dict.fromkeys(valid)]
    ks = sc.ground_truth.key_signals
    recall = sum(_signal_covered(k, cited_evidence) for k in ks) / len(ks) if ks else 1.0
    conf = top.confidence if top else 0.0
    s = r.stats
    return {
        "scenario_id": sc.id, "system": r.system, "traits": sc.traits, "outcome": r.outcome,
        "expected_outcomes": sc.ground_truth.expected_outcomes,
        "predicted_component": top.root_cause_component if top else None,
        "predicted_category": top.root_cause_category if top else None,
        "truth_component": sc.ground_truth.root_cause_component, "truth_category": sc.ground_truth.root_cause_category,
        "top1_correct": top1, "top3_correct": top3,
        "outcome_correct": r.outcome in sc.ground_truth.expected_outcomes,
        "confident_wrong": r.outcome == "ROOT_CAUSE_IDENTIFIED" and not top1,
        "citation_validity": round(citation_validity, 4), "claim_coverage": round(claim_coverage, 4),
        "key_evidence_recall": round(recall, 4), "top_confidence": conf, "brier": round((conf - float(top1)) ** 2, 6),
        "rounds": s.rounds, "tool_calls": s.tool_calls, "tool_errors": s.tool_errors, "llm_calls": s.llm_calls,
        "input_tokens": s.input_tokens, "output_tokens": s.output_tokens, "cost_usd": s.cost_usd,
        "latency_s": s.latency_s, "degraded_nodes": len(s.degraded_nodes), "error": None,
        "cached_input_tokens": s.cached_input_tokens,
        "cache_hit_rate": round(s.cached_input_tokens / s.input_tokens, 4) if s.input_tokens else 0.0,
        "llm_fallbacks": s.llm_fallbacks,
        "tokens_by_role": {u.role: u.input_tokens + u.output_tokens for u in s.by_role},
    }


METRICS = [
    ("top1_accuracy", "top1_correct", "Top-1 accuracy"),
    ("top3_accuracy", "top3_correct", "Top-3 accuracy"),
    ("outcome_correctness", "outcome_correct", "Outcome correctness"),
    ("confident_wrong_rate", "confident_wrong", "Confident-wrong rate ↓"),
    ("citation_validity", "citation_validity", "Citation validity"),
    ("claim_coverage", "claim_coverage", "Claim coverage (cited)"),
    ("key_evidence_recall", "key_evidence_recall", "Key-evidence recall"),
    ("brier", "brier", "Brier score ↓"),
    ("rounds", "rounds", "Rounds (mean)"),
    ("tool_calls", "tool_calls", "Tool calls (mean)"),
    ("llm_calls", "llm_calls", "LLM calls (mean)"),
    ("input_tokens", "input_tokens", "Input tokens (mean)"),
    ("output_tokens", "output_tokens", "Output tokens (mean)"),
    ("cost_usd", "cost_usd", "Cost per run ($)"),
    ("latency_s", "latency_s", "Latency (s, mean)"),
    ("cache_hit_rate", "cache_hit_rate", "Cache hit rate (cached/input)"),
    ("llm_fallbacks", "llm_fallbacks", "Model fallbacks (mean)"),
]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_sys: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if not r.get("skipped"):
            by_sys[r["system"]].append(r)
    out: dict[str, dict[str, float]] = {}
    for system, rs in by_sys.items():
        ok = [r for r in rs if r.get("error") is None]
        agg: dict[str, float] = {"n": len(rs), "errors": len(rs) - len(ok)}
        for key, field, _ in METRICS:
            agg[key] = round(_mean([float(r[field]) for r in ok]), 4)
        correct = sum(1 for r in ok if r["top1_correct"])
        agg["cost_per_correct_usd"] = round(sum(r["cost_usd"] for r in ok) / correct, 4) if correct else float("nan")
        out[system] = agg
    return out


def render_summary(rows: list[dict[str, Any]]) -> str:
    agg = summarize(rows)
    systems = list(agg)
    lines = ["# Evaluation results\n", f"Runs: {len(rows)} · systems: {', '.join(systems)}\n",
             "## Aggregate\n", "| Metric | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for key, _, label in [*METRICS, ("cost_per_correct_usd", None, "Cost per correct diagnosis ($)"), ("errors", None, "Crashed runs")]:
        lines.append(f"| {label} | " + " | ".join(f"{agg[s][key]:.4g}" for s in systems) + " |")
    # by adversarial trait
    # where do the tokens go? (mean tokens per run, by agent role)
    roles = sorted({role for r in rows if r.get("error") is None for role in r.get("tokens_by_role", {})})
    lines += ["\n## Tokens by agent (mean per run)\n", "| Agent | " + " | ".join(systems) + " |",
              "|---|" + "---|" * len(systems)]
    for role in roles:
        cells = []
        for s in systems:
            rs = [r for r in rows if r["system"] == s and r.get("error") is None]
            cells.append(f"{_mean([float(r.get('tokens_by_role', {}).get(role, 0)) for r in rs]):.0f}")
        lines.append(f"| {role} | " + " | ".join(cells) + " |")
    lines += ["\n## Top-1 accuracy by scenario trait\n", "| Trait | " + " | ".join(systems) + " |",
              "|---|" + "---|" * len(systems)]
    traits = sorted({t for r in rows for t in r.get("traits", [])})
    for t in traits:
        cells = []
        for s in systems:
            rs = [r for r in rows if r["system"] == s and t in r.get("traits", []) and r.get("error") is None]
            cells.append(f"{_mean([float(r['top1_correct']) for r in rs]):.2f} (n={len(rs)})")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines += ["\n## Per-scenario results\n",
              "| Scenario | System | Outcome | Predicted | Truth | Top-1 | Outcome ok | KE recall | Tools | Tokens | $ | s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["scenario_id"], r["system"])):
        if r.get("error"):
            lines.append(f"| {r['scenario_id']} | {r['system']} | ERROR | {r['error'][:60]} | | | | | | | | |")
            continue
        lines.append(
            f"| {r['scenario_id']} | {r['system']} | {r['outcome']} | {r['predicted_component']}/"
            f"{r['predicted_category']} | {r['truth_component']}/{r['truth_category']} | "
            f"{'✅' if r['top1_correct'] else '❌'} | {'✅' if r['outcome_correct'] else '❌'} | "
            f"{r['key_evidence_recall']:.2f} | {r['tool_calls']} | {r['input_tokens'] + r['output_tokens']} | "
            f"{r['cost_usd']:.4f} | {r['latency_s']:.1f} |")
    return "\n".join(lines) + "\n"


async def run_matrix(
    systems: list[str],
    scenario_paths: list[Path],
    settings: Settings,
    *,
    repeats: int = 1,
    llm_factory: Callable[[str], LLM] | None = None,
    out_dir: Path,
    topology_path: Path | None = None,
    history_path: Path | str | None = "default",
    concurrency: int = 2,
    sleep_scale: float = 1.0,
    progress: Callable[[str], None] | None = None,
    max_cost_usd: float | None = None,
) -> list[dict[str, Any]]:
    """Run every (scenario, system, repeat). ``max_cost_usd`` is a hard spend cap: once the summed cost of
    finished runs reaches it, runs not yet started are recorded as skipped instead of calling the API."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    rows: list[dict[str, Any]] = []
    runs_file = out_dir / "runs.jsonl"
    runs_file.write_text("")
    spent = 0.0
    session_id = f"eval-{datetime.now():%Y%m%d-%H%M%S}"  # one Langfuse session per evaluation batch

    async def one(path: Path, system: str, rep: int) -> None:
        sc = load_scenario(path)
        nonlocal spent
        async with sem:
            if max_cost_usd is not None and spent >= max_cost_usd:
                row = {"scenario_id": sc.id, "system": system, "repeat": rep,
                       "skipped": f"cost cap ${max_cost_usd:.2f} reached"}
                rows.append(row)
                if progress:
                    progress(f"{sc.id} {system} r{rep}: skipped ({row['skipped']})")
                return
            try:
                report = await investigate(
                    path, system, settings=settings, llm=llm_factory(system) if llm_factory else None,
                    topology_path=topology_path, history_path=history_path,
                    out_dir=out_dir / "runs" / f"{sc.id}-{system}-r{rep}", sleep_scale=sleep_scale,
                    session_id=session_id)
                row = grade(report, sc)
            except Exception as e:  # a crashed run is recorded, not fatal for the matrix
                row = {"scenario_id": sc.id, "system": system, "traits": sc.traits,
                       "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
            row["repeat"] = rep
            spent += float(row.get("cost_usd") or 0.0)
            rows.append(row)
            with runs_file.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            if progress:
                status = row.get("error") or f"{row['outcome']} top1={'Y' if row['top1_correct'] else 'N'}"
                progress(f"{sc.id} {system} r{rep}: {status}")

    await asyncio.gather(*(one(p, s, k) for p in scenario_paths for s in systems for k in range(repeats)))
    rows.sort(key=lambda r: (r["scenario_id"], r["system"], r["repeat"]))
    graded = [r for r in rows if not r.get("skipped")]
    skipped = len(rows) - len(graded)
    note = f"\n> {skipped} run(s) skipped: spend cap ${max_cost_usd:.2f} reached.\n" if skipped else ""
    (out_dir / "summary.md").write_text(render_summary(graded) + note)
    ok = [r for r in graded if r.get("error") is None]
    if ok:
        fields = [k for k in ok[0] if k not in ("traits", "expected_outcomes", "tokens_by_role")]
        with (out_dir / "summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(ok)
    return rows
