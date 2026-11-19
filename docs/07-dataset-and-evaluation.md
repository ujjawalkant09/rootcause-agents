# 07 – Synthetic Dataset & Evaluation Plan

## 1. Test incidents (S01–S10 implemented; S11–S12 on the roadmap)

> S09 is the *expected-inconclusive* case. S11 and S12 are roadmap scenarios.

| ID | Scenario | True root cause | Adversarial traits | Expected outcome |
|----|----------|-----------------|--------------------|------------------|
| S01 | Bad deploy introduces a NullPointer in `checkout` | deploy in `checkout` | None (control/easy) | IDENTIFIED |
| S02 | Feature-flag config change raises the `search` timeout to 30s, causing thread starvation | config change | Deploy at the same time on a different service (**misleading**) | IDENTIFIED |
| S03 | `payments-db` connection-pool exhaustion | resource exhaustion (`payments-db`) | Cosmetic `checkout` deploy 3 min after onset (**misleading coincident deploy**), and a metrics timeout ×2 (**tool failure**) | IDENTIFIED |
| S04 | Upstream `auth` latency causes cascading 5xx in gateway | dependency failure (`auth`) | Symptoms reported only on `api-gateway`, so the agents must traverse the graph | IDENTIFIED |
| S05 | Traffic surge (marketing campaign) saturates `inventory` | traffic surge | A historical incident with the same symptoms had a bad deploy as its cause (**misleading history prior**) | IDENTIFIED |
| S06 | Cache eviction storm after a `cache` memory limit config change | config change (`cache`) | Log noise: chronic `notifications` errors (**noisy**) | IDENTIFIED |
| S07 | Expired TLS certificate on `payments` → external provider | infra / cert | Metrics show CPU spike unrelated (**conflicting observations**) | IDENTIFIED |
| S08 | Disk full on `queue` broker, causing async backlog | resource exhaustion (`queue`) | Logs tool returns **partial** shards, and the metrics are **stale** | PROBABLE or IDENTIFIED |
| S09 | Root cause in `user-svc`, but its logs and metrics tools are **unavailable** | dependency (`user-svc`) | Critical data missing (**tool failure**) | **INCONCLUSIVE** (or PROBABLE with low confidence) |
| S10 | Two simultaneous changes (deploy + config), only one of which is causal | config change | **Conflicting** signals, so temporal ordering decides | IDENTIFIED |
| S11 | Rollback of a deploy *did not* fix the issue; the real cause is a schema migration | data / migration | Deploy-then-rollback strongly suggests the deploy (**misleading**) | IDENTIFIED |
| S12 | Intermittent errors with no clear signal; telemetry is consistent with normal noise | unknown | **Insufficient evidence by design** | **INCONCLUSIVE** |

Coverage: 4 misleading-evidence, 3 conflicting-observation, 4 tool-failure, 2 expected-inconclusive and 1 easy control scenario (several scenarios have more than one trait).

## 2. Systems under test

| System | Description |
|---|---|
| **MAS** | Full multi-agent pipeline (docs 03–06) |
| **SAB** (single-agent baseline) | One ReAct agent, the same 6 tools, model, fault seeds, budget and output schema. Its system prompt covers the same investigation guidance (including "consider alternatives" and "inconclusive allowed") so the comparison is fair |
| **MAS-cheap** (confirmed ablation) | The same MAS, but Observability and Dependency use `gpt-6-luna`. Orchestrator, Reasoning, Critic and Reporter stay on `gpt-6.2-sol` |
| *(Stretch, if time allows)* | MAS without the Critic, to show the Critic's value |

## 3. Metrics

| Metric | Definition | How measured |
|---|---|---|
| **Root-cause accuracy (top-1)** | Top hypothesis `root_cause_component` and `category` match the ground truth (with aliases) | Deterministic match. Falls back to an LLM judge on mismatch-but-maybe-synonym, with the judge's rationale logged |
| Top-3 accuracy | Ground truth appears in the top 3 hypotheses | Same |
| **Outcome correctness** | The `outcome` matches the expected one (especially `INCONCLUSIVE` on S09/S12). A confident wrong answer is penalised | Deterministic |
| **Evidence grounding** | (a) **Citation validity**: % of cited evidence IDs that exist. (b) **Claim coverage**: % of non-assumption claims that cite ≥1 evidence ID. (c) **Key-evidence recall**: % of the scenario's `key_evidence` signals that the cited evidence covers (matched on service + signal type). *(An LLM-judge "citation support" score is on the roadmap)* | Code |
| Calibration | Brier score of top-1 confidence vs correctness | Code |
| **Tool-call count** | Total tool calls, retries and failed calls | From trace |
| **Latency** | Wall-clock per investigation (p50/p95) | From trace |
| **Token consumption** | Input, output and cache-read tokens per run, plus `$` cost from published pricing | From `response.usage` |
| Efficiency | Cost per correct diagnosis | Derived |

## 4. Protocol
- In the MVP, each scenario runs **once per system**: 10 scenarios × 3 systems (MAS, SAB, MAS-cheap) = **30 runs**. More repeats are listed as future work (`--repeats N` flag supported).
- All systems use the same fault seeds and budget.
- Results go to `eval/results/runs.jsonl`, `summary.csv` and `summary.md`.
- **Estimated cost**: measured on a 2-scenario pilot first and reported to you before the full run. With `gpt-6.2-sol` at $2/$10 per 1M tokens and pre-aggregated tools, a full run should cost only a few dollars, but this will be confirmed by the pilot.

## 5. Results table template (`eval/results/summary.md`)

| Metric | MAS | MAS-cheap | SAB |
|---|---|---|---|
| Top-1 accuracy | | | |
| Top-3 accuracy | | | |
| Outcome correctness | | | |
| Citation validity | | | |
| Citation support (judge) | | | |
| Key-evidence recall | | | |
| Brier score ↓ | | | |
| Tool calls (mean) | | | |
| Latency p50 / p95 (s) | | | |
| Tokens (mean in / out) | | | |
| Cost per run / per correct ($) | | | |

The tables will be followed by a breakdown by adversarial trait (misleading / conflicting / tool-failure / inconclusive).

## 6. Automated tests (`pytest`, offline, no API key needed)

| Test group | Examples |
|---|---|
| Tools | Schema validation; log clustering correctness; change-point detection on synthetic series; pagination |
| Fault injection | Timeout → retry → success; unavailable → gap recorded; partial completeness propagates to Evidence |
| Graph / routing | Parallel `Send` branches actually run concurrently (timing test); budget exhaustion stops gracefully; no-progress termination; max rounds respected; `recursion_limit` never hit in normal runs |
| State | Reducers merge parallel evidence; citation validator rejects unknown IDs |
| Agents (fake chat model) | Scripted LLM responses drive the full pipeline end-to-end, including a NEED_MORE_EVIDENCE loop and an INCONCLUSIVE path |
| Generator | Scenario YAML → telemetry is deterministic for a fixed seed; injected signals are detectable by the tools |
| **No-hardcoding guard** | Greps `src/rootcause_agents/agents` and `prompts/` for scenario IDs and ground-truth strings, and asserts none are present |
| Report | Every claim either cites existing evidence or is an ASSUMPTION |

An optional `-m live` marker runs 1–2 real scenarios against the API as a smoke test.
