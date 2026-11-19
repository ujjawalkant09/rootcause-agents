# Evaluation Results: Multi-Agent vs Single-Agent Baseline

**Run:** 2026-11-19 · 10 incidents × 3 systems × 1 run = 30 runs · **total cost $2.02** · 0 crashed runs
**Models:** main `gpt-6.1-terra`, cheap worker `gpt-6-luna` · max 2 rounds · same tools, fault injection, budget, report schema and outcome policy for every system
**Raw data:** [`eval/results/summary.md`](../eval/results/summary.md) (full tables), [`summary.csv`](../eval/results/summary.csv), and per-run reports and traces (local `eval/results/full/runs/`, also in Langfuse as one session)

Systems:
- `mas`: 6-agent LangGraph system, all agents on the main model
- `mas-cheap`: same graph, with the Observability and Dependency workers on the cheap model
- `single`: one tool-calling agent with the same 6 tools

## 1. Headline comparison

| Metric | mas | mas-cheap | single |
|---|---|---|---|
| **Top-1 root-cause accuracy** (component + category) | **10/10** | 9/10 | **10/10** |
| Top-3 accuracy | 10/10 | 10/10 | 10/10 |
| **Outcome correctness** (outcome is in the expected set) | **10/10** | 8/10 | 8/10 |
| Confident-wrong rate ↓ | 0 | 0 | 0 |
| Citation validity (cited IDs exist) | 1.00 | 1.00 | 1.00 |
| Claim coverage (claims with citations) | 0.89 | 0.88 | 1.00 |
| Key-evidence recall | 1.00 | 1.00 | 1.00 |
| Brier score ↓ (top confidence vs correctness) | 0.164 | 0.237 | **0.056** |
| Rounds (mean) | 1.8 | 1.9 | 1.0 |
| **Tool calls** (mean) | 34.7 | 35.1 | 39.3 |
| LLM calls (mean) | 22.6 | 24.1 | 3.5 |
| **Tokens** in / out (mean) | 59.1k / 11.3k | 59.8k / 15.2k | 25.2k / 3.4k |
| **Latency** (mean) | 147 s | 159 s | **51 s** |
| **Cost per run** | $0.109 | $0.057 | **$0.036** |
| Cost per correct diagnosis | $0.109 | $0.063 | **$0.036** |

## 2. What the numbers say

1. **On root-cause accuracy, the single agent is as good as the multi-agent system on this dataset** (both 10/10 top-1), at **about 3× lower cost and latency**. The main reason is that the tools pre-aggregate the telemetry (log clusters, metric change points), so one capable agent sees the signal quickly. The 10 scenarios are near the ceiling for both systems, so they do not separate them on accuracy.
2. **Where the multi-agent system wins: outcome calibration on the hardest cases.** It reached an acceptable outcome on all 10 incidents. The single agent got the cause right but declared INCONCLUSIVE on **S02** (misleading) and **S03** (misleading + tool failure). The critic loop gathered the extra evidence (round 2) needed to commit. That is the only behavioural difference, and with n=10 and 1 run each it is **not statistically significant**.
3. **Grounding is enforced by code for every system:** 100% valid citations and 100% key-evidence recall. The single agent's higher claim coverage (1.00 vs 0.89) reflects that the multi-agent mechanism chains contain more explicitly labelled *assumptions* (uncited by design), not invalid citations.
4. **Calibration:** the multi-agent system is more hedged (lower confidence on correct answers), so its Brier score is worse. Confidence comes from the LLM and is uncalibrated.
5. **Cost of collaboration:** 6.5× more LLM calls and about 2.6× more tokens. The orchestrator, critic and reporter overhead is fixed per round. The per-task tool quota makes the multi-agent system use *fewer* tool calls (34.7 vs 39.3).
6. **mas-cheap** cuts cost by 48% versus `mas`, at one miss: on **S07** (conflicting signals: a TLS certificate expiry) it named the right service but ended with category `unknown` / INCONCLUSIVE. It also missed the outcome on S03. Cheap workers gather noisier evidence on conflicting scenarios (conflicting-trait accuracy 0.67 vs 1.00).

### Top-1 accuracy by scenario trait

| Trait | n | mas | mas-cheap | single |
|---|---|---|---|---|
| misleading | 5 | 1.00 | 1.00 | 1.00 |
| conflicting | 3 | 1.00 | 0.67 | 1.00 |
| tool_failure | 3 | 1.00 | 1.00 | 1.00 |
| control / noisy / inconclusive | 1 each | 1.00 | 1.00 | 1.00 |

## 3. Development pilot: what the first real run exposed

Before this evaluation, a 2-incident pilot (S01 + S03) on the more expensive model exposed two defects that offline tests could not show. Both were fixed test-first:

| Pilot | mas top-1 | mas outcome | Cause |
|---|---|---|---|
| Before the fixes | 0.5 | 0.0 | Specialists used the whole 40-call budget in round 1, so the critic could not loop; the critic also demanded proof the tools cannot provide (rollbacks, request traces) |
| After the fixes (per-task quota + calibrated critic + metric catalogue in the environment prompt) | **1.0** | 0.5–1.0 | — |

Switching from `gpt-6.2-sol` to `gpt-6.1-terra`, with max 2 rounds and output length limits, then cut cost per incident by about 47% at the same pilot accuracy.

## 4. Caveats (read before quoting the numbers)
- **Small sample:** 10 synthetic incidents, 1 run each. Differences of 1–2 incidents are within run-to-run noise. `--repeats N` is supported.
- **Lenient grading choices are declared in the dataset:** S05 accepts `api-gateway` as an alias for the root cause (the traffic surge enters through it), and S09 (designed to be inconclusive) accepts category `unknown`.
- **The synthetic telemetry is cleaner than production.** A harder, noisier dataset is needed to separate the two architectures on accuracy.
- Prices are list prices hard-coded in `config.py`; Langfuse computes its own costs independently.
- No model fallbacks occurred in this run (they did in the pilots, and recovered automatically).
