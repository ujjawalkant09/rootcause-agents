# 08 – Project Structure, Setup & Roadmap

## 1. Repository layout (as built)

```
rootcause-agents/
├── README.md                     # setup, usage, results, limitations, external deps
├── CLAUDE.md                     # rules and architecture map for AI-assisted development
├── pyproject.toml  uv.lock       # uv-managed
├── Makefile                      # setup | app | test | lint | live
├── start-ui.command              # double-click to open the web UI (macOS)
├── docker-compose.yml            # local Postgres for saved sessions
├── .env.example                  # OpenAI, models, fallbacks, Postgres, Langfuse
├── .streamlit/config.toml        # UI theme
├── docs/                         # design docs, TECHNICAL_NOTE.md, EVALUATION_RESULTS.md, diagrams/
├── src/rootcause_agents/
│   ├── config.py  schemas.py  budget.py  validation.py  context.py  tracing.py  report.py
│   ├── llm.py                    # OpenAILLM (fallback chain, circuit breaker, prompt caching), ScriptedLLM (tests)
│   ├── graph.py                  # LangGraph StateGraph: nodes, Send fan-out, critic loop, human_review interrupt
│   ├── runner.py                 # start_session / resume_session / investigate
│   ├── persistence.py            # Postgres checkpointer + session registry
│   ├── telemetry.py              # Langfuse tracing (optional)
│   ├── evaluation.py             # grading + comparison matrix
│   ├── agents/                   # orchestrator, specialists, reasoning, critic, reporter, single_agent, tool_loop
│   ├── tools/                    # impl (6 tools), executor, faults, analytics
│   ├── sim/                      # scenario models, generator, store
│   ├── prompts/*.md              # system prompts (generic)
│   └── ui/                       # Streamlit app: app.py, app_pages/, common.py, jobs.py, view.py
├── data/                         # topology.yaml, history/incidents.yaml, scenarios/S01…S10.yaml
├── eval/results/                 # evaluation summary (summary.md, summary.csv)
├── tests/                        # offline pytest suite (scripted LLM)
└── runs/                         # per-run traces + reports (local)
```

## 2. Dependencies
`langgraph`, `langgraph-checkpoint-postgres`, `langchain-openai`, `langchain-core`, `pydantic`, `pyyaml`, `numpy`, `python-dotenv`, `streamlit`, `pandas`, `langfuse`, `psycopg`, plus `pytest`, `pytest-asyncio` and `ruff` for development.
External: the **OpenAI API** (needs `OPENAI_API_KEY` and internet). Optional: Docker (Postgres) and Langfuse Cloud. Tests run fully offline.

## 3. Setup

The web UI is the only interface. Install uv, copy `.env.example` to `.env` and add the OpenAI key, optionally `docker compose up -d`, then double-click `start-ui.command` (or `make app`). See the root README.

## 4. Build plan (MVP)

| Block | Work | Time |
|---|---|---|
| 1 | Scaffold: uv project, schemas, config, LLM factory, tracing | ~1 h |
| 2 | Simulator: topology, generator, 6 tools, analytics, fault injection, tool tests | ~2.5 h |
| 3 | 10 scenario YAMLs + ~15 historical incidents | ~1.5 h |
| 4 | LangGraph MAS: 6 agents, Send fan-out, Critic loop, budgets, recovery; demo on S01 and S03 | ~3 h |
| 5 | Single-agent baseline | ~0.5 h |
| 6 | Grader + eval runner, 2-scenario pilot (cost check), then the full 30 runs | ~1.5 h |
| 7 | README, final architecture doc, technical note, limitations | ~1 h |
| | **Total** | **~11 h** |

Build order is a vertical slice first: one tool → one agent → full graph on S01 → then widen. If time runs short, the cuts in order are: S11/S12, the "no-Critic" ablation, then extra tests. The core features below are never cut.

## 5. Feature checklist

- [ ] Runnable implementation with setup instructions
- [ ] Architecture diagram + coordination explanation (`docs/03-architecture.md`)
- [ ] Synthetic incident dataset (≥10) + automated tests
- [ ] Evaluation results: multi-agent vs single-agent (+ cheaper-worker variant)
- [ ] Technical note ≤500 words (`docs/TECHNICAL_NOTE.md`)
- [ ] No hardcoded answers (guard test)
- [ ] Incomplete functionality + external dependencies documented
