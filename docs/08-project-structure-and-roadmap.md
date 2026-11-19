# 08 – Project Structure, Setup & Roadmap

## 1. Repository layout

> As built, the layout differs slightly (e.g. `tools/impl.py` + `tools/executor.py`, `evaluation.py` inside the package). See the root `README.md` → Project layout and `CLAUDE.md` → Architecture map.

```
rootcause-agents/
├── README.md                     # setup, usage, limitations, external deps
├── pyproject.toml                # uv-managed
├── Makefile                      # setup | test | demo | eval
├── .env.example                  # OPENAI_API_KEY=  MAIN_MODEL=gpt-6.2-sol  WORKER_MODEL=gpt-6-luna
├── docs/                         # these design docs + final TECHNICAL_NOTE.md
├── src/rootcause_agents/
│   ├── cli.py                    # `rootcause-agents investigate S03 --system mas|mas-cheap|single`, `rootcause-agents eval`
│   ├── config.py                 # models, budgets (from .env)
│   ├── schemas.py                # all Pydantic contracts
│   ├── llm.py                    # ChatOpenAI factory, usage tracking callback
│   ├── state.py                  # InvestigationState (TypedDict + reducers), citation validator, budget
│   ├── graph.py                  # LangGraph StateGraph: nodes, Send fan-out, critic router
│   ├── agents/
│   │   ├── tool_loop.py          # bounded tool-calling loop shared by specialists + baseline
│   │   ├── orchestrator.py  observability.py  dependency.py
│   │   ├── reasoning.py  critic.py  reporter.py
│   │   └── single_agent.py       # baseline (same tools, same budget)
│   ├── prompts/*.md              # system prompts
│   ├── tools/
│   │   ├── registry.py           # LangChain @tool wrappers with Pydantic args
│   │   ├── logs.py  metrics.py  deployments.py  config_changes.py  dependencies.py  history.py
│   │   ├── analytics.py          # log templating, change-point detection
│   │   └── faults.py             # FaultInjector + retry
│   ├── sim/generator.py          # scenario YAML → telemetry store
│   ├── report.py                 # JSON → Markdown
│   └── tracing.py                # JSONL events
├── data/
│   ├── topology.yaml
│   ├── history/incidents.yaml
│   └── scenarios/S01…S10.yaml
├── eval/
│   ├── run_eval.py               # MAS vs MAS-cheap vs single
│   ├── grader.py
│   └── results/                  # runs.jsonl, summary.md, summary.csv
├── tests/                        # offline pytest (fake chat model)
└── runs/                         # per-run traces + reports
```

## 2. Dependencies
`langgraph`, `langchain-openai`, `langchain-core`, `pydantic`, `typer`, `rich`, `pyyaml`, `numpy`, `python-dotenv`, plus `pytest` and `pytest-asyncio` for development.
External: the **OpenAI API** (needs `OPENAI_API_KEY` and internet). Tests run fully offline.

## 3. Setup (target UX)

```bash
make setup                     # uv sync
cp .env.example .env           # add OPENAI_API_KEY
make test                      # offline tests
uv run rootcause-agents investigate S03 --system mas
make eval                      # → eval/results/summary.md
```

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
