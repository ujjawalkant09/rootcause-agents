# References (research used for this proposal)

## Agent architecture guidance
- Anthropic – Building effective agents (orchestrator–workers, evaluator–optimizer): https://www.anthropic.com/engineering/building-effective-agents
- Anthropic – How we built our multi-agent research system (lead/sub-agent briefs, ~15× token cost of multi-agent): https://www.anthropic.com/engineering/multi-agent-research-system

## Multi-agent vs single-agent evidence
- Tran & Kiela (2026), single vs multi-agent under equal budgets: https://arxiv.org/pdf/2604.02460
- VentureBeat – "More agents isn't a reliable path to better enterprise AI": https://venturebeat.com/ai/research-shows-more-agents-isnt-a-reliable-path-to-better-enterprise-ai
- Amazon study summary (4.7× tokens for +8.2 pts): https://fourweekmba.com/ai-amazon-multi-agent-ai-uses-4-7x-tokens-for-an-8-point-gain/
- Single-agent "swarm tax" at equal budget: https://particula.tech/blog/single-agent-vs-multi-agent-swarm-tax-equal-budget

## LLM root-cause analysis research
- OpenRCA benchmark (ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/file/d29b8d53678015079e1d245c023e49d2-Paper-Conference.pdf
- Why do AI agents systematically fail at cloud RCA? (12 pitfall types): https://arxiv.org/pdf/2602.09937v1
- How far can RCA go on real-world telemetry? (structured multi-agent RCA): https://www.alphaxiv.org/abs/2607.13548
- RCAFlow (AAAI): https://ojs.aaai.org/index.php/AAAI/article/view/36991/40953
- OpenDerisk – industrial multi-agent SRE framework: https://arxiv.org/html/2510.13561v1
- Cornacchia et al., agent architectures for microservice incidents (APSYS 2025): https://mcanini.github.io/papers/kashef2.apsys25.pdf

## Framework comparisons
- LangGraph vs custom vs DIY (2026): https://dev.to/lamingsrb/ai-agent-stack-in-2026-langgraph-vs-custom-vs-diy-39gg
- Agent frameworks vs platforms: https://www.augmentcode.com/tools/agent-frameworks-vs-platforms-langgraph
- Framework comparison, and when to go custom: https://www.raftlabs.com/blog/ai-agent-framework-comparison
- LangGraph, Temporal & custom workflows guide: https://dev.to/paxrel/ai-agent-orchestration-langgraph-temporal-amp-custom-workflows-2026-guide-4p9n

> Note: several blog sources are secondary or vendor-authored. Figures quoted in the docs are indicative, and the primary papers should be consulted before citing them.

## Chosen stack (added 2026-11-19)
- OpenAI models page (gpt-6.2-astra / gpt-6.2-sol / gpt-6.1-terra / gpt-6-luna): https://developers.openai.com/api/docs/models
- LangGraph Graph API – control flow (reducers, conditional edges, recursion_limit, Send/Command): https://docs.langchain.com/oss/python/langgraph/graph-api-control-flow
- LangGraph parallel fan-out / failure handling (forum): https://forum.langchain.com/t/parallel-nodes-how-to-manage-failures-or-exceptions/3226
- LangGraph map-reduce with Send: https://dev.to/aiengineering/map-reduce-with-the-send-api-in-langgraph-1p5o
