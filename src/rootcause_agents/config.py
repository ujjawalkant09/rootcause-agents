"""Runtime settings (env-driven), model-per-role mapping and pricing."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from rootcause_agents.budget import Budget

ROOT = Path(__file__).resolve().parents[2]

ROLES_MAS = ["orchestrator", "observability", "dependency", "reasoning", "critic", "reporter"]
WORKER_ROLES = {"observability", "dependency"}
SYSTEMS = ("mas", "mas-cheap", "single")

# USD per 1M tokens (input, output) – OpenAI pricing page, checked 2026-11-19.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-6.2-astra": (8.00, 40.00),
    "gpt-6.2-sol": (2.00, 10.00),
    "gpt-6.1-terra": (1.00, 5.00),
    "gpt-6-luna": (0.10, 0.50),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICING.get(model, (0.0, 0.0))
    return round(input_tokens / 1e6 * pin + output_tokens / 1e6 * pout, 6)


class Settings(BaseModel):
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    main_model: str = "gpt-6.2-sol"
    worker_model: str = "gpt-6-luna"
    structured_method: str = "json_schema"  # or "function_calling" for models without json_schema support
    request_timeout_s: float = 120.0
    llm_max_retries: int = 2
    max_concurrency: int = 4
    budget: Budget = Field(default_factory=Budget)
    data_dir: Path = ROOT / "data"
    runs_dir: Path = ROOT / "runs"

    @classmethod
    def from_env(cls) -> Settings:
        try:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env")
        except ImportError:  # pragma: no cover
            pass
        env = os.environ
        kwargs = {
            "openai_api_key": env.get("OPENAI_API_KEY"),
            "openai_base_url": env.get("OPENAI_BASE_URL") or None,
            "main_model": env.get("MAIN_MODEL", cls.model_fields["main_model"].default),
            "worker_model": env.get("WORKER_MODEL", cls.model_fields["worker_model"].default),
            "structured_method": env.get("STRUCTURED_METHOD", "json_schema"),
        }
        budget = Budget()
        if v := env.get("MAX_TOOL_CALLS"):
            budget.max_tool_calls = int(v)
        if v := env.get("MAX_ROUNDS"):
            budget.max_rounds = int(v)
        return cls(budget=budget, **kwargs)

    def models_for(self, system: str) -> dict[str, str]:
        if system == "single":
            return {"single": self.main_model}
        if system == "mas":
            return {r: self.main_model for r in ROLES_MAS}
        if system == "mas-cheap":
            return {r: (self.worker_model if r in WORKER_ROLES else self.main_model) for r in ROLES_MAS}
        raise ValueError(f"Unknown system {system!r}; expected one of {SYSTEMS}")
