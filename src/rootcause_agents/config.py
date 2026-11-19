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

# USD per 1M tokens (input, cached input, output) – OpenAI pricing page, checked 2026-11-19.
PRICING: dict[str, tuple[float, float, float]] = {
    "gpt-6.2-astra": (8.00, 0.40, 40.00),
    "gpt-6.2-sol": (2.00, 0.10, 10.00),
    "gpt-6.1-terra": (1.00, 0.05, 5.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    """Cost of one call; ``cached_tokens`` is the subset of ``input_tokens`` served from the prompt cache."""
    pin, pcached, pout = PRICING.get(model, (0.0, 0.0, 0.0))
    cached = min(cached_tokens, input_tokens)
    return round((input_tokens - cached) / 1e6 * pin + cached / 1e6 * pcached + output_tokens / 1e6 * pout, 6)


def _csv(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


class Settings(BaseModel):
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    main_model: str = "gpt-6.2-sol"
    worker_model: str = "gpt-6-luna"
    structured_method: str = "json_schema"  # or "function_calling" for models without json_schema support
    request_timeout_s: float = 120.0
    llm_max_retries: int = 2
    max_concurrency: int = 4
    # --- resilience: model fallback chain + optional second OpenAI-compatible provider
    fallback_models: list[str] = Field(default_factory=list)
    fallback_base_url: str | None = None
    fallback_api_key: str | None = None
    fallback_provider_models: list[str] = Field(default_factory=list)  # default: same model as primary
    breaker_threshold: int = 3
    breaker_cooldown_s: float = 60.0
    # --- cost: OpenAI prompt caching (stable prompt prefixes + per-role prompt_cache_key)
    prompt_cache: bool = True
    # --- persistence + human-in-the-loop
    database_url: str | None = None  # Postgres for LangGraph checkpoints; in-memory when unset
    hitl_mode: str = "off"  # off | stuck | review (CLI/eval default off; the UI chooses)
    # --- observability: Langfuse (optional; enabled when both keys are set)
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
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
            "fallback_models": _csv(env.get("FALLBACK_MODELS")),
            "fallback_base_url": env.get("FALLBACK_BASE_URL") or None,
            "fallback_api_key": env.get("FALLBACK_API_KEY") or None,
            "fallback_provider_models": _csv(env.get("FALLBACK_PROVIDER_MODELS")),
            "prompt_cache": env.get("PROMPT_CACHE", "1") not in ("0", "false", "False"),
            "database_url": env.get("DATABASE_URL") or None,
            "hitl_mode": env.get("HITL_MODE", "off"),
            "langfuse_public_key": env.get("LANGFUSE_PUBLIC_KEY") or None,
            "langfuse_secret_key": env.get("LANGFUSE_SECRET_KEY") or None,
            "langfuse_base_url": env.get("LANGFUSE_BASE_URL") or env.get("LANGFUSE_HOST")
            or "https://cloud.langfuse.com",
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
