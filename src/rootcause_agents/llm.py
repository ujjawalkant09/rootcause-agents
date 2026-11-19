"""LLM abstraction: OpenAI-backed implementation (fallback chain + circuit breaker + prompt caching)
and a scripted implementation for offline tests."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from rootcause_agents.config import Settings

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0  # subset of input_tokens served from the provider's prompt cache


UsageHook = Callable[[str, str, Usage], None]  # role, model, usage
EventHook = Callable[[str, dict[str, Any]], None]
ClientFactory = Callable[[str, str | None, str | None, dict[str, Any]], Any]  # model, base_url, api_key, kwargs


class StructuredOutputError(RuntimeError):
    pass


class LLM(Protocol):
    async def structured(self, role: str, messages: list[BaseMessage], schema: type[T]) -> T: ...

    async def tool_step(self, role: str, messages: list[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage: ...

    def model_for(self, role: str) -> str: ...


def usage_from_message(msg: Any) -> Usage:
    meta = getattr(msg, "usage_metadata", None) or {}
    details = meta.get("input_token_details") or {}
    return Usage(int(meta.get("input_tokens", 0) or 0), int(meta.get("output_tokens", 0) or 0),
                 int(details.get("cache_read", 0) or 0))


class CircuitBreaker:
    """Skip an endpoint after ``threshold`` consecutive failures until ``cooldown_s`` has passed (half-open)."""

    def __init__(self, threshold: int = 3, cooldown_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.threshold, self.cooldown_s, self.clock = threshold, cooldown_s, clock
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}

    def allow(self, key: str) -> bool:
        return self.clock() >= self._open_until.get(key, float("-inf"))

    def success(self, key: str) -> None:
        self._failures.pop(key, None)
        self._open_until.pop(key, None)

    def failure(self, key: str) -> bool:
        """Record a failure; returns True if this failure opened the circuit."""
        n = self._failures.get(key, 0) + 1
        self._failures[key] = n
        if n >= self.threshold:
            self._open_until[key] = self.clock() + self.cooldown_s
            self._failures[key] = 0
            return True
        return False


@dataclass(frozen=True)
class Endpoint:
    model: str
    base_url: str | None
    api_key: str | None
    primary_provider: bool

    @property
    def key(self) -> str:
        return f"{self.base_url or 'openai'}::{self.model}"


def _default_client_factory(model: str, base_url: str | None, api_key: str | None, model_kwargs: dict[str, Any],
                            settings: Settings) -> Any:
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {"model": model, "api_key": api_key, "timeout": settings.request_timeout_s,
                              "max_retries": settings.llm_max_retries}
    if base_url:
        kwargs["base_url"] = base_url
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    return ChatOpenAI(**kwargs)


class OpenAILLM:
    """Per-role model chain: primary → FALLBACK_MODELS → optional second provider. Usage is reported for every
    attempt that reached the provider (failed attempts cost tokens too)."""

    def __init__(self, settings: Settings, models: dict[str, str], on_usage: UsageHook | None = None,
                 on_event: EventHook | None = None, client_factory: ClientFactory | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.settings = settings
        self.models = models
        self.on_usage = on_usage or (lambda *_: None)
        self.on_event = on_event or (lambda *_: None)
        self._factory = client_factory or (lambda m, b, k, kw: _default_client_factory(m, b, k, kw, settings))
        self._clients: dict[tuple[str, str], Any] = {}
        self._sem = asyncio.Semaphore(settings.max_concurrency)  # protects provider rate limits
        self.breaker = CircuitBreaker(settings.breaker_threshold, settings.breaker_cooldown_s, clock)

    def model_for(self, role: str) -> str:
        return self.models.get(role) or self.settings.main_model

    def chain_for(self, role: str) -> list[Endpoint]:
        s = self.settings
        primary = self.model_for(role)
        chain = [Endpoint(primary, s.openai_base_url, s.openai_api_key, True)]
        for m in s.fallback_models:
            if m != primary:
                chain.append(Endpoint(m, s.openai_base_url, s.openai_api_key, True))
        if s.fallback_base_url:
            for m in s.fallback_provider_models or [primary]:
                chain.append(Endpoint(m, s.fallback_base_url, s.fallback_api_key or s.openai_api_key, False))
        return chain

    def _client(self, role: str, ep: Endpoint) -> Any:
        cache_key = (ep.key, role)
        if cache_key not in self._clients:
            # prompt_cache_key routes same-prefix requests to warm cache; OpenAI-specific, so primary provider only
            kw = {"prompt_cache_key": f"rootcause-agents:{role}"} if self.settings.prompt_cache and ep.primary_provider \
                else {}
            self._clients[cache_key] = self._factory(ep.model, ep.base_url, ep.api_key, kw)
        return self._clients[cache_key]

    async def _with_fallback(self, role: str, attempt: Callable[[Any, Endpoint], Awaitable[Any]]) -> Any:
        chain = self.chain_for(role)
        allowed = [ep for ep in chain if self.breaker.allow(ep.key)] or chain  # all open → try everything anyway
        for ep in chain:
            if ep not in allowed:
                self.on_event("circuit_open", {"role": role, "model": ep.model, "skipped": True})
        last_err: Exception | None = None
        for i, ep in enumerate(allowed):
            try:
                async with self._sem:
                    result = await attempt(self._client(role, ep), ep)
                self.breaker.success(ep.key)
                if ep is not chain[0]:
                    self.on_event("llm_fallback", {"role": role, "from_model": chain[0].model, "to_model": ep.model})
                return result
            except Exception as e:  # provider errors, timeouts, unparseable output → next endpoint
                last_err = e
                opened = self.breaker.failure(ep.key)
                self.on_event("llm_error", {"role": role, "model": ep.model, "error": f"{type(e).__name__}: {e}"[:300],
                                            "attempt": i + 1})
                if opened:
                    self.on_event("circuit_open", {"role": role, "model": ep.model, "skipped": False})
        assert last_err is not None
        raise last_err

    async def structured(self, role: str, messages: list[BaseMessage], schema: type[T]) -> T:
        async def attempt(client: Any, ep: Endpoint) -> T:
            runnable = client.with_structured_output(schema, method=self.settings.structured_method, include_raw=True)
            out = await runnable.ainvoke(messages)
            self.on_usage(role, ep.model, usage_from_message(out.get("raw")))
            parsed = out.get("parsed")
            if parsed is None:
                raise StructuredOutputError(f"{role}: could not parse {schema.__name__}: {out.get('parsing_error')}")
            return parsed

        return await self._with_fallback(role, attempt)

    async def tool_step(self, role: str, messages: list[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage:
        async def attempt(client: Any, ep: Endpoint) -> AIMessage:
            msg = await client.bind_tools(tools).ainvoke(messages)
            self.on_usage(role, ep.model, usage_from_message(msg))
            return msg

        return await self._with_fallback(role, attempt)


StructuredHandler = Callable[[list[BaseMessage], type[BaseModel]], BaseModel]
ToolStepHandler = Callable[[list[BaseMessage], list[dict[str, Any]]], AIMessage]


class ScriptedLLM:
    """Deterministic fake: per-role handlers (callables) or queues (lists). Used for offline tests."""

    def __init__(
        self,
        structured: dict[str, StructuredHandler | list[BaseModel]] | None = None,
        tool_steps: dict[str, ToolStepHandler | list[AIMessage]] | None = None,
        on_usage: UsageHook | None = None,
        usage_per_call: Usage = Usage(100, 50, 0),
        model: str = "scripted",
    ):
        self._structured = {k: (list(v) if isinstance(v, list) else v) for k, v in (structured or {}).items()}
        self._tool_steps = {k: (list(v) if isinstance(v, list) else v) for k, v in (tool_steps or {}).items()}
        self.on_usage = on_usage or (lambda *_: None)
        self.usage_per_call = usage_per_call
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def model_for(self, role: str) -> str:
        return self.model

    def _usage(self, role: str) -> None:
        self.on_usage(role, self.model, self.usage_per_call)

    async def structured(self, role: str, messages: list[BaseMessage], schema: type[T]) -> T:
        self.calls.append({"role": role, "kind": "structured", "schema": schema.__name__, "messages": messages})
        self._usage(role)
        h = self._structured.get(role)
        if h is None:
            raise StructuredOutputError(f"ScriptedLLM: no structured handler for role {role!r}")
        if isinstance(h, list):
            if not h:
                raise StructuredOutputError(f"ScriptedLLM: script exhausted for role {role!r}")
            out = h.pop(0)
            if isinstance(out, Exception):
                raise out
            return out  # type: ignore[return-value]
        return h(messages, schema)  # type: ignore[return-value]

    async def tool_step(self, role: str, messages: list[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage:
        self.calls.append({"role": role, "kind": "tool_step", "messages": list(messages)})
        self._usage(role)
        h = self._tool_steps.get(role)
        if h is None:
            return AIMessage(content="No further tool calls.")
        if isinstance(h, list):
            return h.pop(0) if h else AIMessage(content="No further tool calls.")
        return h(messages, tools)
