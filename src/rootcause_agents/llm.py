"""LLM abstraction: a real OpenAI-backed implementation and a scripted one for offline tests."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from rootcause_agents.config import Settings

T = TypeVar("T", bound=BaseModel)
UsageHook = Callable[[str, str, int, int], None]  # role, model, input_tokens, output_tokens


class StructuredOutputError(RuntimeError):
    pass


class LLM(Protocol):
    async def structured(self, role: str, messages: list[BaseMessage], schema: type[T]) -> T: ...

    async def tool_step(self, role: str, messages: list[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage: ...

    def model_for(self, role: str) -> str: ...


def usage_from_message(msg: Any) -> tuple[int, int]:
    meta = getattr(msg, "usage_metadata", None) or {}
    return int(meta.get("input_tokens", 0) or 0), int(meta.get("output_tokens", 0) or 0)


class OpenAILLM:
    """ChatOpenAI per role; usage is reported through ``on_usage`` for budget + cost accounting."""

    def __init__(self, settings: Settings, models: dict[str, str], on_usage: UsageHook | None = None):
        from langchain_openai import ChatOpenAI

        self.settings = settings
        self.models = models
        self.on_usage = on_usage or (lambda *_: None)
        self._clients: dict[str, ChatOpenAI] = {}
        self._sem = asyncio.Semaphore(settings.max_concurrency)  # protects provider rate limits
        for model in set(models.values()):
            kwargs: dict[str, Any] = dict(model=model, api_key=settings.openai_api_key,
                                          timeout=settings.request_timeout_s, max_retries=settings.llm_max_retries)
            if settings.openai_base_url:
                kwargs["base_url"] = settings.openai_base_url
            self._clients[model] = ChatOpenAI(**kwargs)

    def model_for(self, role: str) -> str:
        return self.models.get(role) or self.settings.main_model

    def _client(self, role: str):
        return self._clients[self.model_for(role)]

    async def structured(self, role: str, messages: list[BaseMessage], schema: type[T]) -> T:
        runnable = self._client(role).with_structured_output(
            schema, method=self.settings.structured_method, include_raw=True
        )
        async with self._sem:
            out = await runnable.ainvoke(messages)
        i, o = usage_from_message(out.get("raw"))
        self.on_usage(role, self.model_for(role), i, o)
        parsed = out.get("parsed")
        if parsed is None:
            raise StructuredOutputError(f"{role}: could not parse {schema.__name__}: {out.get('parsing_error')}")
        return parsed

    async def tool_step(self, role: str, messages: list[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage:
        async with self._sem:
            msg = await self._client(role).bind_tools(tools).ainvoke(messages)
        i, o = usage_from_message(msg)
        self.on_usage(role, self.model_for(role), i, o)
        return msg


StructuredHandler = Callable[[list[BaseMessage], type[BaseModel]], BaseModel]
ToolStepHandler = Callable[[list[BaseMessage], list[dict[str, Any]]], AIMessage]


class ScriptedLLM:
    """Deterministic fake: per-role handlers (callables) or queues (lists). Used for offline tests."""

    def __init__(
        self,
        structured: dict[str, StructuredHandler | list[BaseModel]] | None = None,
        tool_steps: dict[str, ToolStepHandler | list[AIMessage]] | None = None,
        on_usage: UsageHook | None = None,
        usage_per_call: tuple[int, int] = (100, 50),
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
        self.on_usage(role, self.model, *self.usage_per_call)

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
