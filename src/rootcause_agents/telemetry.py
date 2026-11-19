"""Optional Langfuse observability (Langfuse Python SDK v5), following Langfuse's trace best practices.

One trace per investigation (``investigate-incident``, type ``agent``) with this tree:

    investigate-incident                 agent      input: the incident · output: outcome + root cause
    ├─ plan-investigation                agent      orchestrator (one per round)
    ├─ observability-agent / dependency-agent  agent   specialists (parallel, one per task)
    │  ├─ observability / dependency     generation every model call (LangChain CallbackHandler)
    │  └─ query_logs, query_metrics, …   tool       sibling of the generation that requested it
    ├─ search-incident-history           retriever
    ├─ generate-hypotheses               agent
    ├─ critique-hypotheses               evaluator
    └─ write-report                      agent

Names are stable (no IDs/rounds in names; those go in metadata). System and scenario are tags, an evaluation
batch is a session. Resilience signals (model fallbacks, errors, open circuits, degraded nodes) are WARNING/ERROR
events. Outcome, confidence, cost, tool calls and latency are scores.

Enabled only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set. Telemetry never breaks an investigation:
every Langfuse call is guarded.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

from rootcause_agents.config import Settings

log = logging.getLogger(__name__)

TRACE_NAME = "investigate-incident"

# Resilience signals become events; everything else is already an observation (or a generation via callbacks).
FORWARDED = {
    "llm_fallback": "WARNING",
    "llm_error": "WARNING",
    "circuit_open": "WARNING",
    "node_error": "ERROR",
    "node_degraded": "ERROR",
}


class _NoopObservation:
    def update(self, **kw: Any) -> None:
        pass


class NoopTelemetry:
    enabled = False
    trace_url: str | None = None

    def callbacks(self) -> list[Any]:
        return []

    @contextmanager
    def investigation(self, *, run_id: str, system: str, scenario_id: str, title: str,
                      incident: dict[str, Any] | None = None, session_id: str | None = None) -> Iterator[None]:
        yield

    @contextmanager
    def observe(self, name: str, as_type: str = "span", *, input: Any = None,
                metadata: dict[str, Any] | None = None) -> Iterator[Any]:
        yield _NoopObservation()

    def on_event(self, ev: dict[str, Any]) -> None:
        pass

    def finish(self, report: Any) -> None:
        pass

    def pause(self, question: dict[str, Any]) -> None:
        pass


class _SafeObservation:
    def __init__(self, obs: Any):
        self._obs = obs

    def update(self, **kw: Any) -> None:
        LangfuseTelemetry._safe(lambda: self._obs.update(**kw))


class LangfuseTelemetry(NoopTelemetry):
    enabled = True

    def __init__(self, client: Any, propagate: Callable[..., Any] | None = None,
                 handler_factory: Callable[[], Any] | None = None):
        self.client = client
        if propagate is None:
            from langfuse import propagate_attributes as propagate
        self._propagate = propagate
        if handler_factory is None:
            from langfuse.langchain import CallbackHandler

            handler_factory = CallbackHandler
        self._handler_factory = handler_factory
        self._handler: Any | None = None
        self._root: Any | None = None
        self._trace_id: str | None = None
        self.trace_url = None

    def callbacks(self) -> list[Any]:
        return [self._handler] if self._handler is not None else []

    @contextmanager
    def investigation(self, *, run_id: str, system: str, scenario_id: str, title: str,
                      incident: dict[str, Any] | None = None, session_id: str | None = None) -> Iterator[None]:
        attrs: dict[str, Any] = {"trace_name": TRACE_NAME, "tags": [system, scenario_id],
                                 "metadata": {"run_id": run_id, "system": system, "scenario": scenario_id}}
        if session_id:
            attrs["session_id"] = session_id
        with self._start(TRACE_NAME, "agent", input=incident or {"title": title}) as root:
            self._root = root
            prop = self._safe(lambda: self._propagate(**attrs)) or nullcontext()
            with prop:
                # created inside the root context → generations nest under the active observation
                self._handler = self._safe(self._handler_factory)
                self._trace_id = self._safe(self.client.get_current_trace_id)
                if self._trace_id:
                    self.trace_url = self._safe(lambda: self.client.get_trace_url(trace_id=self._trace_id))
                try:
                    yield
                finally:
                    self._handler = None
        self._safe(self.client.flush)

    @contextmanager
    def observe(self, name: str, as_type: str = "span", *, input: Any = None,
                metadata: dict[str, Any] | None = None) -> Iterator[Any]:
        with self._start(name, as_type, input=input, metadata=metadata) as obs:
            yield obs

    @contextmanager
    def _start(self, name: str, as_type: str, **kw: Any) -> Iterator[Any]:
        cm = self._safe(lambda: self.client.start_as_current_observation(
            name=name, as_type=as_type, **{k: v for k, v in kw.items() if v is not None}))
        if cm is None:
            yield _NoopObservation()
            return
        with cm as obs:
            yield _SafeObservation(obs)

    def on_event(self, ev: dict[str, Any]) -> None:
        kind = ev.get("event")
        if kind not in FORWARDED:
            return
        payload = {k: v for k, v in ev.items() if k not in ("ts", "t", "event")}
        self._safe(lambda: self.client.create_event(name=kind, output=payload, level=FORWARDED[kind]))

    def pause(self, question: dict[str, Any]) -> None:
        """The session is waiting for a human: close this trace segment with the question as its output."""
        if self._root is not None:
            self._root.update(output={"status": "waiting_for_human", **question})

    def finish(self, report: Any) -> None:
        top = report.ranked_hypotheses[0] if report.ranked_hypotheses else None
        out = {"outcome": report.outcome, "root_cause": top.root_cause_component if top else None,
               "category": top.root_cause_category if top else None,
               "confidence": top.confidence if top else None, "summary": report.summary}
        if self._root is not None:
            self._root.update(output=out)
        s = report.stats
        for name, value in [("outcome", report.outcome), ("top_confidence", top.confidence if top else 0.0),
                            ("cost_usd", s.cost_usd), ("tool_calls", float(s.tool_calls)),
                            ("latency_s", s.latency_s)]:
            self._safe(lambda n=name, v=value: self.client.create_score(name=n, value=v, trace_id=self._trace_id))

    @staticmethod
    def _safe(fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as e:  # telemetry must never break an investigation
            log.warning("langfuse call failed: %s", e)
            return None


def make_telemetry(settings: Settings) -> NoopTelemetry:
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return NoopTelemetry()
    try:
        from langfuse import Langfuse

        client = Langfuse(public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key,
                          base_url=settings.langfuse_base_url)
        return LangfuseTelemetry(client)
    except Exception as e:
        log.warning("langfuse unavailable, continuing without it: %s", e)
        return NoopTelemetry()
