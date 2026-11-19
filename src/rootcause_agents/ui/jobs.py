"""Background jobs for the UI.

An investigation runs in a worker thread with its own event loop, so Streamlit reruns (any click while it runs)
only redraw the page; they no longer kill the run. The page polls ``Job.lines()`` to show the live log.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field
from typing import Any

from rootcause_agents.ui.view import format_event

Echo = Callable[[Any], None]


@dataclass
class Job:
    label: str
    future: Future
    events: list[Any] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def submit(cls, pool: Executor, label: str, work: Callable[[Echo], Awaitable[Any]],
               context: dict[str, Any] | None = None) -> Job:
        job = cls(label=label, future=Future(), context=context or {})

        def echo(ev: Any) -> None:
            with job._lock:
                job.events.append(ev)

        job.future = pool.submit(lambda: asyncio.run(work(echo)))
        return job

    @property
    def done(self) -> bool:
        return self.future.done()

    def result(self) -> Any:
        return self.future.result()

    def lines(self) -> list[str]:
        with self._lock:
            events = list(self.events)
        out: list[str] = []
        for ev in events:
            if isinstance(ev, str):
                out.append(f"- {ev}")
                continue
            if ev.get("event") in ("run_start", "run_resume") and ev.get("trace_url"):
                out.append(f"📡 [Live trace in Langfuse]({ev['trace_url']})")
            line = format_event(ev)
            if line:
                out.append(line)
        return out
