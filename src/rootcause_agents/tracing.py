"""Structured JSONL tracing of every LLM call, tool call, state change and verdict."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class Tracer:
    def __init__(self, path: Path | None = None, echo: Any | None = None, append: bool = False):
        self.path = path
        self.events: list[dict[str, Any]] = []
        self.echo = echo  # optional callable(event) for live CLI output
        self._t0 = time.monotonic()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not append:  # a resumed session keeps writing to the same trace
                path.write_text("")

    def event(self, kind: str, data: dict[str, Any] | None = None, **kw: Any) -> None:
        ev = {"ts": datetime.now(UTC).isoformat(), "t": round(time.monotonic() - self._t0, 3),
              "event": kind, **(data or {}), **kw}
        self.events.append(ev)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(ev, default=str) + "\n")
        if self.echo:
            try:  # the echo is a display side channel (CLI/UI); it must never break an investigation
                self.echo(ev)
            except Exception as e:
                log.warning("trace echo failed for %s: %s", kind, e)

    def count(self, kind: str) -> int:
        return sum(1 for e in self.events if e["event"] == kind)
