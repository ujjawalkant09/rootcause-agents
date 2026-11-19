"""Structured JSONL tracing of every LLM call, tool call, state change and verdict."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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
            self.echo(ev)

    def count(self, kind: str) -> int:
        return sum(1 for e in self.events if e["event"] == kind)
