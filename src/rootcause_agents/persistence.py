"""Persistent investigation sessions.

- LangGraph checkpointer: ``AsyncPostgresSaver`` when ``DATABASE_URL`` is set (local Docker Postgres, see
  docker-compose.yml), otherwise a process-wide ``InMemorySaver``. Every super-step of the graph is checkpointed,
  so a session paused for human input (``interrupt``) can be resumed later, even from a new process.
- Checkpoint deserialization is restricted to an explicit allowlist (our Pydantic schemas) instead of
  LangGraph's permissive default.
- A small session registry (one row per investigation) lets the UI list sessions and the ones waiting for input.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from rootcause_agents import schemas
from rootcause_agents.config import Settings

_MEMORY_SAVER: InMemorySaver | None = None


def checkpoint_serde() -> JsonPlusSerializer:
    allowed = [("rootcause_agents.schemas", name) for name, obj in inspect.getmembers(schemas, inspect.isclass)
               if issubclass(obj, BaseModel) and obj.__module__ == schemas.__name__]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


@asynccontextmanager
async def open_checkpointer(settings: Settings) -> AsyncIterator[Any]:
    if settings.database_url:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(settings.database_url, serde=checkpoint_serde()) as cp:
            await cp.setup()  # idempotent: creates/migrates checkpoint tables
            yield cp
        return
    global _MEMORY_SAVER
    if _MEMORY_SAVER is None:
        _MEMORY_SAVER = InMemorySaver(serde=checkpoint_serde())
    yield _MEMORY_SAVER


# ---------------------------------------------------------------- session registry
FIELDS = ("status", "system", "scenario_id", "title", "question", "outcome", "out_dir", "trace_url", "cost_usd")


class InMemorySessionRegistry:
    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}

    def get(self, thread_id: str) -> dict[str, Any] | None:
        return self._rows.get(thread_id)

    async def aget(self, thread_id: str) -> dict[str, Any] | None:
        return self.get(thread_id)

    async def aupsert(self, thread_id: str, **fields: Any) -> None:
        now = datetime.now(UTC).isoformat()
        row = self._rows.setdefault(thread_id, {"thread_id": thread_id, "created_at": now})
        row.update({k: v for k, v in fields.items() if k in FIELDS}, updated_at=now)

    async def alist(self, limit: int = 50) -> list[dict[str, Any]]:
        return sorted(self._rows.values(), key=lambda r: r["updated_at"], reverse=True)[:limit]


_MEMORY_REGISTRY = InMemorySessionRegistry()

DDL = """
CREATE TABLE IF NOT EXISTS rootcause_sessions (
    thread_id   TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    system      TEXT,
    scenario_id TEXT,
    title       TEXT,
    question    JSONB,
    outcome     TEXT,
    out_dir     TEXT,
    trace_url   TEXT,
    cost_usd    DOUBLE PRECISION,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)"""


class PostgresSessionRegistry:
    def __init__(self, conn: Any):
        self.conn = conn

    async def setup(self) -> None:
        await self.conn.execute(DDL)

    async def aget(self, thread_id: str) -> dict[str, Any] | None:
        cur = await self.conn.execute("SELECT * FROM rootcause_sessions WHERE thread_id = %s", (thread_id,))
        row = await cur.fetchone()
        return _row(cur, row) if row else None

    async def aupsert(self, thread_id: str, **fields: Any) -> None:
        f = {k: (json.dumps(v) if k == "question" and v is not None else v) for k, v in fields.items() if k in FIELDS}
        cols = ", ".join(f)
        ph = ", ".join(["%s"] * len(f))
        upd = ", ".join(f"{k} = EXCLUDED.{k}" for k in f)
        await self.conn.execute(
            f"INSERT INTO rootcause_sessions (thread_id, {cols}) VALUES (%s, {ph}) "
            f"ON CONFLICT (thread_id) DO UPDATE SET {upd}, updated_at = now()", (thread_id, *f.values()))

    async def alist(self, limit: int = 50) -> list[dict[str, Any]]:
        cur = await self.conn.execute("SELECT * FROM rootcause_sessions ORDER BY updated_at DESC LIMIT %s", (limit,))
        return [_row(cur, r) for r in await cur.fetchall()]


def _row(cur: Any, row: tuple) -> dict[str, Any]:
    d = {c.name: v for c, v in zip(cur.description, row, strict=True)}
    for k in ("created_at", "updated_at"):
        if isinstance(d.get(k), datetime):
            d[k] = d[k].isoformat()
    return d


@asynccontextmanager
async def open_registry(settings: Settings) -> AsyncIterator[Any]:
    if settings.database_url:
        import psycopg

        async with await psycopg.AsyncConnection.connect(settings.database_url, autocommit=True) as conn:
            reg = PostgresSessionRegistry(conn)
            await reg.setup()
            yield reg
        return
    yield _MEMORY_REGISTRY
