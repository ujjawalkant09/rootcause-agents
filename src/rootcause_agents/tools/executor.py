"""Tool executor: validation, fault injection, bounded retries, memoisation, evidence ledger, budget."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from rootcause_agents.budget import BudgetTracker
from rootcause_agents.schemas import Evidence, ToolResult
from rootcause_agents.sim.store import TelemetryStore
from rootcause_agents.tools.faults import FaultInjector
from rootcause_agents.tools.impl import TOOLS, BadInput, error_result

TRANSPORT_ERRORS = {
    "timeout": ("TIMEOUT", "Tool call timed out after 10s", True),
    "rate_limited": ("RATE_LIMITED", "Rate limited by backend, retry after 1s", True),
    "unavailable": ("UNAVAILABLE", "Backend unavailable (503)", False),
}

EventHook = Callable[[str, dict[str, Any]], None]


class ToolExecutor:
    def __init__(self, store: TelemetryStore, injector: FaultInjector, budget: BudgetTracker,
                 max_attempts: int = 3, base_backoff_s: float = 0.2, sleep_scale: float = 1.0,
                 on_event: EventHook | None = None, observe: Callable[..., Any] | None = None):
        self.store = store
        self.injector = injector
        self.budget = budget
        self.max_attempts = max_attempts
        self.base_backoff_s = base_backoff_s
        self.sleep_scale = sleep_scale
        self.on_event = on_event or (lambda *_: None)
        self._observe = observe  # e.g. Telemetry.observe → one `tool` observation per call
        self._memo: dict[str, tuple[ToolResult, Evidence | None]] = {}
        self._next_id = 0
        self.evidence: dict[str, Evidence] = {}
        self.data_gaps: list[str] = []
        self.errors = 0
        self.duplicate_calls = 0
        self._rng = random.Random(0)

    # ------------------------------------------------------------------ schema export
    def tool_schemas(self, names: list[str]) -> list[dict[str, Any]]:
        out = []
        for name in names:
            model, _, desc = TOOLS[name]
            params = model.model_json_schema()
            params.pop("title", None)
            out.append({"type": "function", "function": {"name": name, "description": desc, "parameters": params}})
        return out

    # ------------------------------------------------------------------ execution
    async def execute(self, tool: str, args: dict[str, Any], agent: str, round: int) -> tuple[ToolResult, Evidence | None]:
        if self._observe is None:
            return await self._execute(tool, args, agent, round)
        as_type = "retriever" if tool == "search_incident_history" else "tool"
        with self._observe(tool, as_type, input=args, metadata={"agent": agent, "round": round}) as obs:
            result, ev = await self._execute(tool, args, agent, round)
            obs.update(output=json.loads(result_for_llm(result, ev)) if len(result_for_llm(result, ev)) < 6000
                       else {"status": result.status, "evidence_id": ev.id if ev else None},
                       level="WARNING" if result.is_error else "DEFAULT",
                       metadata={"agent": agent, "round": round, "attempts": result.attempts,
                                 "evidence_id": ev.id if ev else None})
            return result, ev

    async def _execute(self, tool: str, args: dict[str, Any], agent: str, round: int
                       ) -> tuple[ToolResult, Evidence | None]:
        if tool not in TOOLS:
            self.errors += 1
            return error_result(tool, "BAD_INPUT", f"Unknown tool {tool!r}. Available: {', '.join(TOOLS)}"), None
        model, fn, _ = TOOLS[tool]
        try:
            parsed = model.model_validate(args or {})
        except ValidationError as e:
            self.errors += 1
            return error_result(tool, "BAD_INPUT", f"Invalid arguments: {e.errors(include_url=False)}"), None

        params = parsed.model_dump(exclude_none=True)
        key = f"{tool}:{json.dumps(params, sort_keys=True, default=str)}"
        if key in self._memo:
            self.duplicate_calls += 1
            self.on_event("tool_duplicate", {"tool": tool, "params": params, "agent": agent})
            return self._memo[key]

        if not self.budget.can_call_tool():
            return error_result(tool, "BUDGET_EXHAUSTED",
                                f"Tool budget exhausted ({self.budget.exhausted_reason()})."), None
        self.budget.tool_calls += 1

        result: ToolResult | None = None
        summary = ""
        attempt = 0
        for attempt in range(1, self.max_attempts + 1):
            fault = self.injector.check(tool, params)
            if fault is not None and fault.fault in TRANSPORT_ERRORS:
                code, msg, retryable = TRANSPORT_ERRORS[fault.fault]
                result = error_result(tool, code, msg, retryable, attempts=attempt)
                self.on_event("tool_error", {"tool": tool, "params": params, "agent": agent, "code": code,
                                             "attempt": attempt})
                if retryable and attempt < self.max_attempts:
                    delay = self.base_backoff_s * (2 ** (attempt - 1)) * (1 + self._rng.random() * 0.25)
                    await asyncio.sleep(delay * self.sleep_scale)
                    continue
                break
            try:
                result, summary = fn(self.store, parsed, fault)
                result.attempts = attempt
            except BadInput as e:
                result = error_result(tool, "BAD_INPUT", str(e), attempts=attempt)
            except Exception as e:  # defensive: tools must never crash the agent
                result = error_result(tool, "INTERNAL", f"{type(e).__name__}: {e}", attempts=attempt)
            break

        assert result is not None
        evidence: Evidence | None = None
        if result.is_error:
            self.errors += 1
            if result.error and result.error.code != "BAD_INPUT":
                self.data_gaps.append(
                    f"{tool}({_fmt(params)}) failed: {result.error.code} after {result.attempts} attempt(s)"
                )
        else:
            self._next_id += 1
            evidence = Evidence(
                id=f"E-{self._next_id}", source_tool=tool, params=params, status=result.status,
                completeness=result.completeness, summary=summary, data=result.data, warnings=result.warnings,
                collected_by=agent, round=round,
            )
            self.evidence[evidence.id] = evidence
        self.on_event("tool_call", {"tool": tool, "params": params, "agent": agent, "status": result.status,
                                    "attempts": result.attempts, "evidence_id": evidence.id if evidence else None})
        self._memo[key] = (result, evidence)
        return result, evidence


def _fmt(params: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def compact_data(tool: str, data: Any) -> Any:
    """Token diet: what the model needs to reason, not everything the ledger stores.

    - metrics: drop the downsampled series (stats + detected change point carry the signal)
    - logs: keep one sample line per cluster and drop window echo
    Full data stays in the Evidence ledger (report, grader, later queries).
    """
    if not isinstance(data, dict):
        return data
    if tool == "query_metrics":
        return {k: v for k, v in data.items() if k not in ("series", "window", "completeness_points")}
    if tool == "query_logs":
        clusters = [{**c, "sample_lines": c.get("sample_lines", [])[:1]} for c in data.get("clusters", [])]
        return {"total_by_level": data.get("total_by_level"), "clusters": clusters}
    return data


def result_for_llm(result: ToolResult, evidence: Evidence | None) -> str:
    """Compact JSON payload handed back to the model as the tool message content."""
    payload: dict[str, Any] = {"status": result.status}
    if evidence is not None:
        payload["evidence_id"] = evidence.id
        payload["summary"] = evidence.summary
        payload["data"] = compact_data(result.tool, result.data)
    if result.warnings:
        payload["warnings"] = result.warnings
    if result.completeness < 1:
        payload["completeness"] = result.completeness
    if result.error is not None:
        payload["error"] = result.error.model_dump()
        payload["attempts"] = result.attempts
    return json.dumps(payload, default=str)[:6000]
