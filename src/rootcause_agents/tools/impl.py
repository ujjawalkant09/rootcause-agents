"""The six read-only investigation tools.

Every tool takes (store, validated input, fault) and returns (ToolResult, one-line factual summary).
Outputs are pre-aggregated and bounded so the LLM never sees raw telemetry dumps.
"""

from __future__ import annotations

import math
import re
from collections import Counter, deque
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rootcause_agents.schemas import ToolError, ToolResult
from rootcause_agents.sim.scenario import FaultSpec
from rootcause_agents.sim.store import TelemetryStore
from rootcause_agents.tools.analytics import cluster_logs, detect_change, downsample


class BadInput(Exception):
    pass


# --------------------------------------------------------------------------- inputs
class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")


_TIME_DESC = "ISO-8601 timestamp, 'HH:MM' (incident day, UTC) or relative minutes like '-45m'. Optional."


class QueryLogsInput(_In):
    service: str
    start: str | None = Field(None, description=_TIME_DESC + " Default: 60 min before the report.")
    end: str | None = Field(None, description=_TIME_DESC + " Default: 15 min after the report.")
    level: Literal["ERROR", "WARN", "INFO", "ALL"] | None = Field(None, description="Default ERROR+WARN.")
    pattern: str | None = Field(None, description="Case-insensitive substring filter.")
    max_clusters: int = Field(8, ge=1, le=20)


class QueryMetricsInput(_In):
    service: str
    metric: str = Field(description="e.g. latency_p99, error_rate, rps, cpu, memory, or a service-specific metric.")
    start: str | None = Field(None, description=_TIME_DESC + " Default: 120 min before the report.")
    end: str | None = Field(None, description=_TIME_DESC + " Default: 15 min after the report.")


class ChangesInput(_In):
    service: str | None = Field(None, description="Filter by service; omit for all services.")
    start: str | None = Field(None, description=_TIME_DESC + " Default: 180 min before the report.")
    end: str | None = Field(None, description=_TIME_DESC + " Default: 30 min after the report.")


class DependenciesInput(_In):
    service: str
    direction: Literal["up", "down", "both"] = Field(
        "both", description="down = services this one calls/depends on; up = its callers."
    )
    depth: int = Field(2, ge=1, le=3)


class HistoryInput(_In):
    query: str = Field(description="Free-text description of symptoms.")
    services: list[str] | None = None
    top_k: int = Field(3, ge=1, le=5)


# --------------------------------------------------------------------------- helpers
def parse_time(value: str | None, store: TelemetryStore, default_rel_min: int) -> datetime:
    t0 = store.reported_at
    if value is None or value == "":
        t = t0 + timedelta(minutes=default_rel_min)
    elif m := re.fullmatch(r"\s*([+-]?\d+)\s*m(in)?\s*", value):
        t = t0 + timedelta(minutes=int(m.group(1)))
    elif m := re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", value):
        t = t0.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
    else:
        try:
            t = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as e:
            raise BadInput(f"Unparseable time {value!r}. {_TIME_DESC}") from e
        if t.tzinfo is None:
            t = t.replace(tzinfo=t0.tzinfo)
    return min(max(t, store.start), store.end)


def _check_service(store: TelemetryStore, service: str | None) -> None:
    if service is not None and service not in store.topology.services:
        raise BadInput(f"Unknown service {service!r}. Known services: {', '.join(store.services)}")


def _hm(t: datetime) -> str:
    return t.strftime("%H:%M")


def _ok(tool: str, data: Any, fault: FaultSpec | None, warnings: list[str] | None = None,
        completeness: float = 1.0, empty: bool = False) -> ToolResult:
    status = "ok"
    if fault is not None and fault.fault == "partial":
        status = "partial"
    elif fault is not None and fault.fault == "stale":
        status = "stale"
    elif empty:
        status = "empty"
    return ToolResult(tool=tool, status=status, data=data, completeness=completeness, warnings=warnings or [])


# --------------------------------------------------------------------------- tools
def query_logs(store: TelemetryStore, a: QueryLogsInput, fault: FaultSpec | None) -> tuple[ToolResult, str]:
    _check_service(store, a.service)
    start, end = parse_time(a.start, store, -60), parse_time(a.end, store, 15)
    levels = {"ERROR", "WARN"} if a.level is None else ({"ERROR", "WARN", "INFO"} if a.level == "ALL" else {a.level})
    warnings: list[str] = []
    completeness = 1.0
    entries = [e for e in store.logs if e.service == a.service and start <= e.ts <= end and e.level in levels]
    if a.pattern:
        entries = [e for e in entries if a.pattern.lower() in e.message.lower()]
    if fault and fault.fault == "partial":
        entries = [e for e in entries if int(e.trace_id[-1], 16) % 2 == 0]
        completeness = 0.5
        warnings.append("Only 2 of 4 log shards responded; counts are under-reported.")
    if fault and fault.fault == "empty":
        entries = []
    totals = Counter(e.level for e in entries)
    clusters = cluster_logs(entries, max_clusters=a.max_clusters)
    data = {
        "service": a.service, "window": [start.isoformat(), end.isoformat()],
        "total_by_level": dict(totals), "clusters": clusters,
    }
    if clusters:
        top = clusters[0]
        summary = (f"Logs {a.service} {_hm(start)}–{_hm(end)} [{','.join(sorted(levels))}]: {len(entries)} entries; "
                   f"top '{top['template'][:90]}' x{top['count']} (first {top['first_seen'][11:16]}, "
                   f"last {top['last_seen'][11:16]})")
        if len(clusters) > 1:
            summary += f"; 2nd '{clusters[1]['template'][:60]}' x{clusters[1]['count']}"
    else:
        summary = f"Logs {a.service} {_hm(start)}–{_hm(end)}: no {','.join(sorted(levels))} entries"
    return _ok("query_logs", data, fault, warnings, completeness, empty=not entries), summary


def query_metrics(store: TelemetryStore, a: QueryMetricsInput, fault: FaultSpec | None) -> tuple[ToolResult, str]:
    _check_service(store, a.service)
    available = store.available_metrics(a.service)
    if a.metric not in available:
        raise BadInput(f"Metric {a.metric!r} not available for {a.service}. Available: {', '.join(available)}")
    start, end = parse_time(a.start, store, -120), parse_time(a.end, store, 15)
    warnings: list[str] = []
    series = [(t, v) for t, v in store.metric_series(a.service, a.metric) if start <= t <= end]
    if fault and fault.fault == "stale":
        cutoff = store.reported_at - timedelta(minutes=fault.stale_minutes)
        series = [(t, v) for t, v in series if t <= cutoff]
        warnings.append(f"Metrics pipeline lagging: latest datapoint {_hm(cutoff)}, expected up to {_hm(end)}.")
    if fault and fault.fault == "partial":
        series = series[: max(1, len(series) // 2)]
        warnings.append("Partial series returned (some scrape targets missing).")
    if fault and fault.fault == "empty":
        series = []
    if not series:
        return (_ok("query_metrics", {"service": a.service, "metric": a.metric, "points": 0}, fault, warnings,
                    empty=True), f"Metric {a.service}/{a.metric}: no datapoints in window")
    ts, vals = [t for t, _ in series], [v for _, v in series]
    baseline_end = ts[min(len(ts) - 1, max(10, int(len(ts) * 0.4)))]
    change = detect_change(ts, vals, baseline_end=baseline_end)
    data = {
        "service": a.service, "metric": a.metric, "window": [start.isoformat(), end.isoformat()],
        "stats": {"min": round(min(vals), 4), "max": round(max(vals), 4),
                  "mean": round(sum(vals) / len(vals), 4), "last": round(vals[-1], 4)},
        "change": change, "series": downsample(ts, vals), "completeness_points": len(series),
    }
    if change.get("changed"):
        summary = (f"Metric {a.service}/{a.metric}: baseline {change['baseline_mean']:.4g}, changed {change['direction']}"
                   f" at {change['onset'][11:16]} (x{change['magnitude_vs_baseline']}), peak {change['peak']:.4g}")
    else:
        summary = (f"Metric {a.service}/{a.metric} {_hm(start)}–{_hm(end)}: no significant change "
                   f"(mean {data['stats']['mean']:.4g}, last {data['stats']['last']:.4g})")
    completeness = 0.5 if fault and fault.fault == "partial" else 1.0
    return _ok("query_metrics", data, fault, warnings, completeness), summary


def get_deployments(store: TelemetryStore, a: ChangesInput, fault: FaultSpec | None) -> tuple[ToolResult, str]:
    _check_service(store, a.service)
    start, end = parse_time(a.start, store, -180), parse_time(a.end, store, 30)
    items = [d for d in store.deployments if start <= d.time <= end and (a.service is None or d.service == a.service)]
    if fault and fault.fault in ("empty",):
        items = []
    rows = [{"id": d.id, "service": d.service, "version": d.version, "time": d.time.isoformat(), "author": d.author,
             "status": d.status, "diff_summary": d.diff_summary, "files_changed": list(d.files_changed),
             "rollback_of": d.rollback_of} for d in items]
    scope = a.service or "all services"
    if rows:
        parts = [f"{r['id']} {r['service']} {r['version']} at {r['time'][11:16]} ({r['status']}"
                 f"{', rollback of ' + r['rollback_of'] if r['rollback_of'] else ''}): '{r['diff_summary'][:70]}'"
                 for r in rows]
        summary = f"Deployments ({scope}, {_hm(start)}–{_hm(end)}): " + "; ".join(parts)
    else:
        summary = f"Deployments ({scope}, {_hm(start)}–{_hm(end)}): none"
    return _ok("get_deployments", {"deployments": rows}, fault, empty=not rows), summary


def get_config_changes(store: TelemetryStore, a: ChangesInput, fault: FaultSpec | None) -> tuple[ToolResult, str]:
    _check_service(store, a.service)
    start, end = parse_time(a.start, store, -180), parse_time(a.end, store, 30)
    items = [c for c in store.config_changes
             if start <= c.time <= end and (a.service is None or c.service == a.service)]
    if fault and fault.fault == "empty":
        items = []
    rows = [{"id": c.id, "service": c.service, "key": c.key, "old": c.old, "new": c.new, "time": c.time.isoformat(),
             "actor": c.actor, "source": c.source} for c in items]
    scope = a.service or "all services"
    if rows:
        parts = [f"{r['id']} {r['service']} {r['key']}: {r['old']} -> {r['new']} at {r['time'][11:16]} ({r['source']})"
                 for r in rows]
        summary = f"Config changes ({scope}, {_hm(start)}–{_hm(end)}): " + "; ".join(parts)
    else:
        summary = f"Config changes ({scope}, {_hm(start)}–{_hm(end)}): none"
    return _ok("get_config_changes", {"config_changes": rows}, fault, empty=not rows), summary


def get_service_dependencies(store: TelemetryStore, a: DependenciesInput,
                             fault: FaultSpec | None) -> tuple[ToolResult, str]:
    _check_service(store, a.service)
    topo = store.topology
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []
    directions = ["down", "up"] if a.direction == "both" else [a.direction]
    for direction in directions:
        seen = {a.service}
        q = deque([(a.service, 0)])
        while q:
            cur, d = q.popleft()
            if d >= a.depth:
                continue
            for e in topo.edges:
                nxt = e.dst if direction == "down" and e.src == cur else e.src if direction == "up" and e.dst == cur else None
                if nxt is None or nxt in seen:
                    continue
                seen.add(nxt)
                nodes[nxt] = {"tier": topo.services[nxt].tier, "distance": d + 1,
                              "relation": "dependency" if direction == "down" else "caller"}
                edges.append({"from": e.src, "to": e.dst, "type": e.type})
                q.append((nxt, d + 1))
    if fault and fault.fault == "partial":
        keep = sorted(nodes)[: max(1, len(nodes) // 2)]
        nodes = {k: nodes[k] for k in keep}
    data = {"service": a.service, "nodes": nodes, "edges": edges}
    deps = [f"{n}({v['relation']},d{v['distance']})" for n, v in sorted(nodes.items(), key=lambda x: x[1]["distance"])]
    summary = f"Dependencies of {a.service} ({a.direction}, depth {a.depth}): " + (", ".join(deps) or "none")
    completeness = 0.5 if fault and fault.fault == "partial" else 1.0
    return _ok("get_service_dependencies", data, fault, completeness=completeness, empty=not nodes), summary


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2}


def search_incident_history(store: TelemetryStore, a: HistoryInput, fault: FaultSpec | None) -> tuple[ToolResult, str]:
    q = _tokens(a.query) | _tokens(" ".join(a.services or []))
    scored = []
    for inc in store.history:
        doc = _tokens(f"{inc.title} {inc.symptoms} {' '.join(inc.services)}")
        if not doc or not q:
            continue
        overlap = len(q & doc)
        score = overlap / math.sqrt(len(q) * len(doc))
        if a.services and set(a.services) & set(inc.services):
            score += 0.15
        if overlap:
            scored.append((round(min(score, 1.0), 3), inc))
    scored.sort(key=lambda x: (-x[0], x[1].id))
    rows = [{"incident_id": i.id, "title": i.title, "services": i.services, "root_cause": i.root_cause,
             "resolution": i.resolution, "similarity": s} for s, i in scored[: a.top_k]]
    if fault and fault.fault == "empty":
        rows = []
    summary = "History matches: " + ("; ".join(
        f"{r['incident_id']} '{r['title']}' (sim {r['similarity']}): root cause {r['root_cause'][:80]}" for r in rows
    ) or "none")
    return _ok("search_incident_history", {"incidents": rows}, fault, empty=not rows), summary


ToolFn = Callable[[TelemetryStore, Any, FaultSpec | None], tuple[ToolResult, str]]

TOOLS: dict[str, tuple[type[BaseModel], ToolFn, str]] = {
    "query_logs": (QueryLogsInput, query_logs,
                   "Search a service's logs in a time window. Returns error-signature clusters (template, count, "
                   "first/last seen, samples) instead of raw lines. Use to find what errors appeared and when."),
    "query_metrics": (QueryMetricsInput, query_metrics,
                      "Get one metric for one service. Returns stats, an automatically detected change point (onset "
                      "time, direction, magnitude vs baseline) and a downsampled series. Use to build the timeline."),
    "get_deployments": (ChangesInput, get_deployments,
                        "List deployments (id, service, version, time, diff summary, files, rollback info)."),
    "get_config_changes": (ChangesInput, get_config_changes,
                           "List configuration / feature-flag / infra changes (key, old -> new, time, source)."),
    "get_service_dependencies": (DependenciesInput, get_service_dependencies,
                                 "Get the dependency graph around a service: 'down' = what it calls, 'up' = callers."),
    "search_incident_history": (HistoryInput, search_incident_history,
                                "Find similar past incidents with their root causes. These are PRIORS, not proof."),
}


def error_result(tool: str, code: str, message: str, retryable: bool = False, attempts: int = 1) -> ToolResult:
    return ToolResult(tool=tool, status="error", completeness=0.0, attempts=attempts,
                      error=ToolError(code=code, message=message, retryable=retryable))
