"""In-memory telemetry store queried by the tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from rootcause_agents.sim.scenario import HistoricalIncident, Topology


@dataclass(frozen=True, slots=True)
class LogEntry:
    ts: datetime
    service: str
    level: str
    message: str
    trace_id: str


@dataclass(frozen=True, slots=True)
class Deployment:
    id: str
    service: str
    version: str
    time: datetime
    author: str
    status: str
    diff_summary: str
    files_changed: tuple[str, ...]
    rollback_of: str | None


@dataclass(frozen=True, slots=True)
class ConfigChange:
    id: str
    service: str
    key: str
    old: str
    new: str
    time: datetime
    actor: str
    source: str


@dataclass
class TelemetryStore:
    start: datetime
    end: datetime
    reported_at: datetime
    topology: Topology
    metrics: dict[tuple[str, str], list[tuple[datetime, float]]] = field(default_factory=dict)
    logs: list[LogEntry] = field(default_factory=list)
    deployments: list[Deployment] = field(default_factory=list)
    config_changes: list[ConfigChange] = field(default_factory=list)
    history: list[HistoricalIncident] = field(default_factory=list)

    def metric_series(self, service: str, metric: str) -> list[tuple[datetime, float]]:
        return self.metrics.get((service, metric), [])

    def available_metrics(self, service: str) -> list[str]:
        return sorted(m for (s, m) in self.metrics if s == service)

    @property
    def services(self) -> list[str]:
        return sorted(self.topology.services)
