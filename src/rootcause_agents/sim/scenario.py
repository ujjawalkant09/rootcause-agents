"""Scenario + topology file models. Ground truth lives here and is read ONLY by the generator/grader."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rootcause_agents.schemas import IncidentReport, Outcome, RootCauseCategory


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IncidentSpec(_M):
    title: str
    description: str
    reported_at: datetime
    affected_services: list[str]
    symptoms: list[str]


class KeySignal(_M):
    service: str
    kind: Literal["metric", "log", "deployment", "config", "dependency"]
    metric: str | None = None


class GroundTruth(_M):
    root_cause_component: str
    root_cause_category: RootCauseCategory
    aliases: list[str] = []
    alt_categories: list[RootCauseCategory] = []  # also accepted by the grader
    expected_outcomes: list[Outcome]
    key_signals: list[KeySignal] = []


class MetricShift(_M):
    type: Literal["metric_shift"]
    service: str
    metric: str
    at: int  # minutes relative to reported_at
    until: int | None = None
    to: float | None = None
    factor: float | None = None
    ramp: int = 1


class LogBurst(_M):
    type: Literal["log_burst"]
    service: str
    template: str
    level: Literal["ERROR", "WARN", "INFO"] = "ERROR"
    at: int
    until: int | None = None
    rate: float  # per minute


class DeploymentSpec(_M):
    id: str
    service: str
    version: str
    at: int
    author: str = "ci-bot"
    status: str = "succeeded"
    diff_summary: str
    files_changed: list[str] = []
    rollback_of: str | None = None


class ConfigChangeSpec(_M):
    id: str
    service: str
    key: str
    old: str
    new: str
    at: int
    actor: str = "ops"
    source: Literal["feature_flag", "env", "infra"] = "env"


class FaultSpec(_M):
    tool: str
    fault: Literal["timeout", "unavailable", "partial", "stale", "empty", "rate_limited"]
    match: dict[str, str] = {}
    times: int | None = None  # None = every call
    stale_minutes: int = 20


class Scenario(_M):
    id: str
    seed: int = 0
    traits: list[Literal["control", "misleading", "conflicting", "tool_failure", "inconclusive", "noisy"]] = []
    incident: IncidentSpec
    ground_truth: GroundTruth
    signals: list[MetricShift | LogBurst] = Field(default_factory=list)
    deployments: list[DeploymentSpec] = []
    config_changes: list[ConfigChangeSpec] = []
    faults: list[FaultSpec] = []

    def incident_report(self) -> IncidentReport:
        """The ONLY view of a scenario that agents ever receive."""
        digest = hashlib.sha1(self.incident.title.encode()).hexdigest()[:6].upper()
        return IncidentReport(id=f"INC-{digest}", **self.incident.model_dump())


class ServiceSpec(_M):
    tier: str = "app"
    metrics: list[str] | None = None


class Edge(_M):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    src: str = Field(alias="from")
    dst: str = Field(alias="to")
    type: str = "sync"


class Topology(_M):
    services: dict[str, ServiceSpec]
    edges: list[Edge]


class HistoricalIncident(_M):
    id: str
    title: str
    services: list[str]
    symptoms: str
    root_cause: str
    resolution: str


def load_scenario(path: str | Path) -> Scenario:
    return Scenario.model_validate(yaml.safe_load(Path(path).read_text()))


def load_topology(path: str | Path) -> Topology:
    return Topology.model_validate(yaml.safe_load(Path(path).read_text()))


def load_history(path: str | Path) -> list[HistoricalIncident]:
    return [HistoricalIncident.model_validate(x) for x in yaml.safe_load(Path(path).read_text())]
