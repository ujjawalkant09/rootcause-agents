"""Deterministically synthesise telemetry from a scenario's declared causal signals.

Telemetry window: reported_at - 180 min .. reported_at + 30 min at 1-minute resolution.
"""

from __future__ import annotations

import random
import re
from datetime import timedelta

import numpy as np

from rootcause_agents.sim.scenario import HistoricalIncident, LogBurst, MetricShift, Scenario, Topology
from rootcause_agents.sim.store import ConfigChange, Deployment, LogEntry, TelemetryStore

WINDOW_BEFORE_MIN = 180
WINDOW_AFTER_MIN = 30

# metric -> (baseline value, relative noise std)
DEFAULT_METRICS: dict[str, tuple[float, float]] = {
    "latency_p50": (45.0, 0.05),
    "latency_p99": (220.0, 0.08),
    "error_rate": (0.002, 0.25),
    "rps": (120.0, 0.06),
    "cpu": (35.0, 0.06),
    "memory": (52.0, 0.03),
}
EXTRA_METRICS: dict[str, tuple[float, float]] = {
    "db_connections": (30.0, 0.06),
    "query_latency_p99": (40.0, 0.08),
    "disk_used_pct": (55.0, 0.005),
    "queue_depth": (12.0, 0.15),
    "cache_hit_rate": (0.94, 0.01),
    "evictions": (5.0, 0.3),
    "consumer_lag": (20.0, 0.15),
    "replication_lag_s": (0.5, 0.2),
    "lock_wait_ms": (3.0, 0.2),
    "cert_days_remaining": (30.0, 0.0),
    "thread_pool_active": (20.0, 0.08),
}
BOUNDED_0_1 = {"error_rate", "cache_hit_rate"}
BOUNDED_PCT = {"cpu", "memory", "disk_used_pct"}

BACKGROUND_LOGS = [
    ("INFO", "request completed path=/api/v1/{path} status=200 duration={ms}ms"),
    ("INFO", "health check ok"),
    ("WARN", "slow request path=/api/v1/{path} duration={ms}ms"),
]
_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _fill(template: str, rng: random.Random) -> str:
    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name in ("ms", "duration"):
            return str(rng.randint(5, 9000))
        if name == "path":
            return rng.choice(["orders", "cart", "users", "items", "pay"]) + f"/{rng.randint(1, 99999)}"
        if name == "id":
            return f"{rng.getrandbits(32):08x}"
        return str(rng.randint(1, 999))

    return _PLACEHOLDER.sub(repl, template)


def _metrics_for(service: str, topology: Topology) -> list[str]:
    spec = topology.services[service]
    return list(DEFAULT_METRICS) + list(spec.metrics or [])


def generate(scenario: Scenario, topology: Topology, history: list[HistoricalIncident]) -> TelemetryStore:
    t0 = scenario.incident.reported_at
    start = t0 - timedelta(minutes=WINDOW_BEFORE_MIN)
    end = t0 + timedelta(minutes=WINDOW_AFTER_MIN)
    n = WINDOW_BEFORE_MIN + WINDOW_AFTER_MIN + 1
    times = [start + timedelta(minutes=i) for i in range(n)]
    nrng = np.random.default_rng(scenario.seed)
    rng = random.Random(scenario.seed)

    store = TelemetryStore(start=start, end=end, reported_at=t0, topology=topology, history=list(history))

    def idx(rel_min: int) -> int:
        return max(0, min(n, WINDOW_BEFORE_MIN + rel_min))

    # ---- metrics: baseline + noise, then apply shifts
    for svc in sorted(topology.services):
        for metric in _metrics_for(svc, topology):
            base, rel = DEFAULT_METRICS.get(metric) or EXTRA_METRICS.get(metric, (10.0, 0.1))
            values = base * (1 + nrng.normal(0, rel, n)) if rel else np.full(n, base)
            store.metrics[(svc, metric)] = list(zip(times, values.tolist(), strict=False))

    for sig in scenario.signals:
        if isinstance(sig, MetricShift):
            key = (sig.service, sig.metric)
            if key not in store.metrics:
                base, rel = EXTRA_METRICS.get(sig.metric) or DEFAULT_METRICS.get(sig.metric, (10.0, 0.1))
                store.metrics[key] = list(zip(times, (base * (1 + nrng.normal(0, rel, n))).tolist(), strict=False))
            series = store.metrics[key]
            a, b = idx(sig.at), idx(sig.until) if sig.until is not None else n
            vals = [v for _, v in series]
            for i in range(a, b):
                progress = min(1.0, (i - a + 1) / max(1, sig.ramp))
                if sig.to is not None:
                    target = sig.to * (1 + nrng.normal(0, 0.02))
                    vals[i] = vals[i] + (target - vals[i]) * progress
                elif sig.factor is not None:
                    vals[i] = vals[i] * (1 + (sig.factor - 1) * progress)
            store.metrics[key] = list(zip(times, vals, strict=False))

    for (svc, metric), series in store.metrics.items():
        lo, hi = 0.0, None
        if metric in BOUNDED_0_1:
            hi = 1.0
        elif metric in BOUNDED_PCT:
            hi = 100.0
        store.metrics[(svc, metric)] = [(t, round(min(max(v, lo), hi) if hi else max(v, lo), 4)) for t, v in series]

    # ---- logs: background noise for every service
    logs: list[LogEntry] = []
    for svc in sorted(topology.services):
        for t in times:
            for level, tmpl in BACKGROUND_LOGS:
                rate = 2.0 if level == "INFO" else 0.15
                for _ in range(nrng.poisson(rate)):
                    ts = t + timedelta(seconds=rng.randint(0, 59))
                    logs.append(LogEntry(ts, svc, level, _fill(tmpl, rng), f"{rng.getrandbits(48):012x}"))
    for sig in scenario.signals:
        if isinstance(sig, LogBurst):
            a, b = idx(sig.at), idx(sig.until) if sig.until is not None else n
            for i in range(a, b):
                for _ in range(nrng.poisson(sig.rate)):
                    ts = times[i] + timedelta(seconds=rng.randint(0, 59))
                    logs.append(LogEntry(ts, sig.service, sig.level, _fill(sig.template, rng),
                                         f"{rng.getrandbits(48):012x}"))
    logs.sort(key=lambda e: (e.ts, e.service, e.message))
    store.logs = logs

    store.deployments = sorted(
        (
            Deployment(d.id, d.service, d.version, t0 + timedelta(minutes=d.at), d.author, d.status,
                       d.diff_summary, tuple(d.files_changed), d.rollback_of)
            for d in scenario.deployments
        ),
        key=lambda d: d.time,
    )
    store.config_changes = sorted(
        (
            ConfigChange(c.id, c.service, c.key, c.old, c.new, t0 + timedelta(minutes=c.at), c.actor, c.source)
            for c in scenario.config_changes
        ),
        key=lambda c: c.time,
    )
    return store
