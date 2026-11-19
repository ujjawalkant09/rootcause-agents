from datetime import timedelta

import numpy as np

from rootcause_agents.sim.generator import generate


def test_scenario_loads(scenario):
    assert scenario.id == "T01"
    assert scenario.incident.affected_services == ["checkout"]
    assert scenario.ground_truth.root_cause_component == "payments-db"
    assert scenario.signals[0].type == "metric_shift"


def test_generation_is_deterministic(scenario, topology):
    a = generate(scenario, topology, history=[])
    b = generate(scenario, topology, history=[])
    assert a.metric_series("payments-db", "db_connections") == b.metric_series("payments-db", "db_connections")
    assert len(a.logs) == len(b.logs)


def test_metric_shift_applied_after_onset(store, scenario):
    t0 = scenario.incident.reported_at
    series = store.metric_series("payments-db", "db_connections")
    before = [v for ts, v in series if ts < t0 - timedelta(minutes=25)]
    after = [v for ts, v in series if t0 - timedelta(minutes=12) <= ts <= t0]
    assert np.mean(after) > 90
    assert np.mean(before) < 50


def test_factor_shift(store, scenario):
    t0 = scenario.incident.reported_at
    series = store.metric_series("checkout", "latency_p99")
    before = np.mean([v for ts, v in series if ts < t0 - timedelta(minutes=30)])
    after = np.mean([v for ts, v in series if t0 - timedelta(minutes=10) <= ts <= t0])
    assert after > before * 10


def test_log_burst_generated(store, scenario):
    t0 = scenario.incident.reported_at
    errs = [e for e in store.logs if e.service == "payments" and "ConnectionPoolTimeout" in e.message]
    assert len(errs) > 300
    assert min(e.ts for e in errs) >= t0 - timedelta(minutes=17)


def test_background_logs_for_every_service(store, topology):
    services = {e.service for e in store.logs}
    assert set(topology.services) <= services


def test_deployments_resolved_to_absolute_time(store, scenario):
    d = store.deployments[0]
    assert d.id == "D-100"
    assert d.time == scenario.incident.reported_at - timedelta(minutes=14)


def test_service_specific_metrics(store):
    assert "db_connections" in store.available_metrics("payments-db")
    assert "db_connections" not in store.available_metrics("checkout")
    assert "latency_p99" in store.available_metrics("checkout")
