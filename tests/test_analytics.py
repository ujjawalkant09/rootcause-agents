from datetime import UTC, datetime, timedelta

import numpy as np

from rootcause_agents.tools.analytics import cluster_logs, detect_change, to_template

T0 = datetime(2026, 3, 10, 14, 0, tzinfo=UTC)


def test_template_masks_variables():
    a = to_template("ConnectionPoolTimeout: waited 5012ms for connection id=4f9c2a1b-1111-2222-3333-444455556666")
    b = to_template("ConnectionPoolTimeout: waited 87ms for connection id=0a0b0c0d-1111-2222-3333-444455556666")
    assert a == b
    assert to_template("GET /api/v1/orders/12345 from 10.0.0.12") == to_template("GET /api/v1/orders/9 from 10.1.2.3")


def test_cluster_logs_counts_and_orders():
    class E:  # minimal log entry
        def __init__(self, ts, msg, level="ERROR"):
            self.ts, self.message, self.level, self.trace_id, self.service = ts, msg, level, "t", "s"

    entries = [E(T0 + timedelta(seconds=i), f"timeout after {i}ms") for i in range(50)]
    entries += [E(T0, f"disk full on /var/{i}") for i in range(5)]
    clusters = cluster_logs(entries, max_clusters=10)
    assert clusters[0]["count"] == 50
    assert clusters[0]["first_seen"] <= clusters[0]["last_seen"]
    assert len(clusters[0]["sample_lines"]) <= 3
    assert clusters[1]["count"] == 5


def test_detect_change_finds_onset():
    rng = np.random.default_rng(0)
    ts = [T0 + timedelta(minutes=i) for i in range(120)]
    vals = list(30 + rng.normal(0, 1.5, 120))
    for i in range(90, 120):
        vals[i] = 95 + rng.normal(0, 1)
    res = detect_change(ts, vals, baseline_end=ts[60])
    assert res["changed"] is True
    onset = datetime.fromisoformat(res["onset"])
    assert ts[88] <= onset <= ts[92]
    assert res["direction"] == "up"


def test_detect_change_no_change_on_noise():
    rng = np.random.default_rng(1)
    ts = [T0 + timedelta(minutes=i) for i in range(120)]
    vals = list(30 + rng.normal(0, 1.5, 120))
    res = detect_change(ts, vals, baseline_end=ts[60])
    assert res["changed"] is False
