"""Deterministic, cheap analytics so the LLM reads summaries, never raw telemetry."""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import numpy as np

_MASKS = [
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE), "<uuid>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<ip>"),
    (re.compile(r"\b0x[0-9a-f]+\b", re.IGNORECASE), "<hex>"),
    (re.compile(r"\b[0-9a-f]{8,}\b", re.IGNORECASE), "<hex>"),
    (re.compile(r"\d+"), "<n>"),
]


def to_template(message: str) -> str:
    out = message
    for pattern, repl in _MASKS:
        out = pattern.sub(repl, out)
    return out


def cluster_logs(entries: Iterable[Any], max_clusters: int = 10) -> list[dict[str, Any]]:
    """Group log entries by message template. Returns clusters sorted by count desc."""
    clusters: OrderedDict[tuple[str, str, str], dict[str, Any]] = OrderedDict()
    for e in entries:
        key = (e.service, e.level, to_template(e.message))
        c = clusters.get(key)
        if c is None:
            c = clusters[key] = {
                "service": e.service, "level": e.level, "template": key[2], "count": 0,
                "first_seen": e.ts, "last_seen": e.ts, "sample_lines": [],
            }
        c["count"] += 1
        c["first_seen"] = min(c["first_seen"], e.ts)
        c["last_seen"] = max(c["last_seen"], e.ts)
        if len(c["sample_lines"]) < 3:
            c["sample_lines"].append(e.message)
    ranked = sorted(clusters.values(), key=lambda c: (-c["count"], c["template"]))[:max_clusters]
    for c in ranked:
        c["first_seen"] = c["first_seen"].isoformat()
        c["last_seen"] = c["last_seen"].isoformat()
    return ranked


def detect_change(
    ts: list[datetime], values: list[float], baseline_end: datetime, z: float = 4.0, sustain: int = 3
) -> dict[str, Any]:
    """Find the first sustained deviation from the baseline (points before ``baseline_end``)."""
    arr = np.asarray(values, dtype=float)
    base_mask = np.asarray([t < baseline_end for t in ts])
    if base_mask.sum() < 5 or len(arr) == 0:
        return {"changed": False, "reason": "insufficient baseline"}
    base = arr[base_mask]
    mu, sd = float(base.mean()), float(base.std())
    sd = max(sd, abs(mu) * 0.02, 1e-9)
    dev = (arr - mu) / sd
    out: dict[str, Any] = {"changed": False, "baseline_mean": round(mu, 4), "baseline_std": round(sd, 4)}
    run = 0
    for i in range(len(arr)):
        if ts[i] < baseline_end:
            run = 0
            continue
        run = run + 1 if abs(dev[i]) > z else 0
        if run >= sustain:
            j = i - sustain + 1
            after = arr[j:]
            out.update(
                changed=True,
                onset=ts[j].isoformat(),
                direction="up" if dev[j] > 0 else "down",
                peak=round(float(after.max() if dev[j] > 0 else after.min()), 4),
                magnitude_vs_baseline=round(float(after.mean() / mu), 2) if mu else None,
            )
            break
    return out


def downsample(ts: list[datetime], values: list[float], max_points: int = 24) -> list[list[Any]]:
    if not ts:
        return []
    step = max(1, len(ts) // max_points)
    return [[ts[i].strftime("%H:%M"), round(float(np.mean(values[i:i + step])), 4)] for i in range(0, len(ts), step)]
