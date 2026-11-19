"""Seeded, declarative fault injection for simulated tool failures."""

from __future__ import annotations

from typing import Any

from rootcause_agents.sim.scenario import FaultSpec

TRANSPORT_FAULTS = {"timeout", "unavailable", "rate_limited"}  # call fails entirely
DATA_FAULTS = {"partial", "stale", "empty"}  # call succeeds with degraded data


class FaultInjector:
    def __init__(self, specs: list[FaultSpec]):
        self.specs = list(specs)
        self.counts = [0] * len(self.specs)

    def check(self, tool: str, params: dict[str, Any]) -> FaultSpec | None:
        """Return the fault to apply to this call (and consume one occurrence), or None."""
        for i, spec in enumerate(self.specs):
            if spec.tool != tool:
                continue
            if any(str(params.get(k)) != v for k, v in spec.match.items()):
                continue
            if spec.times is not None and self.counts[i] >= spec.times:
                continue
            self.counts[i] += 1
            return spec
        return None
