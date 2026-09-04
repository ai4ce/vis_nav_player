"""
What one ``step()`` cost, and where.

Every number here is measured on this machine. ``rtt_ms`` is stamped on the IO thread the
instant the last frame arrives, so it does not include anything the calling thread was
doing; ``server_ms`` is what the gateway reports it held the step for; the difference is
the network. ``think_ms`` is the caller's own time between two steps -- the same quantity
the gateway measures from the other end and reports as ``result.think``.
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StepTiming:
    seq: int
    repeat: int
    frames: int
    payload_bytes: int
    rtt_ms: float
    server_ms: float | None
    decode_ms: float
    think_ms: float | None
    at: float
    """``time.perf_counter()`` when the step completed."""

    @property
    def network_ms(self) -> float | None:
        if self.server_ms is None:
            return None
        return max(0.0, self.rtt_ms - self.server_ms)


class Telemetry:
    """A rolling window of :class:`StepTiming`, with the summaries a HUD wants."""

    def __init__(self, window: int = 100) -> None:
        self._steps: deque[StepTiming] = deque(maxlen=window)
        self.count = 0
        self.ticks = 0
        self.payload_bytes = 0

    def record(self, timing: StepTiming) -> None:
        self._steps.append(timing)
        self.count += 1
        self.ticks += timing.repeat
        self.payload_bytes += timing.payload_bytes

    @property
    def last(self) -> StepTiming | None:
        return self._steps[-1] if self._steps else None

    def values(self, field: str) -> list[float]:
        """The window's values of one field, oldest first, ``None`` skipped."""
        return [v for v in (getattr(s, field) for s in self._steps) if v is not None]

    def median(self, field: str) -> float | None:
        values = self.values(field)
        return statistics.median(values) if values else None

    def steps_per_second(self) -> float | None:
        """Round trips per second over the window."""
        if len(self._steps) < 2:
            return None
        span = self._steps[-1].at - self._steps[0].at
        return (len(self._steps) - 1) / span if span > 0 else None

    def ticks_per_second(self) -> float | None:
        if len(self._steps) < 2:
            return None
        span = self._steps[-1].at - self._steps[0].at
        return sum(s.repeat for s in list(self._steps)[1:]) / span if span > 0 else None

    def summary(self) -> dict[str, Any]:
        return {
            "steps": self.count,
            "ticks": self.ticks,
            "payload_bytes": self.payload_bytes,
            "rtt_ms_p50": self.median("rtt_ms"),
            "server_ms_p50": self.median("server_ms"),
            "network_ms_p50": self.median("network_ms"),
            "decode_ms_p50": self.median("decode_ms"),
            "think_ms_p50": self.median("think_ms"),
            "steps_per_second": self.steps_per_second(),
        }


def now() -> float:
    return time.perf_counter()
