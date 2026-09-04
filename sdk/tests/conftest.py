from __future__ import annotations

import os

import numpy as np
import pytest
from vis_nav_sdk import protocol as P
from vis_nav_sdk.session import Camera, Limits, Observation, Result, SessionInfo
from vis_nav_sdk.telemetry import StepTiming, Telemetry

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def frame(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (240, 320, 3), dtype=np.uint8)


def make_info(**overrides) -> SessionInfo:
    fields = dict(
        session_id="abc123def456",
        challenge={"id": "c1", "name": "Maze"},
        camera=Camera(
            width=320,
            height=240,
            color_order="bgr",
            encoding="jpeg",
            intrinsic_matrix=np.array([[92, 0, 160], [0, 92, 120], [0, 0, 1]], float),
        ),
        limits=Limits(
            max_steps=500,
            max_repeat=P.MAX_REPEAT,
            attempts_used=0,
            attempts_allowed=3,
            idle_timeout_s=120,
            session_timeout_s=3600,
        ),
        targets=[frame(i) for i in range(4)],
    )
    fields.update(overrides)
    return SessionInfo(**fields)


class FakeSession:
    """Stands in for :class:`vis_nav_sdk.Session` under the runner: counts ticks, honours
    the budget, and remembers how it ended."""

    def __init__(self, info: SessionInfo | None = None) -> None:
        self._info = info or make_info()
        self.step_count = 0
        self.calls: list[tuple[int, int]] = []
        self.aborted: str | None = None
        self.result: Result | None = None
        self.telemetry = Telemetry()
        self.initial_observation = Observation(frame(), step=0, steps_left=self.limits.max_steps)

    @property
    def info(self) -> SessionInfo:
        return self._info

    @property
    def limits(self) -> Limits:
        return self._info.limits

    def step(self, action: int, repeat: int = 1, *, frames: str = "last") -> Observation:
        assert P.is_valid_step_action(int(action))
        assert 1 <= repeat <= self.limits.max_repeat
        assert repeat <= self.limits.max_steps - self.step_count
        self.calls.append((int(action), repeat))
        self.step_count += repeat
        self.telemetry.record(
            StepTiming(
                seq=len(self.calls),
                repeat=repeat,
                frames=1,
                payload_bytes=15_000,
                rtt_ms=12.0,
                server_ms=2.0,
                decode_ms=0.4,
                think_ms=1.0,
                at=len(self.calls) * 0.02,
            )
        )
        return Observation(
            frame(self.step_count),
            step=self.step_count,
            steps_left=self.limits.max_steps - self.step_count,
        )

    def checkin(self) -> Result:
        self.result = Result(
            goal_tier="partial",
            trans_error=0.15,
            nav_steps=self.step_count,
            job_id="job",
            session_id=self._info.session_id,
        )
        return self.result

    def abort(self, reason: str = "") -> None:
        self.aborted = reason

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@pytest.fixture
def fake_connect(monkeypatch):
    """Route :func:`vis_nav_sdk.run` at a :class:`FakeSession`; returns it."""
    from vis_nav_sdk import agent as agent_module

    created: list[FakeSession] = []

    def connect(token, **kwargs):
        session = FakeSession()
        session.connect_kwargs = {"token": token, **kwargs}
        created.append(session)
        return session

    monkeypatch.setattr(agent_module, "connect", connect)
    return created
