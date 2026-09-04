from __future__ import annotations

import os
import time
from typing import Any

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
    def session_id(self) -> str:
        return self._info.session_id

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


class FakeClient:
    """The REST side as the runner sees it: a challenge, a quota, and Start."""

    server = "http://fake"
    quota_extra: dict[str, Any] = {}

    def __init__(self, api_key=None, *, server=None) -> None:
        self.api_key = api_key
        self.started: list[str] = []

    def challenge(self, challenge_id):
        return {"id": challenge_id, "name": "Maze", "final_submission_link": "https://forms/x"}

    def quota(self, challenge_id):
        return {
            "attempts_used": 1,
            "attempts_allowed": 3,
            "running": 0,
            "reservation": None,
            "final_submission_link": "https://forms/x",
            **FakeClient.quota_extra,
        }

    def start_session(self, challenge_id):
        self.started.append(challenge_id)
        return {
            "session_id": "abc123def456",
            "token": f"vns_abc123def456_{len(self.started)}",
            "expires_at": int(time.time() * 1000) + 900_000,
            "attempts_used": 1,
            "attempts_allowed": 3,
            "max_steps": 500,
            "final_submission_link": "https://forms/x",
            "page_url": "https://site/challenges/c1?session=abc123def456",
        }


@pytest.fixture
def fake_connect(monkeypatch, tmp_path):
    """Route :func:`vis_nav_sdk.run` at a :class:`FakeSession` behind a :class:`FakeClient`;
    returns the sessions created. Reservations are cached under ``tmp_path``."""
    from vis_nav_sdk import rest
    from vis_nav_sdk import session as session_module

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    FakeClient.quota_extra = {}
    created: list[FakeSession] = []

    def redeem(token, **kwargs):
        session = FakeSession()
        session.connect_kwargs = {"token": token, **kwargs}
        created.append(session)
        return session

    monkeypatch.setattr(rest, "Client", FakeClient)
    monkeypatch.setattr(session_module, "redeem", redeem)
    return created
