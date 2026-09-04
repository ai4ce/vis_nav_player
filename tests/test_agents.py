"""Both agents against a stand-in session: no server, no window (SDL dummy driver)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from baseline_agent import BaselineAgent  # noqa: E402
from keyboard_agent import KeyboardAgent  # noqa: E402
from vis_nav_sdk import Action, run  # noqa: E402
from vis_nav_sdk import protocol as P  # noqa: E402
from vis_nav_sdk.session import Camera, Limits, Observation, Result, SessionInfo  # noqa: E402
from vis_nav_sdk.telemetry import Telemetry  # noqa: E402
from vis_nav_sdk.viewer import Keys, Viewer  # noqa: E402


def _image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.zeros((240, 320, 3), np.uint8)
    for _ in range(12):
        x, y = rng.integers(0, 300), rng.integers(0, 220)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(image, (x, y), (x + 40, y + 30), color, -1)
    return image


def _info() -> SessionInfo:
    return SessionInfo(
        session_id="s",
        challenge={"id": "c", "name": "maze"},
        camera=Camera(320, 240, "bgr", "jpeg", np.eye(3)),
        limits=Limits(
            max_steps=200,
            max_repeat=P.MAX_REPEAT,
            attempts_used=0,
            attempts_allowed=3,
            silence_timeout_s=120,
            grace_s=300,
            session_timeout_s=3600,
        ),
        targets=[_image(100 + i) for i in range(4)],
    )


class FakeSession:
    def __init__(self) -> None:
        self.info = _info()
        self.limits = self.info.limits
        self.session_id = self.info.session_id
        self.initial_observation = Observation(_image(0), 0, 200)
        self.calls: list[tuple[int, int]] = []
        self.aborted = None
        self.result = None
        self.telemetry = Telemetry()

    def step(self, action, repeat=1):
        self.calls.append((int(action), repeat))
        used = sum(r for _, r in self.calls)
        return Observation(_image(used), used, 200 - used)

    def checkin(self):
        self.result = Result("partial", 0.1, sum(r for _, r in self.calls), "j", "s")
        return self.result

    def abort(self, reason=""):
        self.aborted = reason

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class FakeClient:
    server = "http://fake"

    def __init__(self, api_key=None, *, server=None) -> None:
        pass

    def challenge(self, challenge_id):
        return {"id": challenge_id, "name": "maze"}

    def quota(self, challenge_id):
        return {"attempts_used": 0, "attempts_allowed": 3, "running": 0, "reservation": None}

    def start_session(self, challenge_id):
        return {
            "session_id": "s",
            "token": "vns_s_t",
            "expires_at": 2**53,
            "attempts_used": 0,
            "attempts_allowed": 3,
            "max_steps": 200,
            "final_submission_link": None,
            "page_url": "https://site/challenges/c?session=s",
        }


@pytest.fixture
def session(monkeypatch, tmp_path) -> FakeSession:
    from vis_nav_sdk import rest
    from vis_nav_sdk import session as session_module

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    fake = FakeSession()
    monkeypatch.setattr(rest, "Client", FakeClient)
    monkeypatch.setattr(session_module, "redeem", lambda *a, **k: fake)
    return fake


@pytest.fixture
def keys(monkeypatch):
    """Script the keyboard: one set of held key names per act() call, then nothing."""
    script: list[set[str]] = []

    def fake_keys(self):
        held = frozenset(script.pop(0) if script else set())
        return Keys(held=held, tapped=frozenset())

    monkeypatch.setattr(Viewer, "keys", fake_keys)
    return script


def test_keyboard_agent_moves_one_tick_per_frame(session, keys):
    keys += [
        set(),  # pre-flight
        {"up"},
        {"up", "left"},
        set(),  # released: wait, no tick burnt
        {"space"},
    ]
    viewer = Viewer(scale=1, hold_s=0)
    result = run(KeyboardAgent(), "vns_c_x", viewer=viewer, fps=None, quiet=True)
    assert result is not None
    assert session.calls == [(int(Action.FORWARD), 1), (int(Action.FORWARD | Action.LEFT), 1)]


def test_the_runner_paces_a_window(session, keys):
    keys += [set()] + [{"up"}] * 6 + [{"space"}]
    viewer = Viewer(scale=1, hold_s=0)
    started = time.monotonic()
    run(KeyboardAgent(), "vns_c_x", viewer=viewer, fps=20, quiet=True)
    assert len(session.calls) == 6
    assert time.monotonic() - started >= 6 / 20 * 0.8


def test_keyboard_agent_escape_quits(session, keys):
    keys += [set(), {"escape"}]
    viewer = Viewer(scale=1, hold_s=0)
    assert run(KeyboardAgent(), "vns_c_x", viewer=viewer, quiet=True) is None
    assert session.aborted == "agent quit"


def test_keyboard_agent_needs_a_window(session):
    with pytest.raises(Exception, match="viewer"):
        run(KeyboardAgent(), "vns_c_x", viewer=False, quiet=True)
    assert session.calls == []


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    for t in range(2):
        traj = data / f"traj_{t}"
        traj.mkdir(parents=True)
        records = []
        for k in range(12):
            cv2.imwrite(str(traj / f"{k}.jpg"), _image(1000 * t + k))
            records.append(
                {"step": k, "image": f"{k}.jpg", "action": ["FORWARD" if k % 3 else "LEFT"]}
            )
        records[5]["action"] = ["FORWARD", "LEFT"]  # a combined action is skipped
        (traj / "data_info.json").write_text(json.dumps(records))
    return data


def test_baseline_indexes_localises_and_hints(dataset, tmp_path, session, keys):
    agent = BaselineAgent(
        dataset, n_clusters=4, subsample=1, top_k_shortcuts=3, cache_dir=tmp_path / "cache"
    )
    assert len(agent.frames) == 22
    assert agent.graph.number_of_nodes() == 22

    keys += [set(), {"up"}, {"space"}]
    result = run(agent, "vns_c_x", viewer=Viewer(scale=1, hold_s=0), fps=None, quiet=True)
    assert result is not None
    assert agent.goal is not None and agent.current is not None
    assert any("node" in line for line in agent.hud())
    panel = agent.panel()
    assert panel and panel[0][1].startswith("best match") and panel[0][0].shape == (240, 320, 3)

    # A second construction hits the caches.
    again = BaselineAgent(
        dataset, n_clusters=4, subsample=1, top_k_shortcuts=3, cache_dir=tmp_path / "cache"
    )
    assert np.allclose(again.database, agent.database)
