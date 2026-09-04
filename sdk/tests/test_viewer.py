from __future__ import annotations

import numpy as np
import pytest

pygame = pytest.importorskip("pygame")

from conftest import frame, make_info  # noqa: E402
from vis_nav_sdk import Action, Agent, run  # noqa: E402
from vis_nav_sdk.session import Observation  # noqa: E402
from vis_nav_sdk.telemetry import StepTiming, Telemetry  # noqa: E402
from vis_nav_sdk.viewer import Viewer, open_viewer  # noqa: E402


class Talkative(Agent):
    def __init__(self) -> None:
        self.n = 0

    def act(self, obs):
        self.n += 1
        return Action.CHECKIN if self.n > 3 else Action.FORWARD

    def hud(self):
        return [f"n={self.n}"]

    def panel(self):
        return np.full((60, 320, 3), 90, np.uint8)


def test_viewer_draws_every_stage_headless():
    viewer = Viewer(scale=1, hold_s=0)
    viewer.open()
    viewer.attach(make_info())
    telemetry = Telemetry()
    agent = Talkative()

    viewer.update(Observation(frame(), step=0, steps_left=500), telemetry, agent)
    telemetry.record(
        StepTiming(
            seq=0,
            repeat=4,
            frames=1,
            payload_bytes=15000,
            rtt_ms=20.0,
            server_ms=3.0,
            decode_ms=0.5,
            think_ms=2.0,
            at=1.0,
        )
    )
    viewer.update(Observation(frame(1), step=4, steps_left=496), telemetry, agent)
    assert viewer._size is not None and viewer._size[0] > 320
    viewer.pump()
    assert not viewer.closed
    viewer.hold("done", timeout=0.05)
    viewer.close()


def test_open_viewer_resolves_the_argument():
    assert open_viewer(False) is None
    viewer = open_viewer(None)
    assert isinstance(viewer, Viewer)
    viewer.close()
    custom = Viewer(scale=1, title="t")
    assert open_viewer(custom) is custom
    custom.close()


def test_run_with_a_viewer(fake_connect):
    result = run(Talkative(), "vns_c1_x", viewer=Viewer(scale=1, hold_s=0), quiet=True)
    assert result is not None
    assert not pygame.display.get_init()


def _key(viewer: Viewer, kind: int, key: int) -> None:
    pygame.event.post(pygame.event.Event(kind, key=key, mod=0, unicode="", scancode=0))
    viewer.pump()


def test_arrow_keys_are_seen_held_and_tapped():
    viewer = Viewer(scale=1)
    viewer.open()
    try:
        _key(viewer, pygame.KEYDOWN, pygame.K_UP)
        keys = viewer.keys()
        assert "up" in keys.held and "up" in keys.tapped

        _key(viewer, pygame.KEYDOWN, pygame.K_LEFT)
        keys = viewer.keys()
        assert keys.held == {"up", "left"} and keys.tapped == {"left"}

        _key(viewer, pygame.KEYUP, pygame.K_UP)
        keys = viewer.keys()
        assert keys.held == {"left"} and not keys.tapped

        # A tap shorter than one poll is still reported once.
        _key(viewer, pygame.KEYDOWN, pygame.K_SPACE)
        _key(viewer, pygame.KEYUP, pygame.K_SPACE)
        keys = viewer.keys()
        assert "space" in keys and "space" not in keys.held
        later = viewer.keys()
        assert "space" not in later and later.held == {"left"}

        pygame.event.post(pygame.event.Event(pygame.WINDOWFOCUSLOST))
        viewer.pump()
        assert not viewer.keys().any
    finally:
        viewer.close()
