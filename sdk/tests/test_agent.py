from __future__ import annotations

import numpy as np
import pytest
from vis_nav_sdk import WAIT, Action, Agent, SimError, preflight, run
from vis_nav_sdk.agent import decide


class Straight(Agent):
    def __init__(self, checkin_after: int = 3) -> None:
        self.checkin_after = checkin_after
        self.seen: list[int] = []
        self.setups = 0
        self.finished: list[object] = []

    def setup(self, info) -> None:
        self.setups += 1
        self.count = 0
        assert len(info.targets) == 4

    def act(self, obs):
        self.seen.append(obs.step)
        self.count += 1
        if self.count > self.checkin_after:
            return Action.CHECKIN
        return Action.FORWARD

    def finish(self, result) -> None:
        self.finished.append(result)


def test_decide_accepts_an_action_or_a_pair():
    assert decide(Action.FORWARD, 4) == (1, 4)
    assert decide((Action.LEFT, 6), 4) == (4, 6)
    assert decide(Action.FORWARD | Action.LEFT, 1) == (5, 1)
    with pytest.raises(SimError, match="act\\(\\) must return"):
        decide("forward", 1)
    with pytest.raises(SimError, match="act\\(\\) must return"):
        decide((Action.FORWARD, 1, 2), 1)


def test_run_drives_the_loop_and_reports(fake_connect):
    agent = Straight(checkin_after=3)
    result = run(agent, "vns_c1_x", server="http://x", repeat=4, viewer=False, quiet=True)
    (session,) = fake_connect
    assert session.connect_kwargs == {"token": "vns_c1_x", "server": "http://x"}
    assert session.calls == [(1, 4)] * 3
    assert result is session.result
    assert result.nav_steps == 12
    assert agent.finished == [result]
    # setup() ran twice: once in the pre-flight with junk, once for real.
    assert agent.setups == 2
    # act() saw the pre-flight frame, then the initial observation, then three real steps.
    assert agent.seen == [0, 0, 4, 8, 12]


def test_run_without_preflight(fake_connect):
    agent = Straight()
    run(agent, "vns_c1_x", viewer=False, check=False, quiet=True)
    assert agent.setups == 1


def test_a_tuple_overrides_the_default_repeat(fake_connect):
    class Bursty(Agent):
        def __init__(self) -> None:
            self.n = 0

        def act(self, obs):
            self.n += 1
            return [(Action.FORWARD, 7), (Action.LEFT, 2), Action.CHECKIN][self.n - 1]

    run(Bursty(), "vns_c1_x", repeat=3, viewer=False, check=False, quiet=True)
    assert fake_connect[0].calls == [(1, 7), (4, 2)]


def test_quit_aborts_unscored(fake_connect):
    class Quitter(Agent):
        def act(self, obs):
            return Action.QUIT

    assert run(Quitter(), "vns_c1_x", viewer=False, quiet=True) is None
    assert fake_connect[0].aborted == "agent quit"
    assert fake_connect[0].result is None


def test_repeat_is_clamped_to_the_budget(fake_connect):
    class Greedy(Agent):
        def act(self, obs):
            return (Action.FORWARD, 100_000)

    run(Greedy(), "vns_c1_x", viewer=False, check=False, quiet=True)
    session = fake_connect[0]
    assert session.step_count == session.limits.max_steps
    assert session.aborted == "step budget reached"
    assert all(repeat <= session.limits.max_repeat for _, repeat in session.calls)


def test_preflight_wraps_agent_errors_and_touches_every_hook():
    class Broken(Agent):
        def act(self, obs):
            return obs.image[999, 999]

    with pytest.raises(SimError, match="pre-flight") as caught:
        preflight(Broken())
    assert isinstance(caught.value.__cause__, IndexError)

    class BadHud(Agent):
        def act(self, obs):
            return Action.IDLE

        def hud(self):
            raise RuntimeError("no")

    with pytest.raises(SimError, match="pre-flight"):
        preflight(BadHud())


def test_preflight_honours_the_camera_size():
    shapes = []

    class Shape(Agent):
        def setup(self, info):
            shapes.append(info.camera.width)

        def act(self, obs):
            shapes.append(obs.image.shape)
            return Action.FORWARD

    preflight(Shape(), width=640, height=480)
    assert shapes == [640, (480, 640, 3)]


def test_finish_runs_even_when_the_agent_crashes_mid_run(fake_connect):
    finished = []

    class Crash(Agent):
        def __init__(self):
            self.n = 0

        def act(self, obs):
            self.n += 1
            if self.n > 2:  # past the pre-flight and the first real frame
                raise RuntimeError("boom")
            return Action.FORWARD

        def finish(self, result):
            finished.append(result)

    with pytest.raises(RuntimeError, match="boom"):
        run(Crash(), "vns_c1_x", viewer=False, quiet=True)
    assert finished == [None]


def test_panel_and_hud_defaults():
    agent = Agent()
    assert agent.hud() == []
    assert agent.panel() is None
    with pytest.raises(NotImplementedError):
        agent.act(None)  # type: ignore[arg-type]
    assert isinstance(np.zeros(1), np.ndarray)


def test_wait_polls_without_stepping(fake_connect):
    class Hesitant(Agent):
        def __init__(self):
            self.calls = 0

        def act(self, obs):
            self.calls += 1
            if self.calls < 5:
                return WAIT
            return Action.CHECKIN if obs.step else Action.FORWARD

    agent = Hesitant()
    run(agent, "vns_c1_x", viewer=False, check=False, quiet=True)
    assert fake_connect[0].calls == [(1, 1)]
    assert agent.calls == 6


def test_a_long_wait_burns_one_idle_tick_to_stay_alive(fake_connect, monkeypatch):
    from vis_nav_sdk import agent as agent_module

    clock = [0.0]
    monkeypatch.setattr(agent_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(agent_module.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + 30))

    class Frozen(Agent):
        def __init__(self):
            self.calls = 0

        def act(self, obs):
            self.calls += 1
            return WAIT if self.calls < 9 else Action.CHECKIN

    run(Frozen(), "vns_c1_x", viewer=False, check=False, quiet=True)
    # Eight waits of 30 simulated seconds: keepalives at 90 s and 180 s, nothing else.
    assert fake_connect[0].calls == [(0, 1), (0, 1)]
