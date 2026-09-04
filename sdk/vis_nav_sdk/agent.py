"""
The callback interface: subclass :class:`Agent`, implement :meth:`Agent.act`, hand it to
:func:`run`. The imperative :class:`~vis_nav_sdk.session.Session` is the primitive; this is
the loop most students want on top of it.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from . import protocol as P
from .errors import SimError
from .session import (
    Action,
    Camera,
    Limits,
    Observation,
    Result,
    SessionInfo,
    connect,
)

if TYPE_CHECKING:
    from .viewer import Viewer

WAIT: Final = (P.Action.IDLE, 0)
"""Return this from ``act()`` to be called again without stepping. Unlike ``Action.IDLE``,
which burns a tick, waiting costs nothing -- the viewer keeps refreshing and the session
stays open, subject to the gateway's idle timeout."""

Tile = "tuple[np.ndarray, str] | tuple[np.ndarray, str, tuple[int, int, int]]"
"""One image for :meth:`Agent.panel`: ``(bgr_image, label)`` or ``(bgr_image, label, rgb)``."""

VIEWER_FPS: Final = 30.0
"""How fast :func:`run` loops when a window is open. Without it, a local server answers in
a few milliseconds and a held key would race the robot across the maze; headless runs are
not paced at all."""

KEEPALIVE_S: Final = 90.0
"""How long :func:`run` lets an agent wait before burning one IDLE tick so the gateway's
120 s idle timeout does not end the session. One step per 90 s is noise against a run."""


class Agent:
    """One navigation policy.

    Lifecycle, per :func:`run`::

        __init__     your own work: load the exploration data, build indexes. Nothing
                     here touches the server, so a crash costs no attempt.
        setup(info)  once the session is open: the four target views, the camera, the
                     step budget. Cheap work only -- the session clock is running.
        act(obs)     every step. Return an Action, or (Action, repeat) to hold it for
                     several ticks in one round trip. Action.CHECKIN scores the run and
                     ends it; Action.QUIT ends it unscored; WAIT asks to be called again
                     with the same observation without stepping.
        finish(res)  after the session ended, with the Result or None.

    :meth:`hud` and :meth:`panel` feed the viewer window and are optional. :attr:`viewer`
    is set by :func:`run` before anything else is called, and is ``None`` when headless.
    """

    viewer: Viewer | None = None

    def setup(self, info: SessionInfo) -> None:
        pass

    def act(self, obs: Observation) -> Action | tuple[Action, int]:
        raise NotImplementedError("your Agent must implement act()")

    def finish(self, result: Result | None) -> None:
        pass

    def hud(self) -> list[str]:
        """Extra lines for the viewer's status column."""
        return []

    def panel(self) -> np.ndarray | list[Any] | None:
        """Shown under the camera view: either one BGR image, or a list of tiles --
        ``(image, label)`` or ``(image, label, (r, g, b))`` -- that the viewer lays out
        and labels in its own style."""
        return None


def decide(decision: Any, default_repeat: int) -> tuple[int, int]:
    """Normalise what ``act()`` returned into ``(action bits, repeat)``."""
    if isinstance(decision, tuple):
        if len(decision) != 2:
            raise SimError(f"act() must return an Action or (Action, repeat), got {decision!r}")
        action, repeat = decision
    else:
        action, repeat = decision, default_repeat
    try:
        action, repeat = int(action), int(repeat)
    except (TypeError, ValueError) as exc:
        raise SimError(
            f"act() must return an Action or (Action, repeat), got {decision!r}"
        ) from exc
    if repeat < 0:
        raise SimError(f"repeat must be >= 0, got {repeat}")
    return action, repeat


def preflight(agent: Agent, *, width: int = P.CAMERA_WIDTH, height: int = P.CAMERA_HEIGHT) -> None:
    """Exercise the agent with random frames before any session is opened.

    Attempts are a quota, so an agent that raises on its first frame would cost one.
    Failing here costs nothing. ``setup()`` is therefore called twice per run -- once with
    junk, once for real -- so anything it derives from the targets must be recomputed
    rather than cached.
    """
    rng = np.random.default_rng()

    def frame() -> np.ndarray:
        return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)

    fx = round(width / 2 / np.tan(np.radians(60)))
    info = SessionInfo(
        session_id="preflight",
        challenge={"id": "preflight", "name": "preflight"},
        camera=Camera(
            width=width,
            height=height,
            color_order=P.COLOR_ORDER,
            encoding="raw",
            intrinsic_matrix=np.array(
                [[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]], dtype=np.float64
            ),
        ),
        limits=Limits(
            max_steps=P.DEFAULT_MAX_STEPS,
            max_repeat=P.MAX_REPEAT,
            attempts_used=0,
            attempts_allowed=None,
            idle_timeout_s=P.DEFAULT_IDLE_TIMEOUT_S,
            session_timeout_s=P.DEFAULT_SESSION_TIMEOUT_S,
        ),
        targets=[frame() for _ in range(P.TARGET_COUNT)],
    )
    try:
        agent.setup(info)
        decision = agent.act(Observation(frame(), step=0, steps_left=P.DEFAULT_MAX_STEPS))
        decide(decision, 1)
        agent.hud()
        agent.panel()
    except SimError:
        raise
    except Exception as exc:
        raise SimError(
            "your Agent raised during the pre-flight check, before any session was "
            f"opened, so no attempt was used: {type(exc).__name__}: {exc}"
        ) from exc


def run(
    agent: Agent,
    token: str | None = None,
    *,
    server: str | None = None,
    repeat: int = 1,
    viewer: Viewer | bool | None = None,
    fps: float | None = None,
    check: bool = True,
    quiet: bool = False,
    **connect_kwargs: Any,
) -> Result | None:
    """Drive ``agent`` through the session ``token`` reserved (from Start on the challenge
    page; falls back to ``$VIS_NAV_SESSION``). Returns the :class:`Result` if it checked in,
    ``None`` if it quit or ran out of budget.

    ``repeat`` is the tick count used when ``act()`` returns a bare ``Action``.
    ``viewer`` opens a window showing the camera, the targets and per-step latency:
    ``None`` opens one if pygame is installed, ``True`` insists, ``False`` runs headless.
    ``fps`` caps how often ``act()`` is called: ``None`` means :data:`VIEWER_FPS` with a
    window and unlimited without. ``check=False`` skips the pre-flight.
    """
    from .viewer import open_viewer

    window: Viewer | None = open_viewer(viewer)
    agent.viewer = window
    if fps is None and window is not None:
        fps = VIEWER_FPS
    period = 1 / fps if fps else 0.0
    say = (lambda *_: None) if quiet else print
    result: Result | None = None
    try:
        if check:
            preflight(agent)

        with connect(token, server=server, **connect_kwargs) as session:
            info = session.info
            say(
                f"session {info.session_id} on {info.challenge.get('name', '?')!r}: "
                f"{info.limits.max_steps} steps, attempt "
                f"{info.limits.attempts_used}"
                + (
                    f"/{info.limits.attempts_allowed}"
                    if info.limits.attempts_allowed is not None
                    else ""
                )
            )
            agent.setup(info)
            if window is not None:
                window.attach(info)

            obs = session.initial_observation
            assert obs is not None
            reason = None
            last_step_at = time.monotonic()
            next_frame = time.monotonic()
            while True:
                if window is not None:
                    window.update(obs, session.telemetry, agent)
                    if window.closed:
                        reason = "viewer window closed"
                        break
                now = time.monotonic()
                if now < next_frame:
                    time.sleep(next_frame - now)
                next_frame = max(next_frame + period, time.monotonic() - period)
                action, ticks = decide(agent.act(obs), repeat)
                if ticks == 0 and not action & int(Action.CHECKIN | Action.QUIT):
                    if time.monotonic() - last_step_at < KEEPALIVE_S:
                        if not period:
                            time.sleep(0.02)
                        continue
                    action, ticks = int(Action.IDLE), 1
                if action & int(Action.CHECKIN):
                    result = session.checkin()
                    break
                if action & int(Action.QUIT):
                    reason = "agent quit"
                    break
                if obs.steps_left <= 0:
                    reason = "step budget reached"
                    break
                ticks = max(1, min(ticks, obs.steps_left, info.limits.max_repeat))
                obs = session.step(action, ticks)
                last_step_at = time.monotonic()

            if result is None:
                session.abort(reason or "")
                say(f"session ended unscored: {reason}")
            else:
                say(
                    f"{result.goal_tier}: {result.trans_error:.3f} m from the goal in "
                    f"{result.nav_steps} steps"
                )
            t = session.telemetry.summary()
            if t["steps"]:
                say(
                    f"{t['steps']} round trips, median rtt {t['rtt_ms_p50']:.1f} ms "
                    f"(server {_fmt(t['server_ms_p50'])}, network {_fmt(t['network_ms_p50'])}), "
                    f"your code {_fmt(t['think_ms_p50'])} per step"
                )
            if window is not None:
                window.hold(
                    f"{result.goal_tier.upper()}  ·  {result.trans_error:.2f} m from the goal"
                    f"  ·  {result.nav_steps:,} steps"
                    if result is not None
                    else f"ENDED  ·  {reason}"
                )
    finally:
        try:
            agent.finish(result)
        finally:
            if window is not None:
                window.close()
    return result


def _fmt(ms: float | None) -> str:
    return "n/a" if ms is None else f"{ms:.1f} ms"


__all__ = ["KEEPALIVE_S", "VIEWER_FPS", "WAIT", "Agent", "decide", "preflight", "run"]
