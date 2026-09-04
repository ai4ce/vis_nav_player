"""
vis-nav-sdk: the client for the visual navigation challenge.

Two ways in. The callback interface, which is what most students want::

    from vis_nav_sdk import Agent, Action, run

    class Straight(Agent):
        def act(self, obs):
            return Action.FORWARD, 4

    run(Straight(), CHALLENGE_ID)

``run()`` checks the challenge with you first -- whether a run is already going, whether to
spend an attempt -- then prints and opens the page where the run can be followed, and
reminds you where to submit your report. The imperative interface underneath it asks
nothing::

    from vis_nav_sdk import connect, Action

    with connect(CHALLENGE_ID) as session:
        obs = session.initial_observation
        obs = session.step(Action.FORWARD, repeat=4)
        print(session.checkin())

``api_key`` may be omitted in favour of ``$VIS_NAV_API_KEY``; the server defaults to the
course server and may be overridden with ``server=`` or ``$VIS_NAV_SERVER``.
"""

from __future__ import annotations

__version__ = "0.3.0"

from .agent import WAIT, Agent, preflight, run
from .config import DEFAULT_SERVER
from .errors import ConfigError, SessionClosed, SimError
from .protocol import PROTOCOL_VERSION, Action
from .rest import Client
from .session import (
    Camera,
    Limits,
    Observation,
    Result,
    Session,
    SessionInfo,
    connect,
    redeem,
)
from .telemetry import StepTiming, Telemetry

__all__ = [
    "DEFAULT_SERVER",
    "PROTOCOL_VERSION",
    "Action",
    "Agent",
    "Camera",
    "Client",
    "ConfigError",
    "Limits",
    "Observation",
    "Result",
    "Session",
    "SessionClosed",
    "SessionInfo",
    "SimError",
    "StepTiming",
    "Telemetry",
    "WAIT",
    "__version__",
    "connect",
    "preflight",
    "redeem",
    "run",
]
