"""
Your agent. Fill in the methods below, then:

    uv run source/my_agent.py --challenge <id>

Everything your agent may use is on this list; there is nothing else.

    exploration data   images and action labels from earlier drives through this maze
                       (cli.exploration_data downloads them into data/<challenge>/)
    info.targets       four views from the goal, facing front, left, back, right
    info.camera        image size and intrinsic matrix
    info.limits        step budget and attempt count
    obs.image          the camera frame, (240, 320, 3) uint8, BGR
    obs.step / obs.steps_left

The robot's pose, the map and the goal position live on the server and are never sent.
"""

from __future__ import annotations

from pathlib import Path

import cli
import numpy as np
from vis_nav_sdk import Action, Agent, Observation, Result, SessionInfo, run


class MyAgent(Agent):
    def __init__(self, data_dir: Path) -> None:
        # Runs before anything touches the server: load the exploration data from
        # `data_dir`, build your index, warm up your model. A crash here costs no attempt.
        self.data_dir = data_dir

    def setup(self, info: SessionInfo) -> None:
        # Once per session, with the goal views in `info.targets`. Keep it cheap: the
        # session clock is already running. Called twice per run (once with random frames
        # as a pre-flight check), so derive from `info` rather than caching across calls.
        self.targets = info.targets

    def act(self, obs: Observation) -> Action | tuple[Action, int]:
        # Every step. Return one of:
        #   Action.FORWARD / BACKWARD / LEFT / RIGHT, combined with | for an arc
        #   (Action.FORWARD, 4)   hold it for 4 ticks in one round trip
        #   Action.CHECKIN        "I am at the goal": scores the run and ends it
        #   Action.QUIT           give up, unscored
        raise NotImplementedError("decide what to do with obs.image")

    def finish(self, result: Result | None) -> None:
        # After the session ended. `result` is None if you quit or ran out of steps.
        pass

    def hud(self) -> list[str]:
        # Optional: extra lines in the viewer's status column, e.g. your position estimate.
        return []

    def panel(self) -> np.ndarray | list | None:
        # Optional: an image, or a list of (image, label) tiles, shown under the camera.
        return None


if __name__ == "__main__":
    parser = cli.parser(__doc__)
    parser.add_argument("--data", default=None, help="exploration data directory")
    args = parser.parse_args()
    data = cli.exploration_data(args, args.data)
    run(MyAgent(data), cli.challenge(args), **cli.run_options(args))
