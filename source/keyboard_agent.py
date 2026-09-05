"""
Drive the robot yourself.

    arrows      move; hold two for an arc (up + left)
    space       check in -- "I am at the goal"; scores the run and ends it
    escape      quit without a score

The window shows the camera, the four views from the goal, and what each step costs.
"""

from __future__ import annotations

import cli
from vis_nav_sdk import WAIT, Action, Agent, Observation, run

KEYMAP = {
    "up": Action.FORWARD,
    "down": Action.BACKWARD,
    "left": Action.LEFT,
    "right": Action.RIGHT,
}


class KeyboardAgent(Agent):
    """Reads the keyboard through the viewer window, one tick per frame while a key is
    held. ``run()`` paces the loop, so the robot moves at the same speed on any link."""

    def act(self, obs: Observation) -> Action | tuple[Action, int]:
        if self.viewer is None:
            raise RuntimeError("KeyboardAgent needs the viewer window; run with viewer=True")
        keys = self.viewer.keys()
        if "space" in keys:
            return Action.CHECKIN
        if "escape" in keys:
            return Action.QUIT
        action = Action.IDLE
        for name, bit in KEYMAP.items():
            if name in keys:
                action |= bit
        return WAIT if action == Action.IDLE else (action, 1)

    def hud(self) -> list[str]:
        return ["arrows: move   space: check in   esc: quit"]


if __name__ == "__main__":
    args = cli.parser(__doc__).parse_args()
    run(KeyboardAgent(), cli.challenge(args), **cli.run_options(args))
